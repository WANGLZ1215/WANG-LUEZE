"""
PPO on LunarLanderContinuous‑v3
--------------------------------
‣ 稳定收敛到 Avg100 ≥ +200
‣ 终端实时打印 + TensorBoard 可视化
"""

import gymnasium as gym
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import random, time, math, os, pathlib

# ──────────────────── 超参数 ────────────────────
ENV_ID            = "LunarLanderContinuous-v3"
TOTAL_STEPS       = 30_000_000           # 总采样步数
ROLLOUT_STEPS     = 1024 #1024 #2048
MINI_BATCH_SIZE   = 128  #128 #256
UPDATE_EPOCHS     = 10
GAMMA             = 0.99
GAE_LAMBDA        = 0.95
CLIP_EPS          = 0.2
ENT_COEF          = 0.01
LR_ACTOR          = 3e-4 #4e-4 #1e-4
LR_CRITIC         = 2e-4
MAX_GRAD_NORM     = 0.5
TARGET_REWARD     = 200
SEED              = 42
# ────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pathlib.Path("runs").mkdir(exist_ok=True)

# ---------- util: running mean / var ----------
class RunningMeanStd:
    def __init__(self, shape):
        self.mean = np.zeros(shape, np.float32)
        self.var  = np.ones(shape,  np.float32)
        self.count = 1e-4

    def update(self, x: np.ndarray):
        batch_mean = x.mean(axis=0)
        batch_var  = x.var(axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        self.mean += delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2  = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

# ---------- Networks ----------
def orth_init(layer, std=1.0):
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, 0)

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.Tanh(),
            nn.Linear(256, 256),     nn.Tanh(),
            nn.Linear(256, 128),     nn.Tanh()
        )
        self.mu = nn.Linear(128, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                orth_init(m, std=math.sqrt(2))
        orth_init(self.mu, std=0.01)

    def forward(self, x):
        x = self.net(x)
        mean = torch.tanh(self.mu(x))          # [-1,1]
        std  = torch.exp(self.log_std)
        return Normal(mean, std)

class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.Tanh(),
            nn.Linear(256, 256),     nn.Tanh(),
            nn.Linear(256, 1)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                orth_init(m, std=math.sqrt(2))

    def forward(self, x):
        return self.net(x).squeeze(-1)

# ---------- set seed ----------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ---------- Environment ----------
env = gym.make(ENV_ID)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

# ---------- Agent ----------
actor  = Actor(obs_dim, act_dim).to(device)
critic = Critic(obs_dim).to(device)
opt_actor  = optim.Adam(actor.parameters(),  lr=LR_ACTOR,  eps=1e-5)
opt_critic = optim.Adam(critic.parameters(), lr=LR_CRITIC, eps=1e-5)

obs_rms = RunningMeanStd(obs_dim)
writer  = SummaryWriter(f"runs/PPO_clean_{time.strftime('%Y%m%d_%H%M%S')}")

# ---------- Buffer ----------
class Buffer:
    def __init__(self, size, obs_dim, act_dim):
        self.obs  = np.zeros((size, obs_dim), np.float32)
        self.act  = np.zeros((size, act_dim), np.float32)
        self.rew  = np.zeros(size, np.float32)
        self.done = np.zeros(size, np.float32)
        self.logp = np.zeros(size, np.float32)
        self.val  = np.zeros(size, np.float32)
        self.ptr = 0; self.max_size = size

    def store(self, obs, act, rew, done, logp, val):
        self.obs[self.ptr]  = obs
        self.act[self.ptr]  = act
        self.rew[self.ptr]  = rew
        self.done[self.ptr] = done
        self.logp[self.ptr] = logp
        self.val[self.ptr]  = val
        self.ptr += 1

    def reset(self):
        self.ptr = 0

buf = Buffer(ROLLOUT_STEPS, obs_dim, act_dim)

# ---------- helper ----------
def compute_gae(rewards, values, dones, last_val):
    adv = np.zeros_like(rewards)
    gae = 0
    for t in reversed(range(len(rewards))):
        next_val = last_val if t == len(rewards)-1 else values[t+1]
        delta = rewards[t] + GAMMA * (1 - dones[t]) * next_val - values[t]
        gae   = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * gae
        adv[t] = gae
    ret = adv + values
    return adv, ret

# ---------- training loop ----------
episode, ep_ret, ep_len = 0, 0, 0
obs, _ = env.reset(seed=SEED)
global_step = 0
reward_queue = deque(maxlen=100)
best_avg = -np.inf
t0 = time.time()

while global_step < TOTAL_STEPS:
    # ── 收集一段轨迹 ───────────────────────────────
    buf.reset()
    for _ in range(ROLLOUT_STEPS):
        obs_norm = obs_rms.normalize(obs)
        obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            dist  = actor(obs_t)
            action = dist.sample()
            logp   = dist.log_prob(action).sum(-1)
            value  = critic(obs_t)

        act_np = action.cpu().numpy()[0]
        next_obs, reward, done, trunc, _ = env.step(act_np)
        done_flag = done or trunc

        buf.store(obs_norm, act_np, reward, done_flag, logp.item(), value.item())

        obs_rms.update(obs.reshape(1, -1))
        obs, ep_ret, ep_len = next_obs, ep_ret + reward, ep_len + 1
        global_step += 1

        if done_flag:
            reward_queue.append(ep_ret)

            # ① 先算 avg100
            avg100 = np.mean(reward_queue) if reward_queue else -np.inf

            # ② 再写所有与奖励相关的日志
            writer.add_scalar("Reward/Episode", ep_ret, global_step)
            writer.add_scalar("Reward/Avg100", avg100, global_step)

            # ③ 其余逻辑保持不变
            ep_ret = 0
            episode += 1
            obs, _ = env.reset()

    # ── GAE & returns ─────────────────────────────
    with torch.no_grad():
        last_val = critic(torch.tensor(obs_rms.normalize(obs),
                                       dtype=torch.float32, device=device).unsqueeze(0)).item()

    adv, ret = compute_gae(buf.rew, buf.val, buf.done, last_val)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    # ── 转 tensor ─────────────────────────────────
    obs_t  = torch.tensor(buf.obs, dtype=torch.float32, device=device)
    act_t  = torch.tensor(buf.act, dtype=torch.float32, device=device)
    logp_t = torch.tensor(buf.logp, dtype=torch.float32, device=device)
    adv_t  = torch.tensor(adv, dtype=torch.float32, device=device)
    ret_t  = torch.tensor(ret, dtype=torch.float32, device=device)

    # ── PPO 更新 ─────────────────────────────────
    policy_losses, value_losses, entropies, clip_fracs = [], [], [], []
    for _ in range(UPDATE_EPOCHS):
        idx = np.random.permutation(ROLLOUT_STEPS)
        for start in range(0, ROLLOUT_STEPS, MINI_BATCH_SIZE):
            mb = idx[start:start+MINI_BATCH_SIZE]

            dist     = actor(obs_t[mb])
            new_logp = dist.log_prob(act_t[mb]).sum(-1)
            entropy  = dist.entropy().sum(-1).mean()

            ratio    = (new_logp - logp_t[mb]).exp()
            surr1    = ratio * adv_t[mb]
            surr2    = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * adv_t[mb]
            pg_loss  = -torch.min(surr1, surr2).mean()

            value_pred = critic(obs_t[mb])
            v_loss = 0.5 * (value_pred - ret_t[mb]).pow(2).mean()

            loss = pg_loss + v_loss - ENT_COEF * entropy

            clipped = ((ratio < 1-CLIP_EPS) | (ratio > 1+CLIP_EPS)).float().mean().item()

            policy_losses.append(pg_loss.item())
            value_losses.append(v_loss.item())
            entropies.append(entropy.item())
            clip_fracs.append(clipped)

            opt_actor.zero_grad(); opt_critic.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(),  MAX_GRAD_NORM)
            nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD_NORM)
            opt_actor.step(); opt_critic.step()

    # ── 日志 & 打印 ────────────────────────────────
    avg100 = np.mean(reward_queue) if reward_queue else -np.inf
    writer.add_scalar("Reward/Avg100", avg100, global_step)
    writer.add_scalar("Loss/Policy",  np.mean(policy_losses), global_step)
    writer.add_scalar("Loss/Value",   np.mean(value_losses),  global_step)
    writer.add_scalar("Info/Entropy", np.mean(entropies),     global_step)
    writer.add_scalar("Info/ClipFrac",np.mean(clip_fracs),    global_step)

    print(f"Step {global_step:7d} | Ep {episode:4d} | "
          f"Reward {reward_queue[-1] if reward_queue else 0:7.1f} | "
          f"Avg100 {avg100:7.1f} | "
          f"P_Loss {np.mean(policy_losses):7.3f} | "
          f"V_Loss {np.mean(value_losses):7.3f} | "
          f"Ent {np.mean(entropies):5.3f} | "
          f"ClipFrac {np.mean(clip_fracs):4.2%}")

    # ── 保存最佳 & 早停 ────────────────────────────
    if avg100 > best_avg:
        best_avg = avg100
        torch.save({
            "actor":  actor.state_dict(),
            "critic": critic.state_dict(),
            "rms_mean": obs_rms.mean,
            "rms_var":  obs_rms.var
        }, "ppo_lander_best.pth")

    if avg100 >= TARGET_REWARD and len(reward_queue) == 100:
        print(f"Solved! Avg100 reward {avg100:.1f} at step {global_step}")
        break

writer.close()
env.close()

elapsed = time.time() - t0
print(f"\n🏁 Training finished in {elapsed/60:.1f} minutes "
      f"({elapsed:.1f} seconds)")
