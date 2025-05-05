# WANG-LUEZE
***PPO vs GRPO on LunarLanderContinuous-v3***

This repository contains two self-contained PyTorch implementations that train an actor-critic PPO and a critic-free GRPO agent on the classic control task LunarLanderContinuous-v3. The code is written for clarity and reproducibility:

ppo_lander.py – Proximal Policy Optimization with separate Actor & Critic (value function).

grpo_lander.py – Group-Relative Policy Optimization (baseline-normalized returns, no Critic).


# Requirements

python >= 3.8  
torch >= 2.0  
gymnasium[box2d] >= 0.29  
tensorboard  


# Key Observations

## 1.PPO consistently reaches the 200-reward solve threshold
Fastest run solved in ≈9 M steps; even smaller 2.5 M-step run climbs steadily.

## 2.GRPO is markedly less stable
All three GRPO runs exhibit rising policy entropy, high clip-fraction (60-85 %), and failing reward curves.
The critic-free design plus batch-normalised returns amplify variance once the lander’s shaped rewards turn sparse.

## 3.Entropy becomes the tell-tale signal
PPO entropy drops as the policy sharpens; GRPO entropy instead diverges upward after ~5 M steps, indicating exploding log-std parameters.

## 4.Clip-fraction mirrors the story
In PPO it settles ≈0.5; in GRPO it oscillates >0.7 and often saturates the clamp, throttling learning.

## 5.The convergence speed of PPO varies significantly depending on the parameters. However, GRPO will not converge even if the parameters are adjusted.

# References

Schulman et al., 2017 – “Proximal Policy Optimization Algorithms”

DeepSeek-R1 (2024) – introduces GRPO for language-model RLHF
