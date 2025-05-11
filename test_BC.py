import gymnasium as gym
import torch
import random
import numpy as np
import argparse
import os
import warnings
import wandb
#from stable_baselines3 import SAC
from sac_v2 import PolicyNetwork
from MPC_DM_model import ReplayBuffer, MPC_DM
from MPC import ReplayBuffer_traj, MPC
from stable_baselines3 import SAC

target_env = "HalfCheetah-3legs"
n_traj=10
hidden_dim = 256
env = "reacher"

if env == "reacher":
    source_env = "Reacher-v4"
    target_env = "Reacher-3joints"
    source_s_dim = 11
    source_a_dim = 2
    target_s_dim = 14
    target_a_dim = 3
elif env == "cheetah":
    source_env = "HalfCheetah-v4"
    target_env = "HalfCheetah-3legs"
    source_s_dim = 18
    source_a_dim = 6
    target_s_dim = 23
    target_a_dim = 9

action_range = 10.0 if env=="reacher" else 1.0

# expert = SAC.load('models/cheetah/seed_7/HalfCheetah-3legs_SAC_3_128_200000_2.zip', device='cuda') # SB3
expert = PolicyNetwork(target_s_dim, target_a_dim, hidden_dim, action_range, 'cuda').to('cuda')
expert.load_state_dict(torch.load( 'models/reacher/seed_7/7_reacher_target.pth', map_location='cuda',weights_only=True ))
mpc_dm = MPC_DM(target_s_dim, target_a_dim, 'cuda')
# mpc_dm.mpc_policy_net.load_state_dict(torch.load( 'models_multiple/2_MPCModel.pth', map_location='cuda',weights_only=True))
# mpc_dm.dynamic_model.load_state_dict(torch.load( f'models_test/7_DynamicModel.pth', weights_only=True, map_location='cuda' ))
mpc_dm.mpc_policy_net.load_state_dict(torch.load( 'models/reacher/seed_7/expert_ratio_1.0/7_MPCModel.pth', map_location='cuda',weights_only=True))
mpc_dm.dynamic_model.load_state_dict(torch.load( f'models/reacher/seed_7/expert_ratio_1.0/7_DynamicModel.pth', weights_only=True, map_location='cuda' ))

env_target = gym.make(target_env)
max_episode_steps = env_target.spec.max_episode_steps 


total_score = 0
for episode in range(int(n_traj)):
    score = 0
    state, _ = env_target.reset(seed=3*episode) ############################################
    for _ in range(max_episode_steps):
        # action, _ = expert.predict(state, deterministic=True) # SB3
        # action = mpc_dm.mpc_policy_net(torch.tensor(state, device='cuda', dtype=torch.float))
        action = expert.get_action(state, deterministic=True)

        next_state, reward, terminated, truncated, _ = env_target.step(action.tolist())
        done = truncated or terminated

        done_mask = 0.0 if done else 1.0

        state = next_state
        score += reward
        
        if done: break

    print(f'{score:.0f}')
    total_score += score

print(f'avg. score: {total_score/int(n_traj):.0f}')
env_target.close()