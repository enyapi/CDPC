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
from MPC_DM_model import ReplayBuffer, MPC_DM, Dynamic_Model
from MPC_v2 import Action_Projector, State_Projector
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

expert_ratio = 0.8
device = 'cuda'
env = 'reacher'
seed = 2


mpc_dm = MPC_DM(target_s_dim, target_a_dim, 'cuda')
mpc_dm.mpc_policy_net.load_state_dict(torch.load( f'models/reacher/seed_{str(seed)}/expert_ratio_{expert_ratio}/{str(seed)}_MPCModel.pth', map_location='cuda',weights_only=True))
mpc_dm.dynamic_model.load_state_dict(torch.load( f'models/reacher/seed_{str(seed)}/expert_ratio_{expert_ratio}/{str(seed)}_DynamicModel.pth', weights_only=True, map_location='cuda' ))

action_projector = Action_Projector(target_s_dim, target_a_dim, source_a_dim).to(device)
state_projector = State_Projector(target_s_dim, source_s_dim).to(device)

action_projector.load_state_dict(torch.load(f'models/{env}/seed_{str(seed)}/expert_ratio_{expert_ratio}/{str(seed)}_action_projector_500.pth', weights_only=True, map_location='cuda'))
state_projector.load_state_dict(torch.load(f'models/{env}/seed_{str(seed)}/expert_ratio_{expert_ratio}/{str(seed)}_state_projector_500.pth', weights_only=True, map_location='cuda'))

source_dynamics_model = Dynamic_Model(source_s_dim + source_a_dim, source_s_dim).to(device)
source_dynamics_model.load_state_dict(torch.load( f'models/{env}/seed_{str(seed)}/expert_ratio_{expert_ratio}/{str(seed)}_DynamicModel_source.pth', weights_only=True, map_location=device ))

env_target = gym.make(target_env)
max_episode_steps = env_target.spec.max_episode_steps 


total_score = 0
for episode in range(int(n_traj)):
    score = 0
    state, _ = env_target.reset(seed=3*episode) ############################################
    state = torch.tensor(state, device='cuda', dtype=torch.float)

    dcc_errors = []
    for _ in range(max_episode_steps):
        # action, _ = expert.predict(state, deterministic=True) # SB3
        
        action = mpc_dm.mpc_policy_net(state)
        next_state, reward, terminated, truncated, _ = env_target.step(action.tolist())
        done = truncated or terminated

        done_mask = 0.0 if done else 1.0
        score += reward
        next_state = torch.tensor(next_state, device='cuda', dtype=torch.float)

        # dcc
        source_action = action_projector(state.unsqueeze(0), action.unsqueeze(0))
        source_state = state_projector(state.unsqueeze(0))
        source_next_state = state_projector(next_state.unsqueeze(0))
        dcc_source_next_state = source_dynamics_model(torch.cat([source_state, source_action], dim=-1))
        dcc_errors.append(torch.mean(torch.abs(source_next_state - dcc_source_next_state)).item())
        
        state = next_state
        if done: break

    print(f'dcc error: {sum(dcc_errors) / len(dcc_errors)}')
    print(f'{score:.0f}')
    total_score += score

print(f'avg. score: {total_score/int(n_traj):.0f}')
env_target.close()