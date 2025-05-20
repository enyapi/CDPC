import gymnasium as gym
import torch
import random
import numpy as np
import argparse
import os
import wandb
from sac_v2 import PolicyNetwork
from MPC_DM_model import ReplayBuffer, MPC_DM
from MPC import ReplayBuffer_traj, MPC
from stable_baselines3 import SAC

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  

parser = argparse.ArgumentParser()
parser.add_argument("MPC_pre_ep", type=int, nargs='?', default=1000)
parser.add_argument("decoder_batch", type=int, nargs='?', default=32)
parser.add_argument("--seed", type=int, nargs='?', default=7)
parser.add_argument("--n_traj", type=int, nargs='?', default=5) # 1000/10000
parser.add_argument("--expert_ratio", type=float, nargs='?', default=0.2) # random_ratio=1-expert_ratio
parser.add_argument("--decoder_ep", type=int, nargs='?', default=500) # 500/200
parser.add_argument("--device", type=str, nargs='?', default="cuda")
parser.add_argument("--env", type=str, nargs='?', default="cheetah") # cheetah reacher
parser.add_argument("--wandb", action='store_true', default=True)
parser.add_argument("--use_flow", action='store_true', default=False)

args = parser.parse_args()
seed_everything(args.seed)

if args.env == "reacher":
    source_env = "Reacher-v4"
    target_env = "Reacher-3joints"
    source_s_dim = 11
    source_a_dim = 2
    target_s_dim = 14
    target_a_dim = 3
elif args.env == "cheetah":
    source_env = "HalfCheetah-v4"
    target_env = "HalfCheetah-3legs"
    source_s_dim = 18
    source_a_dim = 6
    target_s_dim = 23
    target_a_dim = 9

if args.wandb:
    wandb.init(project="cdpc", name = f'BC training', tags=["cdpc"])
location = f'./models/{args.env}/seed_{str(args.seed)}/'
mpc_location = f'{location}/expert_ratio_{args.expert_ratio}/'
hidden_dim = 256
action_range = 10.0 if args.env=="reacher" else 1.0

# agent_target = SAC.load('models/cheetah/seed_7/HalfCheetah-3legs_SAC_3_128_200000_2.zip', device=args.device) # SB3
agent_target = PolicyNetwork(target_s_dim, target_a_dim, hidden_dim, action_range, args.device).to(args.device)
agent_target.load_state_dict(torch.load( f'{location}{str(args.seed)}_{args.env}_target.pth', weights_only=True, map_location=args.device ))
agent_target_medium = PolicyNetwork(target_s_dim, target_a_dim, hidden_dim, action_range, args.device).to(args.device)
agent_target_medium.load_state_dict(torch.load( f'{location}{str(args.seed)}_{args.env}_target_medium.pth', weights_only=True, map_location=args.device ))

# train_set, buffer, buffer_expert_only = collect_target_data(agent_target, agent_target_medium, target_env, args.n_traj, args.expert_ratio, args.device, args.seed)

target_buffer_path = f'./train_set/{str(args.seed)}_{args.env}_{args.expert_ratio}_target_buffer_expert.pkl'

import pickle
with open(target_buffer_path, 'rb') as f:
    buffer = pickle.load(f)

mpc_dm = MPC_DM(target_s_dim, target_a_dim, args.device)

batch_size=128
print('start training...')

for i in range(args.MPC_pre_ep):
    loss_mpc, loss_dm = mpc_dm.update(batch_size, buffer)
    if args.wandb:
        wandb.log({"mpc_dm episode": i, "train/loss_mpc": loss_mpc, "train/loss_dm": loss_dm,})

os.makedirs(mpc_location, exist_ok=True)
torch.save(mpc_dm.mpc_policy_net.state_dict(), f'{mpc_location}/{str(args.seed)}_MPCModel.pth')
torch.save(mpc_dm.dynamic_model.state_dict(), f'{mpc_location}/{str(args.seed)}_DynamicModel.pth')

print(f'{mpc_location}/{str(args.seed)}_DynamicModel.pth saved')