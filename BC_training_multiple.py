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
import pickle

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  

def load_buffer(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
def split_buffer(buffer:ReplayBuffer, num_BC, device):
    buffer_maxlen = 1000000
    sub_buffers = [ReplayBuffer(buffer_maxlen, device) for i in range(num_BC)]
    
    for i in range(len(buffer.buffer)):
        sub_buffers[i % num_BC].push(buffer.buffer[i])

    return sub_buffers

parser = argparse.ArgumentParser()
parser.add_argument("MPC_pre_ep", type=int, nargs='?', default=1000)
parser.add_argument("decoder_batch", type=int, nargs='?', default=32)
parser.add_argument("--seed", type=int, nargs='?', default=2)
parser.add_argument("--n_traj", type=int, nargs='?', default=25) # 1000/10000
parser.add_argument("--expert_ratio", type=float, nargs='?', default=1.0) # random_ratio=1-expert_ratio
parser.add_argument("--decoder_ep", type=int, nargs='?', default=500) # 500/200
parser.add_argument("--device", type=str, nargs='?', default="cuda")
parser.add_argument("--env", type=str, nargs='?', default="cheetah") # cheetah reacher
parser.add_argument("--wandb", action='store_true', default=True)
parser.add_argument("--use_flow", action='store_true', default=False)
parser.add_argument("--num_BC", type=int, default=10)

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


location = f'./models/{args.env}/seed_{str(args.seed)}/'
mpc_location = f'{location}/expert_ratio_{args.expert_ratio}/'
hidden_dim = 256
action_range = 10.0 if args.env=="reacher" else 1.0


os.makedirs('./train_set/', exist_ok=True)
target_buffer_path = f'./train_set/{str(args.seed)}_{args.env}_{args.expert_ratio}_target_buffer.pkl'

if os.path.exists(target_buffer_path):
    buffer = load_buffer(target_buffer_path)
buffers = split_buffer(buffer, args.num_BC, args.device)

num_BC = args.num_BC

mpc_dms = [MPC_DM(target_s_dim, target_a_dim, args.device) for i in range(num_BC)]
mpc_dm_for_transition = MPC_DM(target_s_dim, target_a_dim, args.device)

batch_size=512
print('start training...')

for i in range(args.MPC_pre_ep):
    for ith in range(num_BC):
        loss_mpc, loss_dm = mpc_dms[ith].update(batch_size, buffers[ith])
        mpc_dm_for_transition.update(batch_size, buffers[ith])
    print(f'epoch {i} done.')

os.makedirs('./models_multiple/', exist_ok=True)
os.makedirs(f'./models_multiple/{args.env}_{str(args.seed)}', exist_ok=True)

for ith in range(num_BC):
    torch.save(mpc_dms[ith].mpc_policy_net.state_dict(), f'models_multiple/{args.env}_{str(args.seed)}/{str(args.seed)}_MPCModel_{ith}.pth')
    torch.save(mpc_dms[ith].dynamic_model.state_dict(), f'models_multiple/{args.env}_{str(args.seed)}/{str(args.seed)}_DynamicModel_{ith}.pth')

torch.save(mpc_dm_for_transition.mpc_policy_net.state_dict(), f'models_multiple/{args.env}_{str(args.seed)}/{str(args.seed)}_MPCModel.pth')
torch.save(mpc_dm_for_transition.dynamic_model.state_dict(), f'models_multiple/{args.env}_{str(args.seed)}/{str(args.seed)}_DynamicModel.pth')

