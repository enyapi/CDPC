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

def collect_target_data(agent_target, agent_target_medium, target_env, n_traj, expert_ratio, device, seed):
    buffer_maxlen = 1000000
    buffer = ReplayBuffer(buffer_maxlen, device)
    buffer_expert_only = ReplayBuffer(buffer_maxlen, device)
    train_set = ReplayBuffer_traj()

    env_target = gym.make(target_env)
    max_episode_steps = env_target.spec.max_episode_steps 
    for episode in range(int(n_traj)):
        score = 0
        state, _ = env_target.reset(seed=seed*episode) ############################################
        state_list = []
        next_state_list = []
        for _ in range(max_episode_steps):
            # action, _ = agent_target.predict(state, deterministic=True) # SB3
            # if episode % 2 == 0:
            #     action = agent_target.get_action(state, deterministic=True)
            # else:
            #     action = agent_target_medium.get_action(state, deterministic=True)
            action = agent_target.get_action(state, deterministic=True)
            
            next_state, reward, terminated, truncated, _ = env_target.step(action)
            done = truncated or terminated

            done_mask = 0.0 if done else 1.0
            buffer.push((state, action, reward, next_state, done_mask))

            if episode < int(n_traj*expert_ratio):
                buffer_expert_only.push((state, action, reward, next_state, done_mask))

            state_list.append(state)
            next_state_list.append(next_state)
            state = next_state
            score += reward
            
            if done: break

        # if truncated:
        #     action = agent_target.get_action(state, deterministic=True)
        #     score += target_Q(torch.tensor(state[np.newaxis, :], dtype=torch.float).to('cuda'), torch.tensor(action[np.newaxis, :], dtype=torch.float).to('cuda'))
        
        train_set.push(score, state_list, next_state_list)
        print("episode:{}, Return:{}, buffer_capacity:{}".format(episode, score, buffer.buffer_len()))
    env_target.close()
    
    print(f"Collected {n_traj} trajectories.")
    print(f"Collected {n_traj*max_episode_steps} transitions.")
    return train_set, buffer, buffer_expert_only

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

train_set, buffer, buffer_expert_only = collect_target_data(agent_target, agent_target_medium, target_env, args.n_traj, args.expert_ratio, args.device, args.seed)

# import pickle
# with open('train_set/7_cheetah_1.0_buffer.pkl', 'rb') as f:
#     buffer = pickle.load(f)

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