import torch
import numpy as np
import random
import torch.nn as nn
import collections
import torch.nn.functional as F
import torch.optim as optim
import argparse
import os
import sys
import gymnasium as gym
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sac_v2 import PolicyNetwork

def evaluate_policy(policy, env, seed, device):
    total_reward = 0.0
    state, _ = env.reset(seed=seed)
    for i in range(env.spec.max_episode_steps):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        action = policy(state).cpu().detach().numpy().squeeze(0)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
    return total_reward

class MPC_Policy_Net(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 64)
        # self.bn1 = nn.BatchNorm1d(64)
        self.linear2 = nn.Linear(64, 128)
        # self.bn2 = nn.BatchNorm1d(128)
        self.linear3 = nn.Linear(128, 64)
        # self.bn3 = nn.BatchNorm1d(64)
        self.linear4 = nn.Linear(64, 32)
        # self.bn4 = nn.BatchNorm1d(32)
        self.linear5 = nn.Linear(32, output_size)
        
       
    def forward(self, x):
        x = F.gelu(self.linear1(x))
        x = F.gelu(self.linear2(x))
        x = F.gelu(self.linear3(x))
        x = F.gelu(self.linear4(x))
        x = F.tanh(self.linear5(x))

        return x
    
# class ReplayBuffer():
#     def __init__(self, buffer_maxlen, device):
#         self.buffer = collections.deque(maxlen=buffer_maxlen)
#         self.device = device

#     def push(self, data):
#         self.buffer.append(data)

#     def sample(self, batch_size):
#         state_list = []
#         action_list = []
#         reward_list = []
#         next_state_list = []
#         done_list = []

#         batch = random.sample(self.buffer, batch_size)
#         for experience in batch:
#             s, a, r, n_s, d = experience
#             # state, action, reward, next_state, done

#             state_list.append(s)
#             action_list.append(a)
#             reward_list.append(r)
#             next_state_list.append(n_s)
#             done_list.append(d)

#         return (
#             torch.tensor(np.array(state_list), dtype=torch.float32).to(self.device),
#             torch.tensor(np.array(action_list), dtype=torch.float32).to(self.device),
#             torch.tensor(np.array(reward_list), dtype=torch.float32).unsqueeze(-1).to(self.device),
#             torch.tensor(np.array(next_state_list), dtype=torch.float32).to(self.device),
#             torch.tensor(np.array(done_list), dtype=torch.float32).unsqueeze(-1).to(self.device),
#         )

class ReplayBuffer():
    def __init__(self, buffer_maxlen, device):
        self.buffer = collections.deque(maxlen=buffer_maxlen)
        self.device = device

    def push(self, data):
        self.buffer.append(data)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        state_list, action_list, reward_list, next_state_list, done_list = zip(*batch)

        return (
            torch.stack(state_list).to(self.device),
            torch.stack(action_list).to(self.device),
            torch.tensor(reward_list, dtype=torch.float32, device=self.device).unsqueeze(-1),
            torch.stack(next_state_list).to(self.device),
            torch.tensor(done_list, dtype=torch.float32, device=self.device).unsqueeze(-1),
        )

def collect_target_data(agent_target, env_target, n_traj, device, seed):
    buffer_maxlen = 1000000
    buffer = ReplayBuffer(buffer_maxlen, device)

    states = []
    actions = []
    max_episode_steps = env_target.spec.max_episode_steps 
    for episode in range(int(n_traj)):
        score = 0
        state, _ = env_target.reset(seed=seed*episode) ############################################
        for _ in range(max_episode_steps):
            action = agent_target.get_action(state, deterministic=False)

            next_state, reward, terminated, truncated, _ = env_target.step(action)
            done = truncated or terminated

            done_mask = 0.0 if done else 1.0
            buffer.push((state, action, reward, next_state, done_mask))

            states.append(state)
            actions.append(action)

            state = next_state
            score += reward
            
            if done: break

        print("episode:{}, Return:{}, buffer_capacity:{}".format(episode, score, len(buffer.buffer)))
    env_target.close()

    data_dict = {
        'obs': states,
        'actions': actions,
    }
    torch.save(data_dict, 'cheetah.pt')
    
    print(f"Collected {n_traj} trajectories.")
    print(f"Collected {n_traj*max_episode_steps} transitions.")
    return buffer


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    


def store_data_from_file(buffer, file_path):
    data = torch.load(file_path)
    print(data.keys())

    # 確保數據是 (state, action, reward, next_state, done) 格式
    for i in range(len(data['obs'])):
        state = data['obs'][i]
        action = data['actions'][i]
        reward = 0
        next_state = data['next_obs'][i]
        done = 0 #data['done'][i]

        # state = torch.tensor(np.array(state), dtype=torch.float32)
        # action = torch.tensor(np.array(action), dtype=torch.float32)
        # reward = torch.tensor(np.array(reward), dtype=torch.float32)
        # next_state = torch.tensor(np.array(next_state), dtype=torch.float32)
        # done = torch.tensor(np.array(done), dtype=torch.float32)
        
        buffer.push((state, action, reward, next_state, done)) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, nargs='?', default=2)
    parser.add_argument("--ep", type=int, nargs='?', default=1000) # the same as cdpc
    parser.add_argument("--b", type=int, nargs='?', default=128) # the same as cdpc
    parser.add_argument("--lr", type=float, nargs='?', default=5e-3) # the same as cdpc
    parser.add_argument("--device", type=str, nargs='?', default="cuda")
    parser.add_argument("--wandb", action='store_true', default=False)
    parser.add_argument("--vedio", action='store_true', default=False)
    
    args = parser.parse_args()
    device = args.device
    seed = args.seed
    seed_everything(seed)

    if args.wandb:
        wandb.init(project="cdpc", name = f'baseline: d4rl dataset')

    env = gym.make("HalfCheetah-3legs", render_mode="rgb_array")
    env = gym.make("HalfCheetah-v4")#, exclude_current_positions_from_observation=False)

    ## parameters
    batch_size = args.b
    hidden_dim = 512
    action_range = 1.0
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]

    # ## load expert policy
    # trained_agent = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range, device).to(device)
    # model_path = f'models/cheetah/seed_{str(args.seed)}/{str(args.seed)}_cheetah_target.pth'
    # model_path = f'models/cheetah/seed_{str(args.seed)}/{str(args.seed)}_cheetah_source0.pth'
    # trained_agent.load_state_dict(torch.load( model_path, map_location=device ))

    # ## Create the Transitions object
    # buffer = collect_target_data(trained_agent, env, 5, device, seed)

    buffer = ReplayBuffer(buffer_maxlen=100000, device=device)
    #store_data_from_file(buffer, f"./traj_data/{str(seed)}_HalfCheetah-3legs_PPO.pt")
    store_data_from_file(buffer, f"./traj_data/halfcheetah-medium.pt")
    print(f"Replay buffer size: {len(buffer.buffer)}")

    ## train BC
    policy = MPC_Policy_Net(state_dim, action_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)#, weight_decay=1e-4)

    reward = evaluate_policy(policy, env, args.seed, args.device)
    if args.wandb:
        wandb.log({"episode": 0, "test/score": reward})
    print(f"BC Policy (Epoch {0}): mean_reward={reward:.2f}")

    loss_function = nn.MSELoss()
    for i in range(args.ep):
        ## update
        for _ in range(10):
            state, action, reward, next_state, done = buffer.sample(batch_size)
            pred_action = policy(state)
            loss_mpc = loss_function(pred_action, action)
            optimizer.zero_grad()
            loss_mpc.backward()
            optimizer.step()

        # if i%100==0 and args.vedio: env = gym.wrappers.RecordVideo(env, f'video/bc2')
        reward = evaluate_policy(policy, env, args.seed, args.device)
        if args.wandb:
            wandb.log({"episode": i+1, "test/score": reward, "train/loss": loss_mpc.item()})
        print(f"BC Policy (Epoch {i+1}): mean_reward={reward:.2f}, train_loss={loss_mpc.item():.4f}")
