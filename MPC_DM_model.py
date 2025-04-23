import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import os
import gymnasium as gym
import wandb
from utils import seed_everything, ReplayBuffer, get_top_k_trajs, d4rl2Transition, load_buffer


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
        x = torch.tanh(self.linear5(x))

        return x

class Dynamic_Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Dynamic_Model, self).__init__()
        self.fc1 = nn.Linear(input_size, 250)
        self.fc2 = nn.Linear(250, 250)
        self.fc3 = nn.Linear(250, 250)  
        self.fc4 = nn.Linear(250, 250)
        self.fc5 = nn.Linear(250, 250)
        self.fc6 = nn.Linear(250, output_size)

        

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        out = F.relu(self.fc5(out))
        out = self.fc6(out)
        
        return out

class MPC_DM:
    def __init__(self, target_obs, target_action, device):

        self.state_dim = target_obs
        self.action_dim = target_action
        self.device = device

        # initialize policy network parameters
        self.mpc_policy_net = MPC_Policy_Net(self.state_dim, self.action_dim).to(device)
        self.mpc_policy_optimizer = optim.Adam(self.mpc_policy_net.parameters(), lr=5e-3, weight_decay=1e-4)

        # initialize dynamic model parameters
        self.dynamic_model = Dynamic_Model(self.state_dim+self.action_dim, self.state_dim).to(device)
        self.dynamic_model_optimizer = optim.Adam(self.dynamic_model.parameters(), lr=1e-3, weight_decay=1e-4)


    def update(self, batch_size, buffer):
        # update MPC policy net
        state, action, reward, next_state, done = buffer.sample(batch_size)
        pred_action = self.mpc_policy_net(state)
        loss_mpc = F.mse_loss(pred_action, action)
        self.mpc_policy_optimizer.zero_grad()
        loss_mpc.backward()
        self.mpc_policy_optimizer.step()

        # update dynamic model
        state, action, reward, next_state, done = buffer.sample(batch_size)
        pred_next_state = self.dynamic_model(torch.cat([state, action], dim=1))
        # pred_next_state = self.dynamic_model(state, action)
        loss_dm = F.mse_loss(pred_next_state, next_state)
        self.dynamic_model_optimizer.zero_grad()
        loss_dm.backward()
        self.dynamic_model_optimizer.step()
        
        return loss_mpc, loss_dm
    
    def evaluate_policy(self, env, seed):
        total_reward = 0.0
        eval_episode = 10
        for ep in range(eval_episode):
            state, _ = env.reset(seed=seed*ep)
            for i in range(env.spec.max_episode_steps):
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                action = self.mpc_policy_net(state).cpu().detach().numpy().squeeze(0)
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
        return total_reward / eval_episode


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("MPC_pre_ep", type=int, nargs='?', default=10000)
    parser.add_argument("--seed", type=int, nargs='?', default=2)
    parser.add_argument("--expert_ratio", type=float, nargs='?', default=0.8) # random_ratio=1-expert_ratio
    parser.add_argument("--device", type=str, nargs='?', default="cuda")
    parser.add_argument("--env", type=str, nargs='?', default="cheetah") # cheetah reacher
    parser.add_argument("--top_k", type=int, nargs='?', default=10) # cheetah reacher
    parser.add_argument("--wandb", action='store_true', default=False)
    args = parser.parse_args()

    seed = args.seed
    device = args.device
    seed_everything(args.seed)

    if args.wandb:
        wandb.init(project="cdpc", name = f'MPC policy&DM: {str(args.seed)}_{args.env} top{str(args.top_k)}', tags=["policy&DM"])

    if args.env == "reacher":
        target_env = "Reacher-3joints"
        target_s_dim = 14
        target_a_dim = 3
        traj_len = 50
    elif args.env == "cheetah":
        target_env = "HalfCheetah-3legs"
        target_s_dim = 23
        target_a_dim = 9
        traj_len = 1000

    ##### 1 Load target domain offline data #####
    print("##### Loading offline data #####")
    data_path = f'./train_set/{str(args.seed)}_{args.env}_{args.expert_ratio}.pkl'
    d4rl_data = load_buffer(data_path)
    top_k_data = get_top_k_trajs(d4rl_data, traj_len=traj_len, top_k=10)
    
    buffer_maxlen = 1000000
    buffer = ReplayBuffer(buffer_maxlen, device)
    d4rl2Transition(top_k_data, buffer)
    print(f"Loaded top {buffer.buffer_len()} trajectories from {data_path}")


    ##### 2 Train MPC policy and Dynamic Model #####
    location = f'./models/{args.env}/seed_{str(args.seed)}/'
    mpc_location = f'{location}/expert_ratio_{args.expert_ratio}/'

    mpc_dm = MPC_DM(target_s_dim, target_a_dim, args.device)
    os.makedirs(mpc_location, exist_ok=True)
    if not os.path.exists(f'{mpc_location}/{str(args.seed)}_MPCModel.pth'):
        print("##### Training MPC policy and Dynamic Model #####")
        batch_size = 128
        for i in range(args.MPC_pre_ep):
            loss_mpc, loss_dm = mpc_dm.update(batch_size, buffer)
            #print(f"episode: {i}/{args.MPC_pre_ep}, train/loss_mpc: {loss_mpc}, train/loss_dm: {loss_dm}")

            if i % 100==0:
                env = gym.make(target_env)
                reward = mpc_dm.evaluate_policy(env, args.seed)
                print(f"episode: {i}/{args.MPC_pre_ep}, reward: {reward}")
                if args.wandb:
                    wandb.log({"mpc_dm episode": i, "test/BC_score": reward,})

            if args.wandb:
                wandb.log({"mpc_dm episode": i, "train/loss_mpc": loss_mpc, "train/loss_dm": loss_dm,})
        torch.save(mpc_dm.mpc_policy_net.state_dict(), f'{mpc_location}/{str(args.seed)}_MPCModel.pth')
        torch.save(mpc_dm.dynamic_model.state_dict(), f'{mpc_location}/{str(args.seed)}_DynamicModel.pth')