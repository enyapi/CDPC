import torch
import numpy as np
import random
import torch.nn as nn
import collections
import torch.nn.functional as F
import torch.optim as optim


class ReplayBuffer():
    def __init__(self, buffer_maxlen, device):
        self.buffer = collections.deque(maxlen=buffer_maxlen)
        self.device = device

    def push(self, data):
        self.buffer.append(data)

    def sample(self, batch_size):
        state_list = []
        action_list = []
        reward_list = []
        next_state_list = []
        done_list = []

        batch = random.sample(self.buffer, batch_size)
        for experience in batch:
            s, a, r, n_s, d = experience
            # state, action, reward, next_state, done

            state_list.append(s)
            action_list.append(a)
            reward_list.append(r)
            next_state_list.append(n_s)
            done_list.append(d)

        return (
            torch.tensor(np.array(state_list), dtype=torch.float32).to(self.device),
            torch.tensor(np.array(action_list), dtype=torch.float32).to(self.device),
            torch.tensor(np.array(reward_list), dtype=torch.float32).unsqueeze(-1).to(self.device),
            torch.tensor(np.array(next_state_list), dtype=torch.float32).to(self.device),
            torch.tensor(np.array(done_list), dtype=torch.float32).unsqueeze(-1).to(self.device),
        )

    def buffer_len(self):
        return len(self.buffer)

class MPC_Policy_Net(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        # self.linear1 = nn.Linear(input_size, 64)
        # # self.bn1 = nn.BatchNorm1d(64)
        # self.linear2 = nn.Linear(64, 128)
        # # self.bn2 = nn.BatchNorm1d(128)
        # self.linear3 = nn.Linear(128, 64)
        # # self.bn3 = nn.BatchNorm1d(64)
        # self.linear4 = nn.Linear(64, 32)

        self.linear1 = nn.Linear(input_size, 256)
        self.linear2 = nn.Linear(256, 256)

        # self.bn4 = nn.BatchNorm1d(32)
        #self.linear5 = nn.Linear(32, output_size)
        self.linear3 = nn.Linear(256, output_size)
        
       
    def forward(self, x):
        # x = F.gelu(self.linear1(x))
        # x = F.gelu(self.linear2(x))
        # x = F.gelu(self.linear3(x))
        # x = F.gelu(self.linear4(x))
        # x = F.tanh(self.linear5(x))

        x = F.gelu(self.linear1(x))
        x = F.gelu(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        return x

class Dynamic_Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Dynamic_Model, self).__init__()
        hidden_size = 512
        self.fc1 = nn.Linear(input_size, hidden_size)
        # self.fc2 = nn.Linear(250, 250)
        # self.fc3 = nn.Linear(250, 250)
        # self.fc4 = nn.Linear(250, 250)
        # self.fc5 = nn.Linear(250, 250)
        self.fc6 = nn.Linear(hidden_size, output_size)

        

    def forward(self, x):
        out = F.relu(self.fc1(x))
        # out = F.relu(self.fc2(out))
        # out = F.relu(self.fc3(out))
        # out = F.relu(self.fc4(out))
        # out = F.relu(self.fc5(out))
        out = self.fc6(out)
        
        return out

class MPCPolicyTrainer:
    def __init__(self, state_dim, action_dim, device):
        self.device = device
        self.policy_net = MPC_Policy_Net(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3, weight_decay=1e-4)

    def update(self, batch_size, buffer):
        state, action, reward, next_state, done = buffer.sample(batch_size)
        pred_action = self.policy_net(state)
        loss = F.mse_loss(pred_action, action)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
    
    def evaluate_policy(self, env, seed):
        total_reward = 0.0
        state, _ = env.reset(seed=seed)
        for i in range(env.spec.max_episode_steps):
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            action = self.policy_net(state).cpu().detach().numpy().squeeze(0)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
        return total_reward


class DynamicsModelTrainer:
    def __init__(self, state_dim, action_dim, device):
        self.device = device
        self.model = Dynamic_Model(state_dim + action_dim, state_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)

    def update(self, batch_size, buffer):
        state, action, reward, next_state, done = buffer.sample(batch_size)
        pred_next_state = self.model(torch.cat([state, action], dim=1))
        loss = F.mse_loss(pred_next_state, next_state)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
    
    def evaluate_dm(self, batch_size, buffer):
        state, action, reward, next_state, done = buffer.sample(batch_size)
        
        with torch.no_grad():
            pred_next_state = self.model(torch.cat([state, action], dim=1))
            val_loss = F.mse_loss(pred_next_state, next_state)

        return val_loss