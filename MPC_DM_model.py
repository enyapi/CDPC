import torch
import random
import torch.nn as nn
import collections
import torch.nn.functional as F
import torch.optim as optim
import wandb


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

        return torch.FloatTensor(state_list).to(self.device), \
               torch.FloatTensor(action_list).to(self.device), \
               torch.FloatTensor(reward_list).unsqueeze(-1).to(self.device), \
               torch.FloatTensor(next_state_list).to(self.device), \
               torch.FloatTensor(done_list).unsqueeze(-1).to(self.device)

    def buffer_len(self):
        return len(self.buffer)

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

        # initialize policy network parameters
        self.mpc_policy_net = MPC_Policy_Net(self.state_dim, self.action_dim).to(device)
        self.mpc_policy_optimizer = optim.Adam(self.mpc_policy_net.parameters(), lr=1e-3, weight_decay=1e-4)

        # initialize dynamic model parameters
        self.dynamic_model = Dynamic_Model(self.state_dim+self.action_dim, self.state_dim).to(device)
        self.dynamic_model_optimizer = optim.Adam(self.dynamic_model.parameters(), lr=1e-3, weight_decay=1e-4)


    def update(self, batch_size, buffer):
        # update MPC policy net
        state, action, reward, next_state, done = buffer.sample(batch_size)
        pred_action = self.mpc_policy_net(state)
        loss_mpc = F.mse_loss(pred_action, action)
        wandb.log({"train/loss_mpc": loss_mpc, })
        self.mpc_policy_optimizer.zero_grad()
        loss_mpc.backward()
        self.mpc_policy_optimizer.step()

        # update dynamic model
        state, action, reward, next_state, done = buffer.sample(batch_size)
        pred_next_state = self.dynamic_model(torch.cat([state, action], dim=1))
        # pred_next_state = self.dynamic_model(state, action)
        loss_dm = F.mse_loss(pred_next_state, next_state)
        wandb.log({"train/loss_dm": loss_dm, })
        self.dynamic_model_optimizer.zero_grad()
        loss_dm.backward()
        self.dynamic_model_optimizer.step()