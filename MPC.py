import gymnasium as gym
import torch
import random
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from itertools import combinations
import numpy as np
from joblib import Parallel, delayed
import datetime

def reacher_source_R(state, action):
    vec = state[-3:]#.detach().cpu().numpy()    # last 3 dim
    reward_dist = -torch.norm(vec)
    reward_ctrl = -np.square(action).sum()
    reward = reward_dist + reward_ctrl 
    return reward 

def cheetah_source_R(state, action, next_state):
    x_position_before = state[0] # gradient
    x_position_after = next_state[0] # no gradient
    dt = 0.05
    x_velocity = (x_position_after - x_position_before) / dt

    ctrl_cost_weight = 0.1
    ctrl_cost = ctrl_cost_weight * np.sum(np.square(action))

    forward_reward_weight = 1
    forward_reward = forward_reward_weight * x_velocity
    reward = forward_reward - ctrl_cost
    return reward

class ReplayBuffer_traj():
    def __init__(self):
        self.buffer = []

    def push(self, total_rewards, state, next_state):
        trajectory_info = {
        'total_rewards': total_rewards,
        'state': state,
        'next_state': next_state,
        }
        self.buffer.append(trajectory_info)

    def sample(self, combo):
        trajectory_a = self.buffer[combo[0]]
        trajectory_b = self.buffer[combo[1]]

        return trajectory_a, trajectory_b

    def buffer_len(self):
        return len(self.buffer)
    
        
class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_directions=2):
        super(BidirectionalLSTM, self).__init__()
        self.num_directions = num_directions
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bidirectional=False, batch_first=True)

    def forward(self, x, h):
        # x: (batch_size, seq_len, input_size)
        out, h = self.lstm(x, h)
        # out: (batch_size, seq_len, num_directions * hidden_size)
        return out, h
    

class Decoder_Net(nn.Module):
    def __init__(self, input_size, output_size, batch_size, device):
        super(Decoder_Net, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = 128
        self.num_layers = 1
        self.lstm = BidirectionalLSTM(input_size, self.hidden_size, num_layers=self.num_layers)
        self.dropout = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(self.hidden_size, 64)  
        self.fc2 = nn.Linear(64, output_size)
        self.random_hidden = ((torch.randn(self.num_layers, self.batch_size, self.hidden_size) * 0.01 + 0.0).to(device), (torch.randn(self.num_layers, batch_size, self.hidden_size)*0.001).to(device))
        self.hidden = self.random_hidden
        self.device = device


    def forward(self, x):
        out, self.hidden = self.lstm(x.unsqueeze(1), self.hidden)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.fc2(out[:, -1, :])
        
        return out

    def reset_hidden(self, batch_size, flag=False):
        if not flag:
            self.hidden = ((torch.randn(self.num_layers, batch_size, self.hidden_size) * 0.01 + 0.0).to(self.device), (torch.randn(self.num_layers, batch_size, self.hidden_size)*0.001).to(self.device))
        else:
            self.hidden = ((torch.randn(self.num_layers, batch_size, self.hidden_size) * 0.01 + 0.0).to(self.device), (torch.randn(self.num_layers, batch_size, self.hidden_size)*0.001).to(self.device))
            return self.hidden
    

class Encoder_Net(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 32)
        self.linear2 = nn.Linear(32, 64)
        self.linear3 = nn.Linear(64, 128)
        self.linear4 = nn.Linear(128, 256)
        self.linear5 = nn.Linear(256, output_size)
        
       
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = self.linear5(x)

        return x

    

class MPC(object):
    def __init__(self, h=20, N=50, argmin=True, **kwargs):
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.d = self.target_action_space_dim      # dimension of function input X
        self.h = h                        # sequence length of sample data
        self.N = N                        # sample N examples each iteration
        self.reverse = argmin         # try to maximum or minimum the target function
        self.v_min = [-1.0] * self.d                # the value minimum
        self.v_max = [1.0] * self.d                 # the value maximum

        # state AE
        self.decoder_net = Decoder_Net(self.target_state_space_dim, self.source_state_space_dim, self.batch_size, self.device).to(self.device)
        self.encoder_net = Encoder_Net(self.source_state_space_dim, self.target_state_space_dim).to(self.device)
        self.dec_optimizer = optim.Adam(self.decoder_net.parameters(), lr=self.lr)
        self.enc_optimizer = optim.Adam(self.encoder_net.parameters(), lr=self.lr)
        
        # mpc policy net
        self.mpc_policy_net = self.mpc_dm.mpc_policy_net
        self.dynamic_model = self.mpc_dm.dynamic_model
    
    def learn(self, train_set):
        # organize dateset
        index = list(range(train_set.buffer_len()))
        combinations_list = list(combinations(index, 2))
        selected_combinations = random.sample(combinations_list, self.batch_size)
        state_a, next_state_a, state_b, next_state_b = [], [], [], []
        for combo in selected_combinations:
            traj_a, traj_b = train_set.sample(combo)
            traj_a, traj_b = self.compare_trajectories(traj_a, traj_b)
            # if random.random() < 0.1: traj_a, traj_b = traj_b, traj_a

            state_a_tensor = torch.tensor(traj_a['state'], dtype=torch.float32)
            next_state_a_tensor = torch.tensor(traj_a['next_state'], dtype=torch.float32)
            state_b_tensor = torch.tensor(traj_b['state'], dtype=torch.float32)
            next_state_b_tensor = torch.tensor(traj_b['next_state'], dtype=torch.float32)

            state_a.append(state_a_tensor)
            next_state_a.append(next_state_a_tensor)
            state_b.append(state_b_tensor)
            next_state_b.append(next_state_b_tensor)

        state_a = torch.stack(state_a, dim=0).to(self.device)
        next_state_a = torch.stack(next_state_a, dim=0).to(self.device)
        state_b = torch.stack(state_b, dim=0).to(self.device)
        next_state_b = torch.stack(next_state_b, dim=0).to(self.device)
        
        # update state decoder
        ## traj a, b
        R_s_a_tensor, loss_tran_a, loss_rec_a = self.train_loss(state_a, next_state_a)
        R_s_b_tensor, loss_tran_b, loss_rec_b = self.train_loss(state_b, next_state_b)

        ## transition loss
        loss_tran = (loss_tran_a+loss_tran_b) / state_a[0,:,0].shape[0] # trajectory length

        ## rec loss
        loss_rec = (loss_rec_a+loss_rec_b) / state_a[0,:,0].shape[0]
        
        ## preference consistency loss
        result_tensor = torch.cat((R_s_a_tensor, R_s_b_tensor), dim=-1).type(torch.float32)#.unsqueeze(0)
        sub_first_rewards = result_tensor-result_tensor[:,0][:,None]
        loss_pref = torch.sum(sub_first_rewards.exp(), -1).log().mean()
        pref_acc = (sub_first_rewards[:,1] < 0).sum().item() / self.batch_size
        
        dec_loss = loss_tran + loss_rec + loss_pref # no preference loss
        enc_loss = loss_rec

        self.dec_optimizer.zero_grad()
        self.enc_optimizer.zero_grad()
        dec_loss.backward(retain_graph=True)
        enc_loss.backward(retain_graph=True)
        self.dec_optimizer.step()
        self.enc_optimizer.step()

        return loss_tran.item(), loss_pref.item(), loss_rec.item(), pref_acc
    

    def train_loss(self, state, next_state):
        loss_function = nn.MSELoss()
        env = gym.make(self.source_env, exclude_current_positions_from_observation=False).unwrapped if self.env=="cheetah" else gym.make(self.source_env).unwrapped
        env.reset(seed = self.seed)
        loss_tran = 0
        loss_rec = 0
        R_s_tensor = torch.zeros((self.batch_size, 1)).to(self.device)
        self.decoder_net.reset_hidden(self.batch_size, flag=True)
        dec_s = self.decoder_net(state[:,0,:].squeeze(1))
        for i in range(state[0,:,0].shape[0]): # trajectory length
            tran_s1 = []
            for b in range(self.batch_size):
                env.reset_specific(state = dec_s[b].cpu().detach().numpy())

                action = self.agent.get_action(dec_s[b].cpu(), deterministic=True)
                #action, _ = self.agent.predict(dec_s[b].cpu().detach().numpy(), deterministic=True) # SB3-SAC default
                #action = self.agent.policy.actor(dec_s[b], deterministic=True) # SB3-SAC GPU

                tmp = env.step(np.squeeze(action))
                tran_s1_tensor = torch.tensor(tmp[0], dtype=torch.float32).unsqueeze(0).to(self.device)
                tran_s1.append(tran_s1_tensor)
                ##
                if self.env == "reacher":
                    r = reacher_source_R(dec_s[b], action)
                elif self.env == "cheetah":
                    r = cheetah_source_R(dec_s[b], action, tran_s1_tensor.squeeze(0))
                R_s_tensor[b] = R_s_tensor[b] + (0.99**i)*r
                
            tran_s1 = torch.stack(tran_s1, dim=0).squeeze(1)
            dec_s1 = self.decoder_net(next_state[:,i,:].squeeze(1))
            loss_tran += loss_function(tran_s1, dec_s1)
            rec_s = self.encoder_net(dec_s)  ###
            loss_rec += loss_function(rec_s, state[:,i,:].squeeze(1))  ###
            dec_s = dec_s1
        env.close()
        return R_s_tensor, loss_tran, loss_rec
    

    def compare_trajectories(self, trajectory_info_a, trajectory_info_b):
        if trajectory_info_a['total_rewards'] > trajectory_info_b['total_rewards']:
            return trajectory_info_a, trajectory_info_b
        else:
            return trajectory_info_b, trajectory_info_a
    

    def __sampleTraj(self, state): ## 0.008s
        self.mpc_policy_net.eval()
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        
        action = self.mpc_policy_net(state)

        traj_state = torch.zeros((self.N, self.h+1, state.shape[-1]), dtype=torch.float32, device=self.device)
        traj_action = torch.zeros((self.N, self.h, self.target_action_space_dim), dtype=torch.float32, device=self.device)
        
        traj_state[:, 0, :] = state 
        
        noise = torch.randn((self.N, *action.shape), dtype=torch.float32, device=self.device) * 0.1
        action_noisy = action + noise
        action_noisy[0] = action
        traj_action[:, 0, :] = action_noisy

        s = state.repeat(self.N, 1)
        a = action_noisy

        for j in range(1, self.h):
            s = self.dynamic_model(torch.cat([s, a], dim=-1))
            traj_state[:, j, :] = s

            a = self.mpc_policy_net(s)
            traj_action[:, j, :] = a

        s = self.dynamic_model(torch.cat([s, a], dim=-1))
        traj_state[:, self.h, :] = s

        return traj_state, traj_action

    
    def __decodeTraj(self, s, a): ## 0.54
        with torch.no_grad():
            env = gym.make(self.source_env, exclude_current_positions_from_observation=False).unwrapped if self.env=="cheetah" else gym.make(self.source_env).unwrapped
            env.reset(seed = self.seed)
            traj_action = []
            reward_list = []
            for i in range(self.N):
                self.decoder_net.reset_hidden(1)
                traj_action_one = []
                reward = 0
                state = self.decoder_net(s[i,0,:].unsqueeze(0)).squeeze(0)
                env.reset_specific(state = state.cpu().detach().numpy())
                for j in range(self.h):
                    action = self.agent.get_action(state.cpu(), deterministic=True)
                    #action, _ = self.agent.predict(state.cpu().detach().numpy(), deterministic=True) # SB3-SAC default
                    #action = self.agent.policy.actor(state.unsqueeze(0), deterministic=True) # SB3-SAC GPU

                    if self.env == "reacher":
                        r = reacher_source_R(state, action).cpu().detach().numpy()
                    elif self.env == "cheetah":
                        next_s, r, _, _, _ = env.step(np.squeeze(action))
                        r = cheetah_source_R(state, action, next_s).cpu().detach().numpy()

                    reward += r
                    next_state = self.decoder_net(s[i,j+1,:].unsqueeze(0)).squeeze(0)
                    traj_action_one.append(a[i,j,:])
                    state = next_state
                traj_action.append(traj_action_one)
                reward_list.append(reward)

        data = [(xi, yi) for xi, yi in zip(traj_action, reward_list)]
        sorted_action = sorted(data, key=lambda pair: pair[1], reverse=self.reverse)
        
        sorted_action = [torch.stack(item[0]) for item in sorted_action]
        sorted_action = torch.stack(sorted_action)

        return sorted_action[0,0,:]



    # def __decodeTraj(self, s, a): ## 0.43s
    #     with torch.no_grad():
    #         env = gym.make(self.source_env, exclude_current_positions_from_observation=False).unwrapped if self.env=="cheetah" else gym.make(self.source_env).unwrapped
    #         env.reset(seed = self.seed)
    #         self.decoder_net.reset_hidden(self.N* (self.h+1))
    #         decoded_states = self.decoder_net(s.view(self.N * (self.h+1), -1))
    #         decoded_states = decoded_states.view(self.N, self.h+1, -1)
    #         traj_action = []
    #         reward_list = []
    #         for i in range(self.N):
    #             traj_action_one = []
    #             reward = 0
    #             state = decoded_states[i, 0, :]
    #             env.reset_specific(state = state.cpu().detach().numpy())
    #             for j in range(self.h):
    #                 action = self.agent.get_action(state.cpu(), deterministic=True)

    #                 if self.env == "reacher":
    #                     r = reacher_source_R(state, action).cpu().detach().numpy()
    #                 elif self.env == "cheetah":
    #                     next_s, r, _, _, _ = env.step(np.squeeze(action))
    #                     r = cheetah_source_R(state, action, next_s).cpu().detach().numpy()

    #                 reward += r
    #                 next_state = decoded_states[i, j+1, :]
    #                 traj_action_one.append(a[i,j,:])
    #                 state = next_state
    #             traj_action.append(traj_action_one)
    #             reward_list.append(reward)

    #     data = [(xi, yi) for xi, yi in zip(traj_action, reward_list)]
    #     sorted_action = sorted(data, key=lambda pair: pair[1], reverse=self.reverse)
        
    #     sorted_action = [torch.stack(item[0]) for item in sorted_action]
    #     sorted_action = torch.stack(sorted_action)

    #     return sorted_action[0,0,:]


    def evaluate(self):
        env_target = gym.make(self.target_env, render_mode='rgb_array') #rgb_array
        env_target = gym.wrappers.RecordVideo(env_target, 'video')
        self.decoder_net.eval()

        s0, _ = env_target.reset(seed=self.seed)
        total_reward = 0
        for i in range(env_target.spec.max_episode_steps):
            ## MPC inference
            # generate action sequence using policy network and get state sequence
            s, a = self.__sampleTraj(s0)
            # decode state sequence to source state sequence and get sorted action sequence (in terms of reward)
            a0_target = self.__decodeTraj(s, a)

            s1, r1, terminated, truncated, _ = env_target.step(a0_target.cpu().detach().numpy())

            print(i)
            print(s1)
            print(a0_target)
            print(r1)
            print()
            done = truncated or terminated
            total_reward += r1
            s0 = s1
            
            if done: break

        return total_reward
