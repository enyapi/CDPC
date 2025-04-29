import gymnasium as gym
import torch
import random
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from itertools import combinations
from MPC_DM_model import *
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv


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
    ctrl_cost = ctrl_cost_weight * torch.square(action).sum()

    forward_reward_weight = 1
    forward_reward = forward_reward_weight * x_velocity
    reward = forward_reward - ctrl_cost
    return reward

class ReplayBuffer_traj():
    def __init__(self):
        self.buffer = []

    def push(self, total_rewards, state, action, next_state):
        trajectory_info = {
        'total_rewards': total_rewards,
        'state': state,
        'action': action, 
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
    def __init__(self, input_size, output_size, batch_size, device, use_flow):
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
        self.use_flow = use_flow


    def forward(self, x):
        out, self.hidden = self.lstm(x.unsqueeze(1), self.hidden)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.fc2(out[:, -1, :])

        if self.use_flow:
            out = F.tanh(out)
        
        return out

    def reset_hidden(self, batch_size, flag=False):
        if not flag:
            self.hidden = ((torch.randn(self.num_layers, batch_size, self.hidden_size) * 0.01 + 0.0).to(self.device), (torch.randn(self.num_layers, batch_size, self.hidden_size)*0.001).to(self.device))
        else:
            self.hidden = ((torch.randn(self.num_layers, batch_size, self.hidden_size) * 0.01 + 0.0).to(self.device), (torch.randn(self.num_layers, batch_size, self.hidden_size)*0.001).to(self.device))
            return self.hidden
        
class State_Projector(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        
       
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x
    
class Action_Projector(nn.Module):
    def __init__(self, target_state_size, target_action_size, source_action_size, hidden_size=256, action_range=1.):
        super().__init__()
        self.linear1 = nn.Linear(target_state_size + target_action_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, source_action_size)
        
        self.action_range = action_range
       
    def forward(self, target_state, target_action):
        x = torch.cat([target_state, target_action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.action_range * F.tanh(self.linear3(x))

        return x

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size=256):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        
       
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x

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


        # NN Initialization
        self.state_projector = State_Projector(self.target_state_space_dim, self.source_state_space_dim).to(self.device)
        self.action_projector = Action_Projector(self.target_state_space_dim, self.target_action_space_dim, self.source_action_space_dim).to(self.device)
        self.state_discriminator = Discriminator(self.source_state_space_dim).to(self.device)
        self.action_discriminator = Discriminator(self.source_action_space_dim).to(self.device)
        
        self.projectors_optimizer = optim.Adam(list(self.state_projector.parameters()) + list(self.action_projector.parameters()), lr=self.lr)
        # self.ap_optimizer = optim.Adam(self.action_projector.parameters(), lr=self.lr)
        self.sd_optimizer = optim.Adam(self.state_discriminator.parameters(), lr=self.lr)
        self.ad_optimizer = optim.Adam(self.action_discriminator.parameters(), lr=self.lr)
        
        # mpc policy net
        self.mpc_policy_net = self.mpc_dm.mpc_policy_net
        self.dynamics_model = self.mpc_dm.dynamic_model


    # def learn(self, train_set):
    def learn(self, train_set:ReplayBuffer_traj, target_buffer:ReplayBuffer, source_buffer:ReplayBuffer):
        # organize dateset
        index = list(range(train_set.buffer_len()))
        combinations_list = list(combinations(index, 2))
        selected_combinations = random.sample(combinations_list, self.batch_size)
        state_a, next_state_a, action_a, state_b, next_state_b, action_b = [], [], [], [], [], []
        for combo in selected_combinations:
            traj_a, traj_b = train_set.sample(combo)
            traj_a, traj_b = self.compare_trajectories(traj_a, traj_b)
            # if random.random() < 0.1: traj_a, traj_b = traj_b, traj_a

            state_a_tensor = torch.tensor(traj_a['state'], dtype=torch.float32)
            next_state_a_tensor = torch.tensor(traj_a['next_state'], dtype=torch.float32)
            action_a_tensor = torch.tensor(traj_a['action'], dtype=torch.float32)
            state_b_tensor = torch.tensor(traj_b['state'], dtype=torch.float32)
            action_b_tensor = torch.tensor(traj_b['action'], dtype=torch.float32)
            next_state_b_tensor = torch.tensor(traj_b['next_state'], dtype=torch.float32)

            state_a.append(state_a_tensor)
            action_a.append(action_a_tensor)
            next_state_a.append(next_state_a_tensor)
            state_b.append(state_b_tensor)
            action_b.append(action_b_tensor)
            next_state_b.append(next_state_b_tensor)

        state_a = torch.stack(state_a, dim=0).to(self.device)
        action_a = torch.stack(action_a, dim=0).to(self.device)
        next_state_a = torch.stack(next_state_a, dim=0).to(self.device)
        state_b = torch.stack(state_b, dim=0).to(self.device)
        action_b = torch.stack(action_b, dim=0).to(self.device)
        next_state_b = torch.stack(next_state_b, dim=0).to(self.device)

        R_s_a_tensor = self.pref_calculation(state_a, action_a)
        R_s_b_tensor = self.pref_calculation(state_b, action_a)

        ## preference consistency loss
        result_tensor = torch.cat((R_s_a_tensor, R_s_b_tensor), dim=-1).type(torch.float32)#.unsqueeze(0)
        sub_first_rewards = result_tensor - result_tensor[:, 0][:, None]
        loss_pref = torch.logsumexp(sub_first_rewards, dim=-1).mean()
        pref_acc = (sub_first_rewards[:, 1] < 0).sum().item() / self.batch_size

        source_state, source_action, source_reward, source_next_state, source_done = source_buffer.sample(batch_size=self.batch_size)
        target_state, target_action, target_reward, target_next_state, target_done = target_buffer.sample(batch_size=self.batch_size)

        adv_state_loss_for_D, adv_action_loss_for_D = self.adversarial_loss_D(target_state, target_action, source_state, source_action)
        dcc_loss = self.dcc_loss(target_state, target_action, target_next_state)

        # optimize state discriminator
        self.sd_optimizer.zero_grad()
        adv_state_loss_for_D.backward()
        self.sd_optimizer.step()

        # optimize action discriminator
        self.ad_optimizer.zero_grad()
        adv_action_loss_for_D.backward()
        self.ad_optimizer.step()

        # optimize state projectors
        adv_state_loss_for_G, adv_action_loss_for_G = self.adversarial_loss_G(target_state, target_action)

        projectors_loss = adv_state_loss_for_G + adv_action_loss_for_G + dcc_loss + loss_pref
        self.projectors_optimizer.zero_grad()
        projectors_loss.backward()
        self.projectors_optimizer.step()

        return dcc_loss.item(), adv_state_loss_for_D.item(), adv_action_loss_for_D.item(), adv_state_loss_for_G.item(), adv_action_loss_for_G.item(), loss_pref.item(), pref_acc
    
    
    def make_env(self, seed):
        def _init():
            if self.env == "cheetah":
                env = gym.make(self.source_env, exclude_current_positions_from_observation=False).unwrapped
            else:
                env = gym.make(self.source_env).unwrapped
            env.reset(seed=seed)
            return env
        return _init
    

    def adversarial_loss_D(self, target_state, target_action, source_state, source_action):

        loss_function = nn.MSELoss()

        #----------------------------------------------------------------
        # losses for discriminators
        fake_source_state = self.state_projector(target_state)
        fake_source_action = self.action_projector(target_state, target_action)

        fake_state_score = self.state_discriminator(fake_source_state)
        real_state_score = self.state_discriminator(source_state)

        fake_action_score = self.action_discriminator(fake_source_action)
        real_action_score = self.action_discriminator(source_action)

        state_loss_for_D = (loss_function(real_state_score, torch.ones(real_state_score.shape, device=real_state_score.device)) + \
                            loss_function(fake_state_score, torch.zeros(fake_state_score.shape, device=fake_state_score.device))) / 2.
        
        action_loss_for_D = (loss_function(real_action_score, torch.ones(real_action_score.shape, device=real_action_score.device)) + \
                             loss_function(fake_action_score, torch.zeros(fake_action_score.shape, device=fake_action_score.device))) / 2.
        
        return state_loss_for_D, action_loss_for_D
    
    def adversarial_loss_G(self, target_state, target_action):

        loss_function = nn.MSELoss()
        #----------------------------------------------------------------
        # losses for generators
        fake_source_state = self.state_projector(target_state)
        fake_source_action = self.action_projector(target_state, target_action)

        fake_state_score = self.state_discriminator(fake_source_state)
        fake_action_score = self.action_discriminator(fake_source_action)

        state_loss_for_G = loss_function(fake_state_score, torch.ones(fake_state_score.shape, device=fake_state_score.device))
        action_loss_for_G = loss_function(fake_action_score, torch.ones(fake_action_score.shape, device=fake_action_score.device))

        return state_loss_for_G, action_loss_for_G

    def dcc_loss(self, target_state, target_action, target_next_state):
        loss_function = nn.MSELoss()

        source_state = self.state_projector(target_state)
        source_action = self.action_projector(target_state, target_action)
        predicted_next_state_1 = self.source_dynamics_model(torch.cat([source_state, source_action], dim=-1))
        predicted_next_state_2 = self.state_projector(target_next_state)

        loss = loss_function(predicted_next_state_1, predicted_next_state_2)

        return loss

    def pref_calculation(self, target_states, target_actions):
        
        R_s_tensor = torch.zeros((self.batch_size, 1)).to(self.device)

        length = target_states[0,:,0].shape[0] # trajectory length
        if self.source_env == 'HalfCheetah-v4':
            length -= 1 # prevent from access state[length] in the following for loop

        for i in range(length):

            source_states = self.state_projector(target_states[:,i,:].squeeze(1))
            source_actions = self.action_projector(target_states[:,i,:], target_actions[:,i,:])
            source_next_states = self.state_projector(target_states[:,i + 1,:].squeeze(1)) 
            # source_next_states = torch.tensor(source_next_states, dtype=torch.float32).to(self.device)

            for b in range(self.batch_size):
                if self.env == "reacher":
                    r = reacher_source_R(source_states[b], source_actions[b])
                elif self.env == "cheetah":
                    r = cheetah_source_R(source_states[b], source_actions[b], source_next_states[b])
                R_s_tensor[b] += (0.99**i)*r

        return R_s_tensor


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
            s = self.dynamics_model(torch.cat([s, a], dim=-1))
            traj_state[:, j, :] = s

            a = self.mpc_policy_net(s)
            traj_action[:, j, :] = a

        s = self.dynamics_model(torch.cat([s, a], dim=-1))
        traj_state[:, self.h, :] = s

        return traj_state, traj_action

    
    def __decodeTraj(self, s, a): ## 0.17
        with torch.no_grad():
            env_fns = [self.make_env(self.seed) for _ in range(self.N)]
            vec_env = DummyVecEnv(env_fns)

            rewards = np.zeros(self.N)
            for j in range(self.h):
                source_states = self.state_projector(s[:, j, :])
                source_actions = self.action_projector(s[:, j, :], a[:, j, :])

                for i, env in enumerate(vec_env.envs):
                    env.reset_specific(state=source_states[i].cpu().detach().numpy())

                if self.env == "reacher":
                    r = np.array([
                        reacher_source_R(source_states[i], source_actions[i]).cpu().detach().numpy()
                        for i in range(self.N)
                    ])
                elif self.env == "cheetah":
                    next_states, r, _, _ = vec_env.step(source_actions.cpu().detach().numpy())
                    r = np.array([
                        cheetah_source_R(source_states[i], source_actions[i], next_states[i]).cpu().detach().numpy()
                        for i in range(self.N)
                    ])

                rewards += r * (0.99 ** j)

            best_idx = np.argsort(rewards)[-1]
            best_action = a[best_idx, 0, :]
        return best_action, rewards[best_idx]


    def evaluate(self):
        env_target = gym.make(self.target_env, render_mode='rgb_array') #rgb_array
        #env_target = gym.wrappers.RecordVideo(env_target, 'video')
        self.action_projector.eval()
        self.state_projector.eval()

        s0, _ = env_target.reset(seed=self.seed)
        total_reward = 0
        best_rewards = 0
        for i in range(env_target.spec.max_episode_steps):
            ## MPC inference
            # generate action sequence using policy network and get state sequence
            s, a = self.__sampleTraj(s0)
            # decode state sequence to source state sequence and get sorted action sequence (in terms of reward)
            # a0_target, best_reward = self.__decodeTraj(s, a)
            a0_target, best_reward = self.__decodeTraj(s, a)
            a0_target = a0_target.cpu().detach().numpy()

            s1, r1, terminated, truncated, _ = env_target.step(a0_target)

            done = truncated or terminated
            total_reward += r1
            best_rewards += best_reward
            s0 = s1
            
            if done: break

        return total_reward
