import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import wandb
import random
import argparse
import os
import pickle
from utils import seed_everything

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch)) # stack for each element
        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)
        
        
class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        # self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, 1)
        
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1) # the dim 0 is number of samples
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        # x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x
        
        
class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, action_range=1., device="cuda", init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        # self.linear3 = nn.Linear(hidden_size, hidden_size)
        # self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.action_range = action_range
        self.num_actions = num_actions

        self.device = device

        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        # x = F.relu(self.linear3(x))
        # x = F.relu(self.linear4(x))

        mean    = (self.mean_linear(x))
        # mean    = F.leaky_relu(self.mean_linear(x))
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        '''
        generate sampled action with state as input wrt the policy network;
        '''
        mean, log_std = self.forward(state)
        std = log_std.exp() # no clip in evaluation, clip affects gradients flow
        
        normal = Normal(0, 1)
        z      = normal.sample(mean.shape) 
        action_0 = torch.tanh(mean + std*z.to(self.device)) # TanhNormal distribution as actions; reparameterization trick
        action = self.action_range*action_0
        # The log-likelihood here is for the TanhNorm distribution instead of only Gaussian distribution. \
        # The TanhNorm forces the Gaussian with infinite action range to be finite. \
        # For the three terms in this log-likelihood estimation: \
        # (1). the first term is the log probability of action as in common \
        # stochastic Gaussian action policy (without Tanh); \
        # (2). the second term is the caused by the Tanh(), \
        # as shown in appendix C. Enforcing Action Bounds of https://arxiv.org/pdf/1801.01290.pdf, \
        # the epsilon is for preventing the negative cases in log; \
        # (3). the third term is caused by the action range I used in this code is not (-1, 1) but with \
        # an arbitrary action range, which is slightly different from original paper.
        log_prob = Normal(mean, std).log_prob(mean+ std*z.to(self.device)) - torch.log(1. - action_0.pow(2) + epsilon) -  np.log(self.action_range)
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action); 
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability, 
        # needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob, z, mean, log_std
        
    
    def get_action(self, state, deterministic):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z      = normal.sample(mean.shape).to(self.device)
        action = self.action_range* torch.tanh(mean + std*z)
        
        action = self.action_range* torch.tanh(mean).detach().cpu().numpy()[0] if deterministic else action.detach().cpu().numpy()[0]
        return action


    def sample_action(self,):
        a=torch.FloatTensor(self.num_actions).uniform_(-1, 1)
        return self.action_range*a.numpy()


class SAC():
    def __init__(self, replay_buffer, hidden_dim, action_range, args, observation_space, action_space, device):
        self.replay_buffer = replay_buffer

        self.state_dim = observation_space
        self.action_dim = action_space

        self.soft_q_net1 = SoftQNetwork(self.state_dim, self.action_dim, hidden_dim).to(device)
        self.soft_q_net2 = SoftQNetwork(self.state_dim, self.action_dim, hidden_dim).to(device)
        self.target_soft_q_net1 = SoftQNetwork(self.state_dim, self.action_dim, hidden_dim).to(device)
        self.target_soft_q_net2 = SoftQNetwork(self.state_dim, self.action_dim, hidden_dim).to(device)
        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim, hidden_dim, action_range, device).to(device)
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)
        print('Soft Q Network (1,2): ', self.soft_q_net1)
        print('Policy Network: ', self.policy_net)

        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        soft_q_lr = 3e-4
        policy_lr = 3e-4
        alpha_lr  = 3e-4

        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=soft_q_lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        if args.env == "reacher":
            self.test_env = gym.make("Reacher-v4") if args.domain == "source" else gym.make("Reacher-3joints")
        elif args.env == "cheetah":
            self.test_env = gym.make("HalfCheetah-v4", exclude_current_positions_from_observation=False) if args.domain == "source" else gym.make("HalfCheetah-3legs")
            
        self.args = args
        self.device = device

    
    def update(self, batch_size, reward_scale=10., auto_entropy=True, target_entropy=-2, gamma=0.99,soft_tau=1e-2):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        # print('sample:', state, action,  reward, done)

        state      = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action     = torch.FloatTensor(action).to(self.device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(self.device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        predicted_q_value1 = self.soft_q_net1(state, action)
        predicted_q_value2 = self.soft_q_net2(state, action)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)
        new_next_action, next_log_prob, _, _, _ = self.policy_net.evaluate(next_state)
        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem
    # Updating alpha wrt entropy
        # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q) 
        if auto_entropy is True:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            wandb.log({"alpha_loss": alpha_loss})
            # print('alpha loss: ',alpha_loss)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = 1.
            alpha_loss = 0

    # Training Q Function
        target_q_min = torch.min(self.target_soft_q_net1(next_state, new_next_action),self.target_soft_q_net2(next_state, new_next_action)) - self.alpha * next_log_prob
        target_q_value = reward + (1 - done) * gamma * target_q_min # if done==1, only reward
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())  # detach: no gradients for the variable
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())
        wandb.log({"q1_value_loss": q_value_loss1})
        wandb.log({"q2_value_loss": q_value_loss2})


        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()  

    # Training Policy Function
        predicted_new_q_value = torch.min(self.soft_q_net1(state, new_action),self.soft_q_net2(state, new_action))
        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()
        wandb.log({"policy_loss": policy_loss})

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # print('q loss: ', q_value_loss1, q_value_loss2)
        # print('policy loss: ', policy_loss )


    # Soft update the target value net
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        return predicted_new_q_value.mean()


    def evaluate(self, eval_episode=10):
        print("==============================================")
        print("Evaluating...")
        all_rewards = []
        for i in range(eval_episode):
            score = 0
            state = self.test_env.reset(seed = self.args.seed * i)[0]
            for _ in range(self.test_env.spec.max_episode_steps):
                action = self.policy_net.get_action(state, deterministic = True)
                next_state, reward, terminated, truncated, _ = self.test_env.step(action)
                done = truncated or terminated
                
                score += reward
                if done: break
                state = next_state

            all_rewards.append(score)
        avg = sum(all_rewards) / eval_episode
        print(f"average score: {avg}")
        print("==============================================")
        return avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int, nargs='?', default=10000)
    parser.add_argument("--seed", type=int, nargs='?', default=2)
    parser.add_argument("--device", type=str, nargs='?', default="cuda")
    parser.add_argument("--domain", type=str, nargs='?', default="source")
    parser.add_argument("--env", type=str, nargs='?', default="cheetah")
    parser.add_argument("--hidden_dim", type=int, nargs='?', default=256)
    
    args = parser.parse_args()
    seed_everything(args.seed)
    device = args.device
    
    #os.environ["WANDB_DIR"] = "/media/HDD1"
    wandb.init(project="cdpc", name = f'{args.env} {str(args.seed)}: {args.domain} policy', tags=["policy"])

    ##### Loading source domain policy #####
    print("##### training source domain policy #####")
    if args.env == "reacher":
        env = gym.make("Reacher-v4") if args.domain == "source" else gym.make("Reacher-3joints")
    elif args.env == "cheetah":
        env = gym.make("HalfCheetah-v4", exclude_current_positions_from_observation=False) if args.domain == "source" else gym.make("HalfCheetah-3legs")

    # Params
    batch_size = 256
    
    wandb.config = {
        "batch_size": batch_size
    }
    wandb.config.update() 


    replay_buffer_size = 1e6
    replay_buffer = ReplayBuffer(replay_buffer_size)
    hidden_dim = args.hidden_dim
    action_range = 10.0 if args.env=="reacher" else 1.0
    DETERMINISTIC=False
    AUTO_ENTROPY=True
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    max_episode_steps = env.spec.max_episode_steps

    agent = SAC(replay_buffer, hidden_dim=hidden_dim, action_range=action_range, args=args, observation_space=state_dim, action_space=action_dim, device=device)
    location = f'./models/{args.env}/seed_{str(args.seed)}'

    os.makedirs(location, exist_ok=True)
    if not os.path.exists(f'{location}/{str(args.seed)}_{args.env}_{args.domain}.pth'):
        test_score = agent.evaluate()
        wandb.log({"episode": 0, "test/score": test_score})

        for episode in range(1, args.ep+1):
            score = 0
            state = env.reset(seed = args.seed * episode)[0]
            for time_steps in range(max_episode_steps):
                if random.uniform(0, 1) < 0.1: 
                    action = np.random.uniform(low=-1, high=1, size=(env.action_space.shape[0],))
                else:
                    action = agent.policy_net.get_action(state, deterministic = DETERMINISTIC)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = truncated or terminated

                replay_buffer.push(state, action, reward, next_state, done)

                state = next_state
                score += reward

                if len(replay_buffer) > batch_size:
                    _ = agent.update(batch_size, reward_scale=10., auto_entropy=AUTO_ENTROPY, target_entropy=-1.*action_dim)

                if done: break
            
            print("episode:{}, Return:{}, buffer_capacity:{}".format(episode, score, len(replay_buffer)))
            wandb.log({"episode": episode, "train/score": score})
            
            ## test
            if episode % 20 == 0: 
                test_score = agent.evaluate()
                wandb.log({"episode": episode, "test/score": test_score})

            ## save model
            if episode % 100 == 0:
                torch.save(agent.policy_net.state_dict(), f'{location}/{str(args.seed)}_{args.env}_{args.domain}.pth')
                torch.save(agent.soft_q_net1.state_dict(), f'{location}/{str(args.seed)}_{args.env}_{args.domain}_Q_function.pth')

            ## save medium
            if episode == args.ep // 2:
                torch.save(agent.policy_net.state_dict(), f'{location}/{str(args.seed)}_{args.env}_{args.domain}_medium.pth')
        env.close()

        ## store replay buffer
        if args.domain == 'target':
            with open(f'train_set/{args.env}_seed_{str(args.seed)}_replay.pkl', 'wb') as f:
                pickle.dump(replay_buffer, f)