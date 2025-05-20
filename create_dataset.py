import gymnasium as gym
import torch
import random
import numpy as np
import argparse
import os
from sac_v2 import PolicyNetwork
from MPC_DM_model import ReplayBuffer
# from sac_v2 import ReplayBuffer
from MPC_v2 import ReplayBuffer_traj
import pickle

def collect_data(agent_expert, agent_medium, env, n_traj, expert_ratio, device, seed):
    buffer_maxlen = 1000000
    buffer = ReplayBuffer(buffer_maxlen, device)
    buffer_expert_only = ReplayBuffer(buffer_maxlen, device)
    train_set = ReplayBuffer_traj()

    if env == "HalfCheetah-v4":
        env = gym.make(env, exclude_current_positions_from_observation=False)
    else:
        env = gym.make(env)

    max_episode_steps = env.spec.max_episode_steps 
    for episode in range(int(n_traj)):
        score = 0
        state, _ = env.reset(seed=seed*episode) ############################################
        state_list = []
        action_list = []
        next_state_list = []
        reward_list = []
        done_list = []
        for _ in range(max_episode_steps):
            if episode < int(n_traj*expert_ratio):
                if episode % 2 == 0:
                    # action, _ = agent_target.predict(state, deterministic=True) # SB3
                    action = agent_expert.get_action(state, deterministic=True)
                else:
                    action = agent_medium.get_action(state, deterministic=True)
            else:
                action = np.random.uniform(low=-1, high=1, size=(env.action_space.shape[0],))
                
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = truncated or terminated

            done_mask = 0.0 if done else 1.0
            buffer.push((state, action, reward, next_state, done_mask))

            if episode < int(n_traj*expert_ratio) and episode % 2 == 0:
                buffer_expert_only.push((state, action, reward, next_state, done_mask))

            state_list.append(state)
            action_list.append(action)
            next_state_list.append(next_state)
            reward_list.append(reward)
            done_list.append(done_mask)
            state = next_state
            score += reward
            
            if done: break

        train_set.push(score, state_list, action_list, next_state_list, reward_list, done_list)
        print("episode:{}, Return:{}, buffer_capacity:{}".format(episode, score, buffer.buffer_len()))
    env.close()
    
    print(f"Collected {n_traj} trajectories.")
    print(f"Collected {n_traj*max_episode_steps} transitions.")
    return train_set, buffer, buffer_expert_only

def save_buffer(buffer, filename):
    with open(filename, 'wb') as f:
        pickle.dump(buffer, f)

def load_buffer(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False       

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, nargs='?', default=2)
    parser.add_argument("--n_traj", type=int, nargs='?', default=1000) # 1000/10000
    parser.add_argument("--expert_ratio", type=float, nargs='?', default=0.8) # random_ratio=1-expert_ratio
    parser.add_argument("--device", type=str, nargs='?', default="cuda")
    parser.add_argument("--env", type=str, nargs='?', default="cheetah")

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

    hidden_dim = 256
    action_range = 10.0 if args.env=="reacher" else 1.0
    location = f'./models/{args.env}/seed_{str(args.seed)}/'

    ##### 1 Loading source domain policy #####
    print("##### Loading source domain policy #####")
    agent_source = PolicyNetwork(source_s_dim, source_a_dim, hidden_dim, action_range, args.device).to(args.device)
    agent_source.load_state_dict(torch.load( f'{location}/{str(args.seed)}_{args.env}_source.pth', weights_only=True, map_location=args.device ))
    agent_source_medium = PolicyNetwork(source_s_dim, source_a_dim, hidden_dim, action_range, args.device).to(args.device)
    agent_source_medium.load_state_dict(torch.load( f'{location}/{str(args.seed)}_{args.env}_source_medium.pth', weights_only=True, map_location=args.device ))

    ##### 2 Loading target domain expert policy #####
    print("##### Loading target domain expert policy #####")
    agent_target = PolicyNetwork(target_s_dim, target_a_dim, hidden_dim, action_range, args.device).to(args.device)
    agent_target_medium = PolicyNetwork(target_s_dim, target_a_dim, hidden_dim, action_range, args.device).to(args.device)
    agent_target.load_state_dict(torch.load( f'{location}/{str(args.seed)}_{args.env}_target.pth', weights_only=True, map_location=args.device ))
    agent_target_medium.load_state_dict(torch.load( f'{location}/{str(args.seed)}_{args.env}_target_medium.pth', weights_only=True, map_location=args.device ))

    os.makedirs('./train_set/', exist_ok=True)
    data_path = f'./train_set/{str(args.seed)}_{args.env}_{args.expert_ratio}.pkl'
    target_buffer_path = f'./train_set/{str(args.seed)}_{args.env}_{args.expert_ratio}_target_buffer.pkl'
    target_buffer_expert_path = f'./train_set/{str(args.seed)}_{args.env}_{args.expert_ratio}_target_buffer_expert.pkl'
    source_buffer_path = f'./train_set/{str(args.seed)}_{args.env}_{args.expert_ratio}_source_buffer.pkl'
    source_buffer_expert_path = f'./train_set/{str(args.seed)}_{args.env}_{args.expert_ratio}_source_buffer_expert.pkl'

    if os.path.exists(data_path):
        train_set = load_buffer(data_path)
        target_buffer = load_buffer(target_buffer_path)
        target_buffer_expert_only = load_buffer(target_buffer_expert_path)
        source_buffer = load_buffer(source_buffer_path)
        source_buffer_expert_only = load_buffer(source_buffer_expert_path)

    else:
        train_set, target_buffer, target_buffer_expert_only = collect_data(agent_target, agent_target_medium, target_env, args.n_traj, args.expert_ratio, args.device, args.seed)
        train_set2, source_buffer, source_buffer_expert_only = collect_data(agent_source, agent_source_medium, source_env, args.n_traj, args.expert_ratio, args.device, args.seed)
        
        save_buffer(train_set, data_path)
        save_buffer(target_buffer, target_buffer_path)
        save_buffer(target_buffer_expert_only, target_buffer_expert_path)
        save_buffer(source_buffer, source_buffer_path)
        save_buffer(source_buffer_expert_only, source_buffer_expert_path)