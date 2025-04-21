import pickle
import random
import numpy as np
import os
import torch
import collections

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False     

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


###############################################################
####################### Data Transition #######################
###############################################################


def collect_d4rl_data(agent_target, target_env, n_traj, expert_ratio, device, seed):
    """
    Collect data in D4RL format from the target environment using a mix of expert and random actions.
    
    Args:
        agent_target: The expert policy to use for collecting expert data
        target_env: The environment to collect data from
        n_traj: Number of trajectories to collect
        expert_ratio: Ratio of expert trajectories to collect
        device: Device to use for the agent
        seed: Random seed for reproducibility
        
    Returns:
        A dictionary containing the collected data in D4RL format
    """
    env = gym.make(target_env)
    max_episode_steps = env.spec.max_episode_steps
    
    # Initialize data storage
    data = collections.defaultdict(list)
    trajectory_returns = []
    
    for episode in range(int(n_traj)):
        state, _ = env.reset(seed=seed)
        episode_data = collections.defaultdict(list)
        
        for t in range(max_episode_steps):
            # Use expert policy or random actions based on expert_ratio
            if episode < int(n_traj * expert_ratio):
                #action, _ = agent_target.predict(state, deterministic=False)
                action = agent_target.get_action(state, deterministic=False)
            else:
                action = np.random.uniform(low=-1, high=1, size=(env.action_space.shape[0],))
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Store the transition
            episode_data['observations'].append(state)
            episode_data['actions'].append(action)
            episode_data['rewards'].append(reward)
            episode_data['next_observations'].append(next_state)
            episode_data['terminals'].append(terminated)
            episode_data['timeouts'].append(truncated)
            
            done = terminated or truncated
            state = next_state
            
            if done:
                break

        trajectory_returns.append(np.sum(episode_data['rewards']))
        print(f"episode: {episode}, Return: {trajectory_returns[-1]}")

        # Convert episode data to numpy arrays
        for k in episode_data:
            data[k].append(np.array(episode_data[k]))
    
    env.close()

    print(f"Trajectory Returns - Max: {np.max(trajectory_returns):.2f}, "
          f"Min: {np.min(trajectory_returns):.2f}, "
          f"Mean: {np.mean(trajectory_returns):.2f}, "
          f"Std: {np.std(trajectory_returns):.2f}")
    
    # Convert to D4RL format
    d4rl_data = {
        'observations': np.concatenate([np.array(traj) for traj in data['observations']]),
        'actions': np.concatenate([np.array(traj) for traj in data['actions']]),
        'rewards': np.concatenate([np.array(traj) for traj in data['rewards']]),
        'next_observations': np.concatenate([np.array(traj) for traj in data['next_observations']]),
        'terminals': np.concatenate([np.array(traj) for traj in data['terminals']]),
        'timeouts': np.concatenate([np.array(traj) for traj in data['timeouts']]),
    }
    
    return d4rl_data


def get_top_k_trajs(d4rl_data, traj_len=50, top_k=10):
    num_transitions = len(d4rl_data['rewards'])
    num_trajs = num_transitions // traj_len

    returns = []
    for i in range(num_trajs):
        start = i * traj_len
        end = start + traj_len
        traj_reward = np.sum(d4rl_data['rewards'][start:end])
        returns.append((i, traj_reward))

    top_indices = sorted(returns, key=lambda x: x[1], reverse=True)[:top_k]
    top_indices = [i for i, _ in top_indices]

    top_data = collections.defaultdict(list)
    for i in top_indices:
        start = i * traj_len
        end = start + traj_len
        for key in d4rl_data:
            top_data[key].append(d4rl_data[key][start:end])

    filtered_data = {k: np.concatenate(v, axis=0) for k, v in top_data.items()}

    return filtered_data


def d4rl2Transition(d4rl_data, buffer):
    observations = d4rl_data['observations']
    actions = d4rl_data['actions']
    rewards = d4rl_data['rewards']
    next_observations = d4rl_data['next_observations']
    terminals = d4rl_data['terminals']
    # You can optionally include timeouts as done flags too

    n = len(observations)
    for i in range(n):
        s = observations[i]
        a = actions[i]
        r = rewards[i]
        ns = next_observations[i]
        done = terminals[i]

        buffer.push((s, a, r, ns, done))


def d4rl2Trajs(d4rl_data, traj_buffer, traj_len=50):
    num_transitions = len(d4rl_data['rewards'])
    num_trajs = num_transitions // traj_len

    for i in range(num_trajs):
        start = i * traj_len
        end = start + traj_len

        traj_obs = d4rl_data['observations'][start:end]
        traj_next_obs = d4rl_data['next_observations'][start:end]
        traj_rewards = d4rl_data['rewards'][start:end]

        total_rewards = np.sum(traj_rewards)

        traj_buffer.push(
            total_rewards=total_rewards,
            state=traj_obs,
            next_state=traj_next_obs
        )

    print(f"Saved {traj_buffer.buffer_len()} trajectories into train_set.")  



def load_buffer(data_path):
    with open(data_path, 'rb') as f:
        return pickle.load(f)

def save_buffer(data_path):
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)
