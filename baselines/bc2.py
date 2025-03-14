import numpy as np
import torch
import wandb
import random
import argparse
import os
import gymnasium as gym
import sys
from bc import collect_target_data

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sac_v2 import PolicyNetwork

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate_policy(policy, env, seed, device):
    total_reward = 0.0
    state, _ = env.reset(seed=seed)
    for i in range(env.spec.max_episode_steps):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        action = policy(state).cpu().detach().numpy().squeeze(0)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
    return total_reward

def load_transitions_to_replaybuffer(transitions, replay_buffer, N):
    for i in range(len(transitions.obs)):
        done_mask = 0.0 if transitions.dones[i] else 1.0
        experience = (
            np.array(transitions.obs[i]), 
            np.array(transitions.acts[i]), 
            0,
            np.array(transitions.next_obs[i]), 
            done_mask,
        )
        replay_buffer.push(experience)


def split_and_load_transitions(transitions, replay_buffer, other_buffer, N):
    indices = list(range(len(transitions.obs)))  # 取得所有索引
    sampled_indices = set(random.sample(indices, N))  # 隨機選 N 個索引

    for i in indices:
        done_mask = 0.0 if transitions.dones[i] else 1.0
        experience = (
            np.array(transitions.obs[i]), 
            np.array(transitions.acts[i]), 
            0,  # reward 可能需要調整
            np.array(transitions.next_obs[i]), 
            done_mask,
        )

        if i in sampled_indices:
            other_buffer.push(experience)
        else:
            replay_buffer.push(experience)


def val_data(replay_buffer):
    states = np.array([exp[0] for exp in replay_buffer.buffer])  # 取出 state
    actions = np.array([exp[1] for exp in replay_buffer.buffer]) # 取出 action

    # 轉換成 tensor，並搬移到相應的 device
    states_tensor = torch.tensor(states, dtype=torch.float32).to(replay_buffer.device)
    actions_tensor = torch.tensor(actions, dtype=torch.float32).to(replay_buffer.device)

    return states_tensor, actions_tensor

def validation(policy, state, action):
    import torch.nn.functional as F
    with torch.no_grad():
        pred_action = policy(state)
        loss = F.mse_loss(pred_action, action)
    return loss



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, nargs='?', default=2)
    parser.add_argument("--n_traj", type=int, nargs='?', default=1000) # 1000/10000
    parser.add_argument("--expert_ratio", type=float, nargs='?', default=0.2) # random_ratio=1-expert_ratio
    parser.add_argument("--ep", type=int, nargs='?', default=200) # the same as cdpc
    parser.add_argument("--device", type=str, nargs='?', default="cuda")
    parser.add_argument("--env", type=str, nargs='?', default="cheetah")
    parser.add_argument("--bc_ratio", type=float, nargs='?', default=0.1) # BC_0.1
    parser.add_argument("--wandb", action='store_true', default=False)
    parser.add_argument("--vedio", action='store_true', default=False)
    
    args = parser.parse_args()
    seed_everything(args.seed)
    device = args.device

    ## Create a random number generator
    if args.wandb:
        wandb.init(project="cdpc", name = f'baseline: BC2_{args.bc_ratio} {str(args.seed)}_{args.env}', tags = ["baseline", "BC2"])
    location = f'./baselines/bc2/{args.env}/seed_{str(args.seed)}'

    ## Env
    if args.env == "reacher":
        env = gym.make("Reacher-3joints", render_mode="rgb_array")
    elif args.env == "cheetah":
        env = gym.make("HalfCheetah-3legs", render_mode="rgb_array")

    ## parameters
    batch_size = 128#32*2 * env.spec.max_episode_steps
    hidden_dim = 512
    action_range = 10.0 if args.env=="reacher" else 1.0
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]

    ## load expert policy
    trained_agent = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range, device).to(device)
    model_path = f'models/{args.env}/seed_{str(args.seed)}/{str(args.seed)}_{args.env}_target.pth'
    trained_agent.load_state_dict(torch.load( model_path, map_location=device ))

    ## Create the Transitions object
    print("##### Collecting target domain data #####")
    import pickle
    import collections
    def save_replay_buffer(replay_buffer, filename="replay_buffer.pth"):
        with open(filename, "wb") as f:
            pickle.dump(list(replay_buffer.buffer), f)  # 轉換為 list 再存

    def load_replay_buffer(replay_buffer, filename="replay_buffer.pth"):
        with open(filename, "rb") as f:
            replay_buffer.buffer = collections.deque(pickle.load(f), maxlen=replay_buffer.buffer.maxlen)

    from MPC_DM_model import ReplayBuffer, MPC_Policy_Net
    replay_buffer = ReplayBuffer(1000000, device)
    val_buffer = ReplayBuffer(1000000, device)
    n_traj = args.n_traj

    os.makedirs(f'{location}/data', exist_ok=True)
    data_path = f'{location}/data/{str(n_traj)}traj_{args.bc_ratio}bcratio_{args.expert_ratio}expert.pkl'
    data_path_val = f'{location}/data/{str(n_traj)}traj_{args.bc_ratio}bcratio_{args.expert_ratio}expert_val.pkl'
    if os.path.exists(data_path):
        load_replay_buffer(replay_buffer, data_path)
        load_replay_buffer(val_buffer, data_path_val)
    else:
        val_ratio = 0.05
        transitions = collect_target_data(trained_agent, env, args.seed, n_traj, args.expert_ratio, args.bc_ratio+val_ratio)
        split_and_load_transitions(transitions, replay_buffer, val_buffer, int(val_ratio*n_traj*env.spec.max_episode_steps))
        save_replay_buffer(replay_buffer, data_path)
        save_replay_buffer(val_buffer, data_path_val)
    print(replay_buffer.buffer_len())
    print(val_buffer.buffer_len())


    ## train BC
    import torch.nn.functional as F
    import torch.optim as optim

    states, actions = val_data(val_buffer)
    print(states.shape)   # (N, state_size)
    print(actions.shape)  # (N, action_size)

    policy = MPC_Policy_Net(state_dim, action_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3, weight_decay=1e-4)

    reward = evaluate_policy(policy, env, args.seed, args.device)
    val_loss = validation(policy, states, actions)
    if args.wandb:
        wandb.log({"episode": 0, "test/score": reward, "test/loss": val_loss.item()})
    print(f"BC Policy (Epoch {0}): mean_reward={reward:.2f}")

    for i in range(args.ep):
        ## update
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        pred_action = policy(state)
        loss_mpc = F.mse_loss(pred_action, action)
        optimizer.zero_grad()
        loss_mpc.backward()
        optimizer.step()

        if i%100==0 and args.vedio: env = gym.wrappers.RecordVideo(env, f'video/bc2')
        reward = evaluate_policy(policy, env, args.seed, args.device)
        val_loss = validation(policy, states, actions)
        if args.wandb:
            wandb.log({"episode": i+1, "test/score": reward, "train/loss": loss_mpc.item(), "test/loss": val_loss.item()})
        print(f"BC Policy (Epoch {i+1}): mean_reward={reward:.2f}, train_loss={loss_mpc.item():.2f}, val_loss={val_loss.item():.2f}")

    torch.save(policy.state_dict(), f'{location}/{str(args.seed)}_{args.env}_bc2_{args.bc_ratio}.pth')

