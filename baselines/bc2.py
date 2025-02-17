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

def load_transitions_to_replaybuffer(transitions, replay_buffer):
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, nargs='?', default=2)
    parser.add_argument("--n_traj", type=int, nargs='?', default=1000) # 1000/10000
    parser.add_argument("--expert_ratio", type=float, nargs='?', default=0.2) # random_ratio=1-expert_ratio
    parser.add_argument("--ep", type=int, nargs='?', default=200) # the same as cdpc
    parser.add_argument("--device", type=str, nargs='?', default="cuda")
    parser.add_argument("--env", type=str, nargs='?', default="cheetah")
    parser.add_argument("--bc_ratio", type=float, nargs='?', default=0.1) # BC_0.1
    
    args = parser.parse_args()
    seed_everything(args.seed)
    device = args.device

    # Create a random number generator
    rng = np.random.default_rng(args.seed)

    wandb.init(project="cdpc", name = f'baseline: BC_{args.bc_ratio} {str(args.seed)}_{args.env}')
    location = f'./baselines/bc2/{args.env}/seed_{str(args.seed)}'

    # Env
    if args.env == "reacher":
        env = gym.make("Reacher-3joints")
    elif args.env == "cheetah":
        env = gym.make("HalfCheetah-3legs")

    # parameters
    batch_size = 32*2 * env.spec.max_episode_steps
    hidden_dim = 512
    action_range = 10.0 if args.env=="reacher" else 1.0
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]

    # load expert policy
    trained_agent = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range, device).to(device)
    model_path = f'models/{args.env}/seed_{str(args.seed)}/{str(args.seed)}_{args.env}_target.pth'
    trained_agent.load_state_dict(torch.load( model_path, map_location=device ))

    # Create the Transitions object
    n_traj = args.n_traj
    transitions = collect_target_data(trained_agent, env, args.seed, n_traj, args.expert_ratio, args.bc_ratio)


    # train BC
    from MPC_DM_model import MPC_DM, ReplayBuffer

    replay_buffer = ReplayBuffer(1000000, device)
    load_transitions_to_replaybuffer(transitions, replay_buffer)

    mpc_dm = MPC_DM(state_dim, action_dim, args.device)
    os.makedirs(location, exist_ok=True)
    if not os.path.exists(f'{location}/{str(args.seed)}_{args.env}_bc_{args.bc_ratio}.pth'):
        reward = evaluate_policy(mpc_dm.mpc_policy_net, env, args.seed, args.device)
        wandb.log({"episode": 0, "test/score": reward})
        print(f"BC Policy (Epoch {0}): mean_reward={reward:.2f}")
        for i in range(args.ep):
            loss_mpc, loss_dm = mpc_dm.update(batch_size, replay_buffer)

            reward = evaluate_policy(mpc_dm.mpc_policy_net, env, args.seed, args.device)
            wandb.log({"episode": i+1, "test/score": reward})
            print(f"BC Policy (Epoch {i+1}): mean_reward={reward:.2f}")
        torch.save(mpc_dm.mpc_policy_net.state_dict(), f'{location}/{str(args.seed)}_{args.env}_bc_{args.bc_ratio}.pth')

