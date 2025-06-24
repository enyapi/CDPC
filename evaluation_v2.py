import torch
import numpy as np
import argparse
import warnings
from sac_v2 import PolicyNetwork
from MPC_DM_model import MPC_DM, Dynamic_Model
from MPC_v2 import MPC, ReplayBuffer_traj
from utils import seed_everything
import random
import gymnasium as gym
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

def reacher_target_R(states, actions):
    """
    Calculate rewards for a batch of states and actions
    states: [batch_size, h+1, state_dim]
    actions: [batch_size, h, action_dim]
    """
    # Get the last 3 dimensions of states for all timesteps
    vec = states[..., -3:]  # [batch_size, h+1, 3]
    
    # Calculate distance reward for all timesteps
    reward_dist = -torch.norm(vec, dim=-1)  # [batch_size, h+1]
    
    # Calculate control reward for all timesteps
    reward_ctrl = -torch.square(actions).sum(dim=-1)  # [batch_size, h]
    
    # Sum rewards across timesteps, handling the different lengths
    total_reward = (reward_dist + reward_ctrl).sum(dim=-1)  # [batch_size]
    
    return total_reward

def calculate_preference_accuracy(mpc, buffer, num_comparisons=1000):
    """
    Calculate preference accuracy by comparing pairs of trajectories
    """
    print("\nCalculating preferences...")
    correct_predictions = 0
    total_predictions = 0
    
    for i in range(num_comparisons):
        # Sample two random trajectories
        indices = random.sample(range(buffer.buffer_len()), 2)
        traj_a, traj_b = buffer.sample(indices)
        
        # Get the actual preference (based on total rewards)
        actual_preference = 1 if traj_a['total_rewards'] > traj_b['total_rewards'] else 0
        
        # Get the sampled trajectories
        state_a = torch.tensor(traj_a['states'], dtype=torch.float32).to(mpc.device)
        action_a = torch.tensor(traj_a['actions'], dtype=torch.float32).to(mpc.device)
        state_b = torch.tensor(traj_b['states'], dtype=torch.float32).to(mpc.device)
        action_b = torch.tensor(traj_b['actions'], dtype=torch.float32).to(mpc.device)
        
        # Add batch dimension and ensure sequence lengths match
        state_a = state_a[:-1].unsqueeze(0)  # [1, h, state_dim]
        action_a = action_a.unsqueeze(0)  # [1, h, action_dim]
        state_b = state_b[:-1].unsqueeze(0)  # [1, h, state_dim]
        action_b = action_b.unsqueeze(0)  # [1, h, action_dim]
        
        # Set batch size to 1
        mpc.batch_size = 1
        
        # Calculate source domain returns using pref_calculation
        R_s_a = mpc.pref_calculation(state_a, action_a)
        R_s_b = mpc.pref_calculation(state_b, action_b)
        
        # 4. Measure preference accuracy
        predicted_preference = 1 if R_s_a > R_s_b else 0
        
        if predicted_preference == actual_preference:
            correct_predictions += 1
        total_predictions += 1
        
        if i % 100 == 0:
            print(f'{i}: Acc: {correct_predictions/total_predictions:.3f}')
            print(f'Actual Returns: {traj_a["total_rewards"]:.3f} vs {traj_b["total_rewards"]:.3f}')
            print(f'Predicted Returns: {R_s_a.item():.3f} vs {R_s_b.item():.3f}')
            print(f'Actual Preference: {actual_preference}')
            print(f'Predicted Preference: {predicted_preference}')
            print('---')
    
    final_acc = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"\nFinal Preference Accuracy: {final_acc:.3f}")
    return final_acc


def evaluate(mpc, num_episodes=10):
    rewards1 = []  # For env reward
    rewards2 = []  # For function reward
    best_rewards_list = []  # For best rewards
    
    for episode in range(num_episodes):
        env_target = gym.make(mpc.target_env, render_mode='rgb_array')
        mpc.action_projector.eval()
        mpc.state_projector.eval()

        s0, _ = env_target.reset(seed=args.seed)
        total_reward = 0
        total_reward2 = 0
        best_rewards = 0
        pref_accs = []
        
        for i in range(env_target.spec.max_episode_steps):
        ## MPC inference
            if dynamic == "dm":
                s, a = mpc._MPC__sampleTraj(s0)
            elif dynamic == "env":
                s, a, r = mpc._MPC__sampleTraj_TrueEnv(s0)

            a0_target, best_reward, _ = mpc._MPC__decodeTraj(s, a)

        ## Preference accuracy
            # eval_freq = 5
            # if episode == 0 and i % eval_freq == 0:
            #     ## 1. Save trajs to buffer
            #     buffer = ReplayBuffer_traj()
            #     s = s[:, :-1]
            #     total_rewards = reacher_target_R(s, a)
            #     for j in range(s.shape[0]):
            #         buffer.push(total_rewards[j].item(), 
            #                 s[j].cpu().detach().numpy(),
            #                 a[j].cpu().detach().numpy(),
            #                 np.array([]), np.array([]), np.array([]))
            #     print(f"Stored {s.shape[0]} trajectories in buffer")

            #     ## 2 & 3. Calculate rewards and source domain returns
            #     pref_acc = calculate_preference_accuracy(mpc, buffer) 
            #     pref_accs.append(pref_acc)
            #     print(f"step {i}, preference accuracy: {pref_acc}") 

        ## Take step
            best_idx = torch.argmax(r)
            #print(best_idx)
            true_action = a[best_idx, 0, :].cpu().detach().numpy()
            a0_target = a0_target.cpu().detach().numpy() 
            s1, r1, terminated, truncated, _ = env_target.step(true_action)
            
        ## Convert numpy arrays to PyTorch tensors
            s0_tensor = torch.tensor(s0, dtype=torch.float32, device=mpc.device)
            a0_tensor = torch.tensor(a0_target, dtype=torch.float32, device=mpc.device)
            s0_tensor = s0_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, state_dim]
            a0_tensor = a0_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, action_dim]
            r2 = reacher_target_R(s0_tensor, a0_tensor).item()

            total_reward += r1
            total_reward2 += r2
            best_rewards += best_reward
            s0 = s1
            
            if terminated or truncated:
                break
        
        rewards1.append(total_reward)
        rewards2.append(total_reward2)
        best_rewards_list.append(best_rewards)
        print(f"Episode {episode + 1}: Env Reward: {total_reward:.3f}")
    
    avg_reward1 = np.mean(rewards1)
    std_reward1 = np.std(rewards1)
    avg_reward2 = np.mean(rewards2)
    std_reward2 = np.std(rewards2)
    avg_best_reward = np.mean(best_rewards_list)
    std_best_reward = np.std(best_rewards_list)
    
    print("\nFinal Results:")
    print(f"Average Env Reward: {avg_reward1:.3f} ± {std_reward1:.3f}")
    #print(f"Average Function Reward: {avg_reward2:.3f} ± {std_reward2:.3f}")
    #print(f"Average Best Reward: {avg_best_reward:.3f} ± {std_best_reward:.3f}")

    # Plot results
    # plt.plot(np.arange(len(pref_accs)) * eval_freq, pref_accs)
    # plt.xlabel('Step')
    # plt.ylabel('Acc (%)')
    # plt.title(f'Preference Accuracy Over Time ({dynamic})')

    # plt.savefig(f'pref_acc_{mpc.h}{dynamic}.png')
    # plt.close()
    
    return avg_reward1, avg_reward2, avg_best_reward

def evaluation_mpc(mpc):
    total_reward = 0
    env = gym.make(mpc.target_env)
    state, _ = env.reset(seed=args.seed)
    for i in range(env.spec.max_episode_steps):
        #env.render()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(mpc.device)
        action = mpc_dm.mpc_policy_net(state).cpu().detach().numpy().squeeze(0)
        #action = trained_agent.get_action(state, deterministic=True)
        #action = env.action_space.sample()
        next_state, reward, done, _, _ = env.step(action)

        total_reward += reward
        if done: break
        state = next_state
    return total_reward

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--env", type=str, default="reacher")
    parser.add_argument("--expert_ratio", type=float, default=1.0)
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_ep", type=int, default=500)
    
    args = parser.parse_args()
    seed_everything(args.seed)
    
    # Set up paths
    location = f'./models/{args.env}/seed_{str(args.seed)}/'
    mpc_location = f'{location}/expert_ratio_{args.expert_ratio}/'
    
    # Set environment parameters
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
    
    print("Loading source domain policy...")
    agent_source = PolicyNetwork(source_s_dim, source_a_dim, hidden_dim, action_range, args.device).to(args.device)
    agent_source.load_state_dict(torch.load(f'{location}/{str(args.seed)}_{args.env}_source.pth', map_location=args.device))
    
    print("Loading MPC policy and Dynamic Model...")
    mpc_dm = MPC_DM(target_s_dim, target_a_dim, args.device)
    source_dynamics_model = Dynamic_Model(source_s_dim + source_a_dim, source_s_dim).to(args.device)
    
    # Load trained models
    mpc_dm.mpc_policy_net.load_state_dict(torch.load(f'{mpc_location}/{str(args.seed)}_MPCModel.pth', map_location=args.device))
    mpc_dm.dynamic_model.load_state_dict(torch.load(f'{mpc_location}/{str(args.seed)}_DynamicModel.pth', map_location=args.device))
    source_dynamics_model.load_state_dict(torch.load(f'{mpc_location}/{str(args.seed)}_DynamicModel_source.pth', map_location=args.device))
    
    # Set up MPC parameters
    params = {
        'batch_size': 32,
        'lr': 0.001,
        'source_env': source_env,
        'target_env': target_env,
        'source_state_space_dim': source_s_dim,
        'source_action_space_dim': source_a_dim,
        'target_state_space_dim': target_s_dim,
        'target_action_space_dim': target_a_dim,
        'agent': agent_source,
        "mpc_dm": mpc_dm,
        "source_dynamics_model": source_dynamics_model,
        "device": args.device,
        "seed": args.seed,
        "env": args.env,
    }
    
    # Create MPC instance
    mpc = MPC(**params)
    
    # Load state and action projectors into MPC instance
    mpc.state_projector.load_state_dict(torch.load(f'{mpc_location}/{str(args.seed)}_state_projector_{args.num_ep}.pth', map_location=args.device))
    mpc.action_projector.load_state_dict(torch.load(f'{mpc_location}/{str(args.seed)}_action_projector_{args.num_ep}.pth', map_location=args.device))
    
    print("\nStarting evaluation...")
    # Evaluate preference accuracy
    # Evaluate CDPC reward
    mpc.h = 10
    dynamic = "env" #"dm"env
    total_reward1, total_reward2, best_rewards = evaluate(mpc)

    total_reward = evaluation_mpc(mpc)
    print(f"mpc policy reward: {total_reward}")
    