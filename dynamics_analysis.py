import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo
import argparse
from sac_v2 import PolicyNetwork
from MPC_DM_model import MPC_DM, Dynamic_Model
from MPC_v2 import MPC
from utils import seed_everything

def analyze_dynamics(mpc, num_episodes=5):
    """
    Analyze the difference between trajectories from dynamics model and true environment
    """
    print("Analyzing dynamics...")
    
    # Create environments with video recording
    env_true = gym.make(mpc.target_env, render_mode='rgb_array')
    #env_true = RecordVideo(env_true, 'videos/true_dynamics')
    
    # Lists to store errors
    state_errors = []
    action_errors = []
    
    print(f"\nRunning episode with seed {mpc.seed}")
    state, _ = env_true.reset(seed=mpc.seed)
    
    # Generate trajectory using dynamics model
    s_dm = torch.tensor(state, dtype=torch.float32, device=mpc.device).unsqueeze(0)  # [1, state_dim]
    traj_state_dm = torch.zeros((1, mpc.h + 1, mpc.target_state_space_dim), device=mpc.device)
    traj_action_dm = torch.zeros((1, mpc.h, mpc.target_action_space_dim), device=mpc.device)
    traj_state_dm[:, 0, :] = s_dm
    
    # Generate trajectory using true environment
    traj_state_true = [state]
    traj_action_true = []
    
    # Generate trajectories
    for j in range(mpc.h):
        # Dynamics model trajectory
        a_dm = mpc.mpc_policy_net(s_dm)
        traj_action_dm[:, j, :] = a_dm
        s_dm = mpc.dynamics_model(torch.cat([s_dm, a_dm], dim=-1))
        traj_state_dm[:, j + 1, :] = s_dm
        
        # True environment trajectory
        a_true = a_dm[0].cpu().detach().numpy()
        traj_action_true.append(a_true)
        next_state, _, terminated, truncated, _ = env_true.step(a_true)
        traj_state_true.append(next_state)
        
        if terminated or truncated:
            break
    
    # Convert trajectories to numpy for comparison
    traj_state_dm = traj_state_dm[0].cpu().detach().numpy()
    traj_action_dm = traj_action_dm[0].cpu().detach().numpy()
    traj_state_true = np.array(traj_state_true)
    traj_action_true = np.array(traj_action_true)
    
    # Calculate errors
    state_errors = np.linalg.norm(traj_state_dm[:len(traj_state_true)] - traj_state_true, axis=1)
    action_errors = np.linalg.norm(traj_action_dm[:len(traj_action_true)] - traj_action_true, axis=1)
    
    # Print results
    print("\nTrajectory Analysis Results:")
    print(state_errors)
    print(f"Mean State Error: {np.mean(state_errors):.3f} ± {np.std(state_errors):.3f}")
    print(f"Mean Action Error: {np.mean(action_errors):.3f} ± {np.std(action_errors):.3f}")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Plot 1: State errors over time
    plt.subplot(131)
    plt.plot(state_errors, label='State Error')
    plt.xlabel('Step')
    plt.ylabel('Error')
    plt.title('State Prediction Error Over Time')
    
    # Plot 2: Action errors over time
    plt.subplot(132)
    plt.plot(action_errors, label='Action Error')
    plt.xlabel('Step')
    plt.ylabel('Error')
    plt.title('Action Prediction Error Over Time')
    
    # Plot 3: Error distributions
    plt.subplot(133)
    plt.hist(state_errors, bins=30, alpha=0.5, label='State')
    plt.hist(action_errors, bins=30, alpha=0.5, label='Action')
    plt.xlabel('Error')
    plt.ylabel('Count')
    plt.title('Error Distributions')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('dynamics_analysis.png')
    plt.close()
    
    return state_errors, action_errors

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--env", type=str, default="reacher")
    parser.add_argument("--expert_ratio", type=float, default=1.0)
    parser.add_argument("--num_episodes", type=int, default=5)
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
    mpc.h = 50
    
    # Load state and action projectors into MPC instance
    mpc.state_projector.load_state_dict(torch.load(f'{mpc_location}/{str(args.seed)}_state_projector_{args.num_ep}.pth', map_location=args.device))
    mpc.action_projector.load_state_dict(torch.load(f'{mpc_location}/{str(args.seed)}_action_projector_{args.num_ep}.pth', map_location=args.device))
    
    print("\nStarting dynamics analysis...")
    state_errors, action_errors = analyze_dynamics(mpc, num_episodes=args.num_episodes) 