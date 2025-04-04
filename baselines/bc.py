import numpy as np
import torch
from imitation.algorithms import bc
from imitation.data import types
import wandb
import random
import argparse
import os
import gymnasium as gym
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sac_v2 import PolicyNetwork

def collect_target_data(agent_target, env_target, seed, n_traj, expert_ratio, bc_ratio):
    max_episode_steps = env_target.spec.max_episode_steps

    all_trajectories = []
    scores = []

    for episode in range(int(n_traj)):
        episode_data = {"obs": [], "acts": [], "next_obs": [], "dones": []}
        score = 0
        state, _ = env_target.reset(seed=seed*episode)
        for _ in range(max_episode_steps):
            if episode < int(n_traj*expert_ratio):
                action = agent_target.get_action(state, deterministic=True)
            else:
                action = np.random.uniform(low=-1, high=1, size=(env_target.action_space.shape[0],))
            next_state, reward, terminated, truncated, _ = env_target.step(action)
            done = truncated or terminated

            episode_data["obs"].append(state)
            episode_data["acts"].append(action)
            episode_data["next_obs"].append(next_state)
            episode_data["dones"].append(done)

            score += reward
            if done: break
            state = next_state

        scores.append(score)
        all_trajectories.append(episode_data)
        print(f"Episode: {episode}, Return: {score}")
    env_target.close()
    
    ## select top bc_ratio% trajectories
    top_n = int(n_traj * bc_ratio)
    top_indices = np.argsort(scores)[-top_n:]

    selected_obs, selected_acts, selected_next_obs, selected_dones = [], [], [], []
    for idx in top_indices:
        selected_obs.extend(all_trajectories[idx]["obs"])
        selected_acts.extend(all_trajectories[idx]["acts"])
        selected_next_obs.extend(all_trajectories[idx]["next_obs"])
        selected_dones.extend(all_trajectories[idx]["dones"])
    
    print(f"Selected {len(selected_obs)} transitions from top {top_n} trajectories.")
    for i in range(top_n):
        print(scores[top_indices[i]])

    # List to NumPy
    obs_array = np.array(selected_obs, dtype=np.float32)
    action_array = np.array(selected_acts, dtype=np.float32)
    next_obs_array = np.array(selected_next_obs, dtype=np.float32)
    done_array = np.array(selected_dones, dtype=np.bool_)

    # NumPy to PyTorch Tensor
    obs_ = torch.tensor(obs_array, dtype=torch.float32)
    action_ = torch.tensor(action_array, dtype=torch.float32)
    next_obs_ = torch.tensor(next_obs_array, dtype=torch.float32)
    done_ = done_array

    # Create the Transitions object
    infos_ = [{} for _ in range(len(obs_))]
    transitions = types.Transitions(
        obs=obs_,
        acts=action_,
        next_obs=next_obs_,
        dones=done_,
        infos=infos_
    )

    return transitions


def evaluate_policy(policy, env, seed):
    total_reward = 0.0
    state, _ = env.reset(seed=seed)
    for i in range(env.spec.max_episode_steps):
        action, _ = policy.predict(state)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
    return total_reward

def validation(policy, env, seed):
    loss = 0
    return loss
    

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
    parser.add_argument("--n_traj", type=int, nargs='?', default=10000) # 1000/10000
    parser.add_argument("--expert_ratio", type=float, nargs='?', default=0.2) # random_ratio=1-expert_ratio
    parser.add_argument("--ep", type=int, nargs='?', default=500) # the same as cdpc
    parser.add_argument("--device", type=str, nargs='?', default="cuda")
    parser.add_argument("--env", type=str, nargs='?', default="cheetah")
    parser.add_argument("--bc_ratio", type=float, nargs='?', default=0.1) # BC_0.1
    parser.add_argument("--wandb", action='store_true', default=False)
    
    args = parser.parse_args()
    seed_everything(args.seed)
    device = args.device

    # Create a random number generator
    rng = np.random.default_rng(args.seed)
    if args.wandb:
        wandb.init(project="cdpc", name = f'baseline: BC_{args.bc_ratio} {str(args.seed)}_{args.env}', tags = ["baseline", "BC"])
    location = f'./baselines/bc/{args.env}/seed_{str(args.seed)}'

    # Env
    if args.env == "reacher":
        env = gym.make("Reacher-3joints")
    elif args.env == "cheetah":
        env = gym.make("HalfCheetah-3legs")

    # parameters
    batch_size = 32#*2 * env.spec.max_episode_steps
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

    # Initialize Behavioral Cloning
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=rng,
        batch_size=batch_size,
        device=device,
        optimizer_kwargs={'lr': 0.001},
    )

    # Train BC policy and evaluate after each epoch
    reward = evaluate_policy(bc_trainer.policy, env, args.seed)
    if args.wandb:
        wandb.log({"episode": 0, "test/score": reward})
    print(f"BC Policy (Epoch {0}): reward={reward:.2f}")
    for epoch in range(args.ep):
        loss = bc_trainer.train(n_batches=1) #n_epochs=1
        reward = evaluate_policy(bc_trainer.policy, env, args.seed)
        if args.wandb:
            wandb.log({"episode": epoch+1, "test/score": reward, "train/loss": loss})
        print(f"BC Policy (Epoch {epoch+1}): reward={reward:.2f}")

    # Save the trained BC policy
    os.makedirs(location, exist_ok=True)
    bc_trainer.policy.save(f'{location}/{str(args.seed)}_{args.env}_bc_{args.bc_ratio}.pth')
