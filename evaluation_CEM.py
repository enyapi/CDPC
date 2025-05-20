import gymnasium as gym
import torch
import random
import numpy as np
import argparse
import os
import warnings
import wandb
from stable_baselines3 import SAC
from sac_v2 import PolicyNetwork, SoftQNetwork
# from sac_v2 import ReplayBuffer
from MPC_v2 import *
from MPC_DM_model import *

def objective_for_CEM(agent_Q, num_sample:int, action:np.ndarray, state:np.ndarray, action_projector:Action_Projector, state_projector:State_Projector, dynamics_model:Dynamic_Model):
    
    action = torch.tensor(action, device=args.device, dtype=torch.float32)
    state = torch.tensor(state, device=args.device, dtype=torch.float32)
    state = state.repeat(num_sample, 1)

    with torch.no_grad():
        source_action = action_projector(state, action)
        source_state = state_projector(state)
        # source_next_state = dynamics_model(torch.cat([source_state, source_action], dim=-1))

    val = agent_Q(source_state, source_action).squeeze().cpu().detach().numpy()
    # val = np.zeros(num_sample)
    # for i in range(num_sample):
    #     if args.env == "reacher":
    #         val[i] = reacher_source_R(source_state[i], source_action[i]).cpu().detach().numpy()
    #     elif args.env == "cheetah":
    #         val[i] = cheetah_source_R(source_state[i], source_action[i], source_next_state[i]).cpu().detach().numpy()
    
    return val

def cem(agent_Q,
        dim_action:np.ndarray,
        state,
        action_projector:Action_Projector,
        state_projector:State_Projector,
        dynamics_model:Dynamic_Model,
        num_samples=100000, 
        num_elites=50, 
        max_iters=1000, 
        initial_mean=None, 
        initial_std=1.0, 
        epsilon=1e-5):
    """
    Cross-Entropy Method (CEM) for minimizing a function f(x)
    
    Parameters:
        dim: dimensionality of the input vector x
        num_samples: number of samples per iteration
        num_elites: number of elite samples to update distribution
        max_iters: maximum number of iterations
        initial_mean: initial mean of the distribution (default: zeros)
        initial_std: initial standard deviation of the distribution
        epsilon: convergence threshold on std deviation
    
    Returns:
        best_x: best solution found
        best_score: best objective value found
    """
    mean = np.zeros(dim_action) if initial_mean is None else initial_mean
    std = np.ones(dim_action)
    best_score = -100000
    best_sample = None

    for i in range(max_iters):
        samples = np.clip(np.random.randn(num_samples, dim_action) * std + mean, -1, 1)
        # scores = objective_for_CEM(agent_Q, num_samples, samples, state, action_projector, state_projector, dynamics_model).squeeze().cpu().detach().numpy()
        scores = objective_for_CEM(agent_Q, num_samples, samples, state, action_projector, state_projector, dynamics_model)
        
        elite_indices = scores.argsort()[::-1][:num_elites]  # minimize
        elite_samples = samples[elite_indices]

        mean = elite_samples.mean(axis=0)
        std = elite_samples.std(axis=0)

        # print(scores[elite_indices][0], mean)

        best_index = scores.argmin()
        if best_score < scores[best_index]:
            best_score = scores[best_index]
            best_sample = samples[best_index]

        if np.max(std) < epsilon:
            break
    # print('--------')
    return best_sample, best_score

def evaluate(seed, env, target_a_dim, action_projector, state_projector, dynamics_model, agent, agent_Q):
    env_target = gym.make(env, render_mode='rgb_array') #rgb_array

    state, _ = env_target.reset(seed=seed)
    total_reward = 0
    total_score = 0
    while True:
        ## CEM inference
        action, score = cem(agent_Q, target_a_dim, state, action_projector, state_projector, dynamics_model)
        next_state, reward, terminated, truncated, _ = env_target.step(action)
        done = truncated or terminated
        total_reward += reward
        total_score += score
        state = next_state

        print(score, reward)
        
        if done: break
    print(total_score, total_reward)
    return total_reward

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("MPC_pre_ep", type=int, nargs='?', default=10000)
    parser.add_argument("decoder_batch", type=int, nargs='?', default=32)
    parser.add_argument("--seed", type=int, nargs='?', default=2)
    parser.add_argument("--expert_ratio", type=float, nargs='?', default=1.0) # random_ratio=1-expert_ratio
    parser.add_argument("--device", type=str, nargs='?', default="cuda")
    parser.add_argument("--env", type=str, nargs='?', default="cheetah") # cheetah reacher
    parser.add_argument("--use_flow", action='store_true', default=False)
    parser.add_argument("--num_BC", type=int, default=10)
    
    args = parser.parse_args()

    location = f'./models/{args.env}/seed_{str(args.seed)}/'
    mpc_location = f'{location}/expert_ratio_{args.expert_ratio}/'

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
    num_BC = args.num_BC
    action_range = 10.0 if args.env=="reacher" else 1.0

    ##### 1 Loading source domain policy #####
    print("##### Loading source domain policy #####")
    #agent = SAC.load(f'./experiments/{args.env}_source_18/models/final_model.zip', device=args.device) # SB3
    agent = PolicyNetwork(source_s_dim, source_a_dim, hidden_dim, action_range, args.device).to(args.device)
    agent.load_state_dict(torch.load( f'{location}/{str(args.seed)}_{args.env}_source.pth', weights_only=True, map_location=args.device ))
    agent_Q = SoftQNetwork(source_s_dim, source_a_dim, hidden_dim).to(args.device)
    # agent_Q.load_state_dict(torch.load( f'{location}/{str(args.seed)}_{args.env}_source_Q_function.pth', map_location=args.device ))

    ##### 2 Loading target domain expert policy #####
    print("##### Loading target domain expert policy #####")
    # agent_target = SAC.load('models/cheetah/seed_7/HalfCheetah-3legs_SAC_3_128_200000_2.zip', device=args.device) # SB3
    agent_target = PolicyNetwork(target_s_dim, target_a_dim, hidden_dim, action_range, args.device).to(args.device)
    agent_target_medium = PolicyNetwork(target_s_dim, target_a_dim, hidden_dim, action_range, args.device).to(args.device)
    # target_Q = SoftQNetwork(target_s_dim, target_a_dim, hidden_dim).to(args.device)
    agent_target.load_state_dict(torch.load( f'{location}/{str(args.seed)}_{args.env}_target.pth', weights_only=True, map_location=args.device ))
    agent_target_medium.load_state_dict(torch.load( f'{location}/{str(args.seed)}_{args.env}_target_medium.pth', weights_only=True, map_location=args.device ))
    # target_Q.load_state_dict(torch.load( f'{location}/{str(args.seed)}_{args.env}_target_Q_function.pth', map_location=args.device ))


    ##### 4 Train or Loading MPC policy and Dynamic Model #####
    
    os.makedirs(mpc_location, exist_ok=True)

    #---------------------------------------------------------------------------------------------------------------------
    # load mpc
    mpc_location = f'{location}/expert_ratio_{args.expert_ratio}/'
    mpc_dm = MPC_DM(source_s_dim, source_a_dim, args.device)

    print("##### Loading Dynamics Model #####")
    print(f'{mpc_location}/{str(args.seed)}_DynamicModel.pth loaded')
    mpc_dm.dynamic_model.load_state_dict(torch.load( f'{mpc_location}/{str(args.seed)}_DynamicModel_source.pth', weights_only=True, map_location=args.device ))
    #---------------------------------------------------------------------------------------------------------------------

    action_projector = Action_Projector(target_s_dim, target_a_dim, source_a_dim).to(args.device)
    state_projector = State_Projector(target_s_dim, source_s_dim).to(args.device)

    action_projector.load_state_dict(torch.load(f'models/{args.env}/seed_{str(args.seed)}/expert_ratio_{args.expert_ratio}/{str(args.seed)}_action_projector_500.pth', weights_only=True, map_location='cuda'))
    state_projector.load_state_dict(torch.load(f'models/{args.env}/seed_{str(args.seed)}/expert_ratio_{args.expert_ratio}/{str(args.seed)}_state_projector_500.pth', weights_only=True, map_location='cuda'))

    total = 0
    for i in range(10):
        reward = evaluate(seed=args.seed*(i+1), env=target_env, target_a_dim=target_a_dim, action_projector=action_projector, state_projector=state_projector, dynamics_model=mpc_dm.dynamic_model, agent=agent, agent_Q=agent_Q)
        # print(reward)
        total += reward
    avg = total / 10
    print(f'avg. score of CDPC+CEM: {avg}')
