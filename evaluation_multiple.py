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


def __sampleTraj(state, mpc_dms, DM, target_a_dim, num_BC, h=50): ## 0.008s
    
    traj_state = torch.zeros((num_BC, h+1, state.shape[-1]), dtype=torch.float32, device='cuda')
    traj_action = torch.zeros((num_BC, h, target_a_dim), dtype=torch.float32, device='cuda')

    state = torch.tensor(state, dtype=torch.float32, device='cuda')
    traj_state[:, 0, :] = state 

    actions = torch.zeros((num_BC, target_a_dim), device='cuda')
    for ith, mpc_dm in enumerate(mpc_dms):
        mpc_dm.mpc_policy_net.eval()
        actions[ith, :] = mpc_dm.mpc_policy_net(state)

    traj_action[:, 0, :] = actions

    states = state.repeat(num_BC, 1)

    for j in range(1, h):
        states = DM(torch.cat([states, actions], dim=-1))
        traj_state[:, j, :] = states

        actions = torch.zeros((num_BC, target_a_dim), device='cuda')
        for ith in range(num_BC):
            actions[ith, :] = mpc_dm.mpc_policy_net(states[ith, :])

        traj_action[:, j, :] = actions

    states = DM(torch.cat([states, actions], dim=-1))
    traj_state[:, h, :] = states

    return traj_state, traj_action

def __decodeTraj(s, a, action_projector:Action_Projector, state_projector:State_Projector, agent, agent_Q, num_BC, h=50, with_Q=False): ## 0.17

    def make_env(seed):
        def _init():
            if args.env == "cheetah":
                env = gym.make(source_env, exclude_current_positions_from_observation=False).unwrapped
            else:
                env = gym.make(source_env).unwrapped
            env.reset(seed=seed)
            return env
        return _init

    with torch.no_grad():
        env_fns = [make_env(args.seed) for _ in range(num_BC)]
        vec_env = DummyVecEnv(env_fns)
        rewards = np.zeros(num_BC)

        for j in range(h):
            source_states = state_projector(s[:, j, :])
            source_actions = action_projector(s[:, j, :], a[:, j, :])

            for i, env in enumerate(vec_env.envs):
                env.reset_specific(state=source_states[i].cpu().detach().numpy())

            if args.env == "reacher":
                r = np.array([
                    reacher_source_R(source_states[i], source_actions[i]).cpu().detach().numpy()
                    for i in range(num_BC)
                ])
            elif args.env == "cheetah":
                obs, r, _, _ = vec_env.step(source_actions.cpu().detach().numpy())
                r = np.array([
                    cheetah_source_R(source_states[i], source_actions[i], obs[i]).cpu().detach().numpy()
                    for i in range(num_BC)
                ])

            rewards += r * (0.99 ** j)

        # add a value from value function
        if with_Q:
            actions = agent.get_action(torch.tensor(obs, dtype=torch.float), deterministic=True)
            rewards += (0.99 ** h) * agent_Q(torch.tensor(obs, dtype=torch.float), torch.tensor(actions)).cpu().detach().numpy().squeeze()

        best_idx = np.argsort(rewards)[-1]
        best_action = a[best_idx, 0, :]
    return best_action, rewards[best_idx], best_idx

def evaluate(seed, env, mpc_dms, DM, target_a_dim, action_projector, state_projector, agent, agent_Q, num_BC=10, h=50, with_MPC=True, with_Q=False):
    env_target = gym.make(env, render_mode='rgb_array') #rgb_array
    #env_target = gym.wrappers.RecordVideo(env_target, 'video')

    s0, _ = env_target.reset(seed=seed)
    total_reward = 0
    counter = 0
    best_idx = -1
    for i in range(env_target.spec.max_episode_steps):
        ## MPC inference
        # generate action sequence using policy network and get state sequence

        if not with_MPC:
            if counter % h == 0:
                states, actions = __sampleTraj(s0, mpc_dms, DM, target_a_dim, num_BC, h)
                a0_target, best_reward, best_idx = __decodeTraj(states, actions, action_projector, state_projector, agent, agent_Q, num_BC, h)
                a0_target = a0_target.cpu().detach().numpy()
                counter = 1
            else:
                a0_target = mpc_dms[best_idx].mpc_policy_net(torch.tensor(s0, dtype=torch.float32, device='cuda'))
                a0_target = a0_target.cpu().detach().numpy()
                counter += 1
        else:
            states, actions = __sampleTraj(s0, mpc_dms, DM, target_a_dim, num_BC, h)
            a0_target, best_reward, best_idx = __decodeTraj(states, actions, action_projector, state_projector, agent, agent_Q, num_BC, h, with_Q)
            a0_target = a0_target.cpu().detach().numpy()


        s1, r1, terminated, truncated, _ = env_target.step(a0_target)
        done = truncated or terminated
        total_reward += r1
        s0 = s1
        
        if done: break
    # print(i)
    # print("best reward:", best_rewards)
    # print("total reward:", total_reward)
    # print()
    return total_reward

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
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
    agent_Q = SoftQNetwork(source_s_dim, source_a_dim, hidden_dim)
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
    # load multiple mpcs
    mpc_dms = []
    DM = Dynamic_Model(target_s_dim + target_a_dim, target_s_dim).to('cuda')
    DM.load_state_dict(torch.load( f'models_multiple/{args.env}_{str(args.seed)}/{str(args.seed)}_DynamicModel.pth', weights_only=True, map_location=args.device ))
    print("##### Loading MPC policy and Dynamic Model #####")
    for ith in range(num_BC):
        mpc_dm = MPC_DM(target_s_dim, target_a_dim, args.device)
        mpc_dm.mpc_policy_net.load_state_dict(torch.load( f'models_multiple/{args.env}_{str(args.seed)}/{str(args.seed)}_MPCModel_{ith}.pth', weights_only=True, map_location=args.device ))
        mpc_dm.dynamic_model.load_state_dict(torch.load( f'models_multiple/{args.env}_{str(args.seed)}/{str(args.seed)}_DynamicModel_{ith}.pth', weights_only=True, map_location=args.device ))
        mpc_dms.append(mpc_dm)
    #---------------------------------------------------------------------------------------------------------------------

    action_projector = Action_Projector(target_s_dim, target_a_dim, source_a_dim).to(args.device)
    state_projector = State_Projector(target_s_dim, source_s_dim).to(args.device)

    action_projector.load_state_dict(torch.load(f'models/{args.env}/seed_{str(args.seed)}/expert_ratio_{args.expert_ratio}/{str(args.seed)}_action_projector_500.pth', weights_only=True, map_location='cuda'))
    state_projector.load_state_dict(torch.load(f'models/{args.env}/seed_{str(args.seed)}/expert_ratio_{args.expert_ratio}/{str(args.seed)}_state_projector_500.pth', weights_only=True, map_location='cuda'))

    num_BC = len(mpc_dms)
    horizon = 5
    with_MPC = False
    with_Q = False

    avg_scores = []
    for BC_ID in range(num_BC):
        total = 0
        for i in range(10):
            reward = evaluate(seed=args.seed*(i+1), env=target_env, mpc_dms=mpc_dms[BC_ID:BC_ID+1], DM=DM, target_a_dim=target_a_dim, action_projector=action_projector, state_projector=state_projector, agent=agent, agent_Q=agent_Q, num_BC=1, h=horizon, with_MPC=with_MPC, with_Q=with_Q)
            # print(reward)
            total += reward
        avg = total / 10
        avg_scores.append(avg)
        print(f'BC {BC_ID}: {avg}')
    print(f'avg. over 10 BC models: {sum(avg_scores) / num_BC}')
    print(f'best BC: {max(avg_scores)}')

    total = 0
    for i in range(10):
        reward = evaluate(seed=args.seed*(i+1), env=target_env, mpc_dms=mpc_dms, DM=DM, target_a_dim=target_a_dim, action_projector=action_projector, state_projector=state_projector, agent=agent, agent_Q=agent_Q, num_BC=num_BC, h=horizon, with_MPC=with_MPC, with_Q=with_Q)
        total += reward
        print(f'{i}: {reward}')

    print(f'avg. return of CDPC: {total/10}')