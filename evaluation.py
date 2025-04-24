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
from MPC_DM_model import ReplayBuffer, MPC_DM
# from sac_v2 import ReplayBuffer
from MPC import ReplayBuffer_traj, MPC

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("MPC_pre_ep", type=int, nargs='?', default=10000)
    parser.add_argument("decoder_batch", type=int, nargs='?', default=32)
    parser.add_argument("--seed", type=int, nargs='?', default=2)
    parser.add_argument("--expert_ratio", type=float, nargs='?', default=0.2) # random_ratio=1-expert_ratio
    parser.add_argument("--device", type=str, nargs='?', default="cuda")
    parser.add_argument("--env", type=str, nargs='?', default="cheetah") # cheetah reacher
    parser.add_argument("--use_flow", action='store_true', default=False)
    parser.add_argument("--trained_MPC", type=bool, nargs='?', default=False)
    parser.add_argument("--horizon", type=int, nargs='?', default=20)
    
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
    mpc_dm = MPC_DM(target_s_dim, target_a_dim, args.device)
    os.makedirs(mpc_location, exist_ok=True)
    
    if args.trained_MPC:
        print("##### Loading MPC policy and Dynamic Model #####")
        print(f'{mpc_location}/{str(args.seed)}_DynamicModel.pth loaded')
        mpc_dm.mpc_policy_net.load_state_dict(torch.load( f'{mpc_location}/{str(args.seed)}_MPCModel.pth', weights_only=True, map_location=args.device ))
        mpc_dm.dynamic_model.load_state_dict(torch.load( f'{mpc_location}/{str(args.seed)}_DynamicModel.pth', weights_only=True, map_location=args.device ))
        # mpc_dm.mpc_policy_net.load_state_dict(torch.load( 'models_test/7_MPCModel.pth', map_location='cuda',weights_only=True))
        # mpc_dm.dynamic_model.load_state_dict(torch.load( f'models_test/7_DynamicModel.pth', weights_only=True, map_location='cuda' ))

    params = {
        'batch_size': args.decoder_batch,
        'lr': 0.001,  
        'source_env': source_env,
        'target_env': target_env,
        'source_state_space_dim': source_s_dim,
        'source_action_space_dim': source_a_dim,
        'target_state_space_dim': target_s_dim,
        'target_action_space_dim': target_a_dim,
        'agent': agent,
        'agent_Q': agent_Q,
        "mpc_dm": mpc_dm,
        "device": args.device,
        "seed": args.seed,
        "env": args.env,
        "use_flow": args.use_flow
    }
    mpc = MPC(h=args.horizon, load_decoder = True, **params)
    
    print(f'trained_MPC: {args.trained_MPC} horizon: {args.horizon}')

    total = 0
    for i in range(10):
        reward = mpc.evaluate()
        total += reward
        print(f'{i}: {reward}')

    print(f'avg. return of CDPC: {total/10}')