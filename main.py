import gymnasium as gym
import torch
import random
import numpy as np
import argparse
import os
import warnings
import wandb
from stable_baselines3 import SAC
from sac_v2 import PolicyNetwork
from MPC_DM_model import MPC_DM
from MPC import MPC
from utils import seed_everything, load_buffer, ReplayBuffer_traj, d4rl2Trajs

warnings.filterwarnings('ignore')

def CDPC(mpc, train_set, mpc_location, Is_wandb):
    Return_val = []
    # val state decoder
    total_reward = mpc.evaluate() 
    if Is_wandb:
        wandb.log({"cdpc episode": 0, "valid/reward": total_reward, })

    Return_val.append(total_reward)
    print(f'episode: {0}, validation reward: {total_reward}')
    
    for j in range(1, args.decoder_ep+1):
        # train state decoder
        loss_tran_list, loss_pref_list, loss_rec_list = [], [], []
        for _ in range(1):
            mpc.decoder_net.train()
            loss_tran, loss_pref, loss_rec, pref_acc = mpc.learn(train_set)
            loss_tran_list.append(loss_tran)
            loss_pref_list.append(loss_pref)
            loss_rec_list.append(loss_rec)
        print(f'episode: {j}, transition loss: {np.mean(loss_tran_list)}, pref loss: {np.mean(loss_pref_list)}, rec loss: {np.mean(loss_rec_list)}, pref acc: {pref_acc}')

        # val state decoder
        eval_freq = 1
        if j % eval_freq == 0:
            total_reward = mpc.evaluate() 
            print(f'episode: {j}, avg. validation reward: {total_reward}')

        Return_val.append(total_reward)
        if Is_wandb:
            wandb.log({"cdpc episode": j,
                    "valid/reward": total_reward, 
                    "train/tran loss": np.mean(loss_tran_list),
                    "train/pref loss": np.mean(loss_pref_list),
                    "train/rec loss": np.mean(loss_rec_list),
                    "train/pref acc": pref_acc,
                    })
        # torch.save(mpc.decoder_net.state_dict(), f'{mpc_location}/{str(mpc.seed)}_decoder.pth')
        # if not os.path.exists('./data/'): os.makedirs('./data/')
        # filename = './data/'+str(args.seed)+'_0.8_0.2.npz'
        # np.savez(filename, reward_val = Return_val)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("MPC_pre_ep", type=int, nargs='?', default=10000)
    parser.add_argument("decoder_batch", type=int, nargs='?', default=32)
    parser.add_argument("--seed", type=int, nargs='?', default=2)
    parser.add_argument("--n_traj", type=int, nargs='?', default=10000) # 1000/10000
    parser.add_argument("--expert_ratio", type=float, nargs='?', default=0.2) # random_ratio=1-expert_ratio
    parser.add_argument("--decoder_ep", type=int, nargs='?', default=500) # 500/200
    parser.add_argument("--device", type=str, nargs='?', default="cuda")
    parser.add_argument("--env", type=str, nargs='?', default="cheetah") # cheetah reacher
    parser.add_argument("--wandb", action='store_true', default=False)
    parser.add_argument("--use_flow", action='store_true', default=False)
    
    args = parser.parse_args()
    seed_everything(args.seed)
    if args.wandb:
        wandb.init(project="cdpc", name = f'cdpc {str(args.seed)}_{args.env} {str(args.expert_ratio)}_expert', tags=["cdpc"])
    location = f'./models/{args.env}/seed_{str(args.seed)}/'
    mpc_location = f'{location}/expert_ratio_{args.expert_ratio}/'

    if args.env == "reacher":
        source_env = "Reacher-v4"
        target_env = "Reacher-3joints"
        source_s_dim = 11
        source_a_dim = 2
        target_s_dim = 14
        target_a_dim = 3
        traj_len = 50
    elif args.env == "cheetah":
        source_env = "HalfCheetah-v4"
        target_env = "HalfCheetah-3legs"
        source_s_dim = 18
        source_a_dim = 6
        target_s_dim = 23
        target_a_dim = 9
        traj_len = 1000


    hidden_dim = 512
    action_range = 10.0 if args.env=="reacher" else 1.0

    ##### 1 Load source domain policy #####
    print("##### Loading source domain policy #####")
    #agent = SAC.load(f'./experiments/{args.env}_source_18/models/final_model.zip', device=args.device) # SB3
    agent = PolicyNetwork(source_s_dim, source_a_dim, hidden_dim, action_range, args.device).to(args.device)
    agent.load_state_dict(torch.load( f'{location}/{str(args.seed)}_{args.env}_source.pth', map_location=args.device ))


    ##### 2 Load target domain data #####
    os.makedirs('./train_set/', exist_ok=True)
    data_path = f'./train_set/{str(args.seed)}_{args.env}_{args.expert_ratio}.pkl'
    if os.path.exists(data_path):
        print("##### Loading target domain offline data #####")
        d4rl_data = load_buffer(data_path)
        train_set = ReplayBuffer_traj()
        d4rl2Trajs(d4rl_data, train_set, traj_len=traj_len)


    ##### 3 Load MPC policy and Dynamic Model #####
    mpc_dm = MPC_DM(target_s_dim, target_a_dim, args.device)
    os.makedirs(mpc_location, exist_ok=True)
    if os.path.exists(f'{mpc_location}/{str(args.seed)}_MPCModel.pth'):
        print("##### Loading MPC policy and Dynamic Model #####")
        mpc_dm.mpc_policy_net.load_state_dict(torch.load( f'{mpc_location}/{str(args.seed)}_MPCModel.pth', map_location=args.device ))
        mpc_dm.dynamic_model.load_state_dict(torch.load( f'{mpc_location}/{str(args.seed)}_DynamicModel.pth', map_location=args.device ))

    
    ##### 3.5 Load Flow Model #####
    flow_model = None
    flow_mean = []
    flow_std = []
    if args.use_flow:
        from flowpg.core.flow.real_nvp import RealNvp
        flow_loc = f"./flowpg/flow_models/{args.env}/flow_seed{str(args.seed)}.pt"
        flow_model = RealNvp.load_module(flow_loc).to(args.device)
        flow_model.disable_grad(True)
        flow_mean = torch.from_numpy(np.load(f'./flowpg/data/{args.env}/seed_{str(args.seed)}_mean.npy')).to(args.device).to(torch.float32)
        flow_std = torch.from_numpy(np.load(f'./flowpg/data/{args.env}/seed_{str(args.seed)}_std.npy')).to(args.device).to(torch.float32)


    ##### 4 Training state decoder #####
    print("##### Training state decoder #####")
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
        "mpc_dm": mpc_dm,
        "device": args.device,
        "seed": args.seed,
        "env": args.env,
        "use_flow": args.use_flow,
        "flow_model": flow_model,
        "flow_mean": flow_mean,
        "flow_std": flow_std,
    }
    CDPC(MPC(**params), train_set, mpc_location, args.wandb)
