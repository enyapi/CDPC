import torch
import numpy as np
import argparse
import warnings
import wandb
from stable_baselines3 import SAC
from sac_v2 import PolicyNetwork
from MPC_DM_model import MPC_DM, Dynamic_Model
from MPC_v2 import MPC
from utils import seed_everything, load_buffer

warnings.filterwarnings('ignore')

def CDPC(mpc, train_set, target_buffer, source_buffer, mpc_location, Is_wandb):
    # val state decoder
    total_reward = mpc.evaluate() 
    if Is_wandb:
        wandb.log({"cdpc episode": 0, "test/reward": total_reward, })
    print(f'episode: {0}, validation reward: {total_reward}')
    
    for j in range(1, args.num_ep+1):
        # train state decoder
        loss_dcc_list, adv_state_loss_D_list, adv_action_loss_D_list, adv_state_loss_G_list, adv_action_loss_G_list, loss_pref_list = [], [], [], [], [], []
        for _ in range(1):
            dcc_loss, adv_state_loss_for_D, adv_action_loss_for_D, adv_state_loss_for_G, adv_action_loss_for_G, loss_pref, pref_acc = mpc.learn(train_set, target_buffer, source_buffer)
            loss_dcc_list.append(dcc_loss)
            adv_state_loss_D_list.append(adv_state_loss_for_D)
            adv_action_loss_D_list.append(adv_action_loss_for_D)
            adv_state_loss_G_list.append(adv_state_loss_for_G)
            adv_action_loss_G_list.append(adv_action_loss_for_G)
            loss_pref_list.append(loss_pref)

        print(f'episode: {j}, dcc loss: {np.mean(loss_dcc_list)}, pref loss: {np.mean(loss_pref_list)}, pref acc: {pref_acc}')
        if Is_wandb:
            wandb.log({"cdpc episode": j,
                    "train/dcc loss": np.mean(loss_dcc_list),
                    "train/pref loss": np.mean(loss_pref_list),
                    "train/state D loss": np.mean(adv_state_loss_D_list),
                    "train/state G loss": np.mean(adv_state_loss_G_list),
                    "train/action D loss": np.mean(adv_action_loss_D_list),
                    "train/action G loss": np.mean(adv_action_loss_G_list),
                    "train/pref acc": pref_acc,
                    })

        # val state decoder
        eval_freq = 10
        if j % eval_freq == 0:
            total_reward = mpc.evaluate() 
            print(f'episode: {j}, avg. validation reward: {total_reward}')
            if Is_wandb:
                wandb.log({"cdpc episode": j, "test/reward": total_reward,})
            
        torch.save(mpc.state_projector.state_dict(), f'{mpc_location}/{str(mpc.seed)}_state_projector_{args.num_ep}.pth')
        torch.save(mpc.action_projector.state_dict(), f'{mpc_location}/{str(mpc.seed)}_action_projector_{args.num_ep}.pth')     


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("MPC_pre_ep", type=int, nargs='?', default=10000)
    parser.add_argument("--decoder_batch", type=int, nargs='?', default=32)
    parser.add_argument("--seed", type=int, nargs='?', default=2)
    parser.add_argument("--n_traj", type=int, nargs='?', default=10000) # 1000/10000
    parser.add_argument("--expert_ratio", type=float, nargs='?', default=0.2) # random_ratio=1-expert_ratio
    parser.add_argument("--num_ep", type=int, nargs='?', default=500) # 500/200
    parser.add_argument("--device", type=str, nargs='?', default="cuda")
    parser.add_argument("--env", type=str, nargs='?', default="cheetah") # cheetah reacher
    parser.add_argument("--wandb", action='store_true', default=False)
    
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
    agent_source = PolicyNetwork(source_s_dim, source_a_dim, hidden_dim, action_range, args.device).to(args.device)
    agent_source.load_state_dict(torch.load( f'{location}/{str(args.seed)}_{args.env}_source.pth', map_location=args.device ))


    ##### 2 Loading target domain data #####
    print("##### Loading target domain data #####")
    data_path = f'./train_set/{str(args.seed)}_{args.env}_{args.expert_ratio}.pkl'
    target_buffer_path = f'./train_set/{str(args.seed)}_{args.env}_{args.expert_ratio}_target_buffer.pkl'
    target_buffer_expert_path = f'./train_set/{str(args.seed)}_{args.env}_{args.expert_ratio}_target_buffer_expert.pkl'
    source_buffer_path = f'./train_set/{str(args.seed)}_{args.env}_{args.expert_ratio}_source_buffer.pkl'
    source_buffer_expert_path = f'./train_set/{str(args.seed)}_{args.env}_{args.expert_ratio}_source_buffer_expert.pkl'

    train_set = load_buffer(data_path)
    target_buffer = load_buffer(target_buffer_path)
    target_buffer_expert_only = load_buffer(target_buffer_expert_path)
    source_buffer = load_buffer(source_buffer_path)
    source_buffer_expert_only = load_buffer(source_buffer_expert_path)

    print(train_set.buffer_len())


    ##### 3 Loading MPC policy and Dynamic Model #####
    print("##### Loading MPC policy and Dynamic Model #####")
    mpc_dm = MPC_DM(target_s_dim, target_a_dim, args.device)
    source_dynamics_model = Dynamic_Model(source_s_dim + source_a_dim, source_s_dim).to(args.device)
    mpc_dm.mpc_policy_net.load_state_dict(torch.load( f'{mpc_location}/{str(args.seed)}_MPCModel.pth', map_location=args.device ))
    mpc_dm.dynamic_model.load_state_dict(torch.load( f'{mpc_location}/{str(args.seed)}_DynamicModel.pth', map_location=args.device ))
    source_dynamics_model.load_state_dict(torch.load( f'{mpc_location}/{str(args.seed)}_DynamicModel_source.pth', map_location=args.device ))


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
        'agent': agent_source,
        "mpc_dm": mpc_dm,
        "source_dynamics_model": source_dynamics_model,
        "device": args.device,
        "seed": args.seed,
        "env": args.env,
    }
    CDPC(MPC(**params), train_set, target_buffer, source_buffer, mpc_location, args.wandb)
