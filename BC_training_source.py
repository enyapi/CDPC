import gymnasium as gym
import torch
import argparse
import os
import wandb
from sac_v2 import PolicyNetwork
from MPC_DM_model import MPC_DM
#from stable_baselines3 import SAC
from utils import seed_everything, load_buffer

parser = argparse.ArgumentParser()
parser.add_argument("MPC_pre_ep", type=int, nargs='?', default=2000)
parser.add_argument("--seed", type=int, nargs='?', default=7)
parser.add_argument("--expert_ratio", type=float, nargs='?', default=0.2) # random_ratio=1-expert_ratio
parser.add_argument("--device", type=str, nargs='?', default="cuda")
parser.add_argument("--env", type=str, nargs='?', default="cheetah") # cheetah reacher
parser.add_argument("--wandb", action='store_true', default=True)

args = parser.parse_args()
seed_everything(args.seed)

if args.env == "reacher":
    source_env = "Reacher-v4"
    source_s_dim = 11
    source_a_dim = 2
elif args.env == "cheetah":
    source_env = "HalfCheetah-v4"
    source_s_dim = 18
    source_a_dim = 6

if args.wandb:
    wandb.init(project="cdpc", name = f'BC training source', tags=["policy&DM"])
location = f'./models/{args.env}/seed_{str(args.seed)}/'
mpc_location = f'{location}/expert_ratio_{args.expert_ratio}/'
hidden_dim = 256
action_range = 10.0 if args.env=="reacher" else 1.0

# agent_target = SAC.load('models/cheetah/seed_7/HalfCheetah-3legs_SAC_3_128_200000_2.zip', device=args.device) # SB3
agent = PolicyNetwork(source_s_dim, source_a_dim, hidden_dim, action_range, args.device).to(args.device)
agent.load_state_dict(torch.load( f'{location}{str(args.seed)}_{args.env}_source.pth', weights_only=True, map_location=args.device ))
agent_medium = PolicyNetwork(source_s_dim, source_a_dim, hidden_dim, action_range, args.device).to(args.device)
agent_medium.load_state_dict(torch.load( f'{location}{str(args.seed)}_{args.env}_source_medium.pth', weights_only=True, map_location=args.device ))

source_buffer_path = f'./train_set/{str(args.seed)}_{args.env}_{args.expert_ratio}_source_buffer_expert.pkl'
buffer = load_buffer(source_buffer_path)

mpc_dm = MPC_DM(source_s_dim, source_a_dim, args.device)

if not os.path.exists(f'{mpc_location}/{str(args.seed)}_DynamicModel_source.pth'):
    print("##### Source Domain Dynamic Model #####")
    os.makedirs(mpc_location, exist_ok=True)
    batch_size=512

    for i in range(args.MPC_pre_ep):
        loss_mpc, loss_dm = mpc_dm.update(batch_size, buffer)

        if args.wandb:
            wandb.log({"mpc_dm episode": i, "train/loss_mpc": loss_mpc, "train/loss_dm": loss_dm,})

        if i % 10==0:
            env = gym.make(source_env)
            reward = mpc_dm.evaluate_policy(env, args.seed)
            print(f"episode: {i}/{args.MPC_pre_ep}, reward: {reward}")
            if args.wandb:
                wandb.log({"mpc_dm episode": i, "test/BC_score": reward,})

    #torch.save(mpc_dm.mpc_policy_net.state_dict(), f'{mpc_location}/{str(args.seed)}_MPCModel_source.pth')
    torch.save(mpc_dm.dynamic_model.state_dict(), f'{mpc_location}/{str(args.seed)}_DynamicModel_source.pth')
    print(f'{mpc_location}/{str(args.seed)}_DynamicModel_source.pth saved')