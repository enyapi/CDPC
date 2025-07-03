import gymnasium as gym
import torch
import argparse
import os
import wandb
from MPC_DM_model import MPCPolicyTrainer, DynamicsModelTrainer
from utils import seed_everything, load_buffer

parser = argparse.ArgumentParser()
parser.add_argument("MPC_pre_ep", type=int, nargs='?', default=1000)
parser.add_argument("--seed", type=int, nargs='?', default=7)
parser.add_argument("--expert_ratio", type=float, nargs='?', default=0.2) # random_ratio=1-expert_ratio
parser.add_argument("--device", type=str, nargs='?', default="cuda")
parser.add_argument("--env", type=str, nargs='?', default="cheetah") # cheetah reacher
parser.add_argument("--wandb", action='store_true', default=False)

args = parser.parse_args()
seed_everything(args.seed)

if args.env == "reacher":
    target_env = "Reacher-3joints"
    target_s_dim = 14
    target_a_dim = 3
elif args.env == "cheetah":
    target_env = "HalfCheetah-3legs"
    target_s_dim = 23
    target_a_dim = 9

if args.wandb:
    wandb.init(project="cdpc", name = f'BC training', tags=["policy&DM"])

location = f'./models/{args.env}/seed_{str(args.seed)}/'
mpc_location = f'{location}/expert_ratio_{args.expert_ratio}/'

target_buffer_path = f'./train_set/{str(args.seed)}_{args.env}_{args.expert_ratio}_target_buffer.pkl' ## medium-expert
target_test_buffer_path = f'./train_set/7_{args.env}_{args.expert_ratio}_target_buffer.pkl'
buffer = load_buffer(target_buffer_path)
test_buffer = load_buffer(target_test_buffer_path)

mpc_policy = MPCPolicyTrainer(target_s_dim, target_a_dim, args.device)
dm = DynamicsModelTrainer(target_s_dim, target_a_dim, args.device)

if not os.path.exists(f'{mpc_location}/{str(args.seed)}_MPCModel.pth'):
    print("##### Training MPC policy and Dynamic Model #####")
    os.makedirs(mpc_location, exist_ok=True)
    batch_size = 128

    for i in range(args.MPC_pre_ep):
        loss_mpc = mpc_policy.update(batch_size, buffer)
        loss_dm = dm.update(batch_size, buffer)

        if args.wandb:
            wandb.log({"mpc_dm episode": i, "train/loss_mpc": loss_mpc, "train/loss_dm": loss_dm,})

        if i % 10 == 0:
            env = gym.make(target_env)
            reward = mpc_policy.evaluate_policy(env, args.seed)
            val_loss_dm = dm.evaluate_dm(batch_size, test_buffer)
            print(f"episode: {i}/{args.MPC_pre_ep}, reward: {reward}, val_loss_dm: {val_loss_dm.item()}")
            if args.wandb:
                wandb.log({"mpc_dm episode": i, "test/BC_score": reward, "test/val_loss_dm": val_loss_dm.item()})

    torch.save(mpc_policy.policy_net.state_dict(), f'{mpc_location}/{str(args.seed)}_MPCModel.pth')
    torch.save(dm.model.state_dict(), f'{mpc_location}/{str(args.seed)}_DynamicModel.pth')
    print(f'{mpc_location}/{str(args.seed)}_DynamicModel.pth saved')