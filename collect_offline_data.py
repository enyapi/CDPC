import os
from stable_baselines3 import SAC
from sac_v2 import PolicyNetwork
import argparse
import torch
from utils import seed_everything, save_buffer, collect_d4rl_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, nargs='?', default=2)
    parser.add_argument("--n_traj", type=int, nargs='?', default=10000) # 1000/10000
    parser.add_argument("--expert_ratio", type=float, nargs='?', default=0.2) # random_ratio=1-expert_ratio
    parser.add_argument("--device", type=str, nargs='?', default="cuda")
    parser.add_argument("--env", type=str, nargs='?', default="cheetah") # cheetah reacher
    args = parser.parse_args()

    seed_everything(args.seed)

    data_path = f'./train_set/{str(args.seed)}_{args.env}_{args.expert_ratio}.pkl'
    assert not os.path.exists(data_path), 'dataset already exists'


    if args.env == "reacher":
        target_env = "Reacher-3joints"
        target_s_dim = 14
        target_a_dim = 3
    elif args.env == "cheetah":
        target_env = "HalfCheetah-3legs"
        target_s_dim = 23
        target_a_dim = 9

    seed = args.seed
    device = args.device
    hidden_dim = 512
    action_range = 10.0 if args.env=="reacher" else 1.0
    location = f'./models/{args.env}/seed_{str(seed)}/'
    
    # Load expert policy
    print("##### Loading target domain expert policy #####")
    #agent_target = SAC.load(f'./HalfCheetah-3legs_SAC_{str(args.seed)}_128_200000_2.zip', device=args.device) # SB3
    #agent_target = SAC.load(f'./experiments/{args.env}_target/models/final_model.zip', device=args.device) # SB3
    agent_target = PolicyNetwork(target_s_dim, target_a_dim, hidden_dim, action_range, device).to(device)
    agent_target.load_state_dict(torch.load( f'{location}/{str(seed)}_{args.env}_target.pth', map_location=device ))

    # Collect data
    data = collect_d4rl_data(agent_target, target_env, args.n_traj, args.expert_ratio, device, seed)
    
    # Save data
    os.makedirs('./train_set/', exist_ok=True)
    save_buffer(data, data_path)
    print(f"Saved {len(data['observations'])} transitions to {data_path}")