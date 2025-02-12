import gymnasium as gym
import torch
import random
import numpy as np
import argparse
import os
import warnings
import wandb
#from stable_baselines3 import SAC
from sac_v2 import PolicyNetwork
from MPC_DM_model import ReplayBuffer, MPC_DM
from MPC import ReplayBuffer_traj, MPC


warnings.filterwarnings('ignore')


def collect_target_data(agent_target, target_env, n_traj, expert_ratio, device, seed):
    buffer_maxlen = 1000000
    buffer = ReplayBuffer(buffer_maxlen, device)
    train_set = ReplayBuffer_traj()

    env_target = gym.make(target_env)
    max_episode_steps = env_target.spec.max_episode_steps 
    for episode in range(int(n_traj)):
        score = 0
        state, _ = env_target.reset(seed=seed*episode) ############################################
        state_list = []
        next_state_list = []
        for _ in range(max_episode_steps):
            if episode < int(n_traj*expert_ratio):
                #action, _ = agent_target.predict(state, deterministic=True) # SB3
                action = agent_target.get_action(state, deterministic=True)
            else:
                action = np.random.uniform(low=-1, high=1, size=(env_target.action_space.shape[0],))
            next_state, reward, terminated, truncated, _ = env_target.step(action)
            done = truncated or terminated

            done_mask = 0.0 if done else 1.0
            buffer.push((state, action, reward, next_state, done_mask))

            state_list.append(state)
            next_state_list.append(next_state)
            state = next_state
            score += reward
            
            if done: break

        train_set.push(score, state_list, next_state_list)
        print("episode:{}, Return:{}, buffer_capacity:{}".format(episode, score, buffer.buffer_len()))
    env_target.close()
    
    print(f"Collected {n_traj} trajectories.")
    print(f"Collected {n_traj*max_episode_steps} transitions.")
    return train_set, buffer


def CDPC(mpc, train_set, mpc_location):
    Return_val = []
    # val state decoder
    total_reward = mpc.evaluate() 
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
        print(f'episode: {j}, transition loss: {np.mean(loss_tran_list)}, pref loss: {np.mean(loss_pref_list)}, rec loss: {np.mean(loss_rec_list)}')

        # val state decoder
        eval_freq = 1
        if j % eval_freq == 0:
            total_reward = mpc.evaluate() 
            print(f'episode: {j}, avg. validation reward: {total_reward}')

        Return_val.append(total_reward)
        wandb.log({"cdpc episode": j,
                "valid/reward": total_reward, 
                "train/tran loss": np.mean(loss_tran_list),
                "train/pref loss": np.mean(loss_pref_list),
                "train/rec loss": np.mean(loss_rec_list),
                "train/pref acc": pref_acc,
                })
        torch.save(mpc.decoder_net.state_dict(), f'{mpc_location}/{str(mpc.seed)}_decoder.pth')
        # if not os.path.exists('./data/'): os.makedirs('./data/')
        # filename = './data/'+str(args.seed)+'_0.8_0.2.npz'
        # np.savez(filename, reward_val = Return_val)
    


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
    parser.add_argument("MPC_pre_ep", type=int, nargs='?', default=10000)
    parser.add_argument("decoder_batch", type=int, nargs='?', default=32)
    parser.add_argument("--seed", type=int, nargs='?', default=2)
    parser.add_argument("--n_traj", type=int, nargs='?', default=10000) # 1000/10000
    parser.add_argument("--expert_ratio", type=float, nargs='?', default=0.2) # random_ratio=1-expert_ratio
    parser.add_argument("--decoder_ep", type=int, nargs='?', default=500) # 500/200
    parser.add_argument("--device", type=str, nargs='?', default="cuda")
    parser.add_argument("--env", type=str, nargs='?', default="cheetah") # cheetah reacher
    
    args = parser.parse_args()
    seed_everything(args.seed)

    wandb.init(project="cdpc", name = f'cdpc {str(args.seed)}_{args.env} {str(args.expert_ratio)}_expert')
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


    hidden_dim = 512
    action_range = 10.0 if args.env=="reacher" else 1.0

    ##### 1 Loading source domain policy #####
    print("##### Loading source domain policy #####")
    #agent = SAC.load(f'./experiments/{args.env}_source_18/models/final_model.zip', device=args.device) # SB3
    agent = PolicyNetwork(source_s_dim, source_a_dim, hidden_dim, action_range, args.device).to(args.device)
    agent.load_state_dict(torch.load( f'{location}/{str(args.seed)}_{args.env}_source.pth', map_location=args.device ))


    ##### 2 Loading target domain expert policy #####
    print("##### Loading target domain expert policy #####")
    #agent_target = SAC.load(f'./experiments/{args.env}_target/models/final_model.zip', device=args.device) # SB3
    agent_target = PolicyNetwork(target_s_dim, target_a_dim, hidden_dim, action_range, args.device).to(args.device)
    agent_target.load_state_dict(torch.load( f'{location}/{str(args.seed)}_{args.env}_target.pth', map_location=args.device ))


    ##### 3 Collecting target domain data #####
    print("##### Collecting target domain data #####")
    train_set, buffer = collect_target_data(agent_target, target_env, args.n_traj, args.expert_ratio, args.device, args.seed)
    import pickle
    # def save_buffer(buffer, filename):
    #     with open(filename, 'wb') as f:
    #         pickle.dump(buffer, f)

    # def load_buffer(filename):
    #     with open(filename, 'rb') as f:
    #         return pickle.load(f)

    # #save_buffer(train_set, '2_cheetah_0.8.pkl')
    # train_set = load_buffer('2_cheetah_0.8.pkl')
    # print(train_set.buffer_len())


    ##### 4 Train or Loading MPC policy and Dynamic Model #####
    mpc_dm = MPC_DM(target_s_dim, target_a_dim, args.device)
    if not os.path.exists(mpc_location): os.makedirs(mpc_location)
    if os.path.exists(f'{mpc_location}/{str(args.seed)}_MPCModel.pth'):
        print("##### Loading MPC policy and Dynamic Model #####")
        mpc_dm.mpc_policy_net.load_state_dict(torch.load( f'{mpc_location}/{str(args.seed)}_MPCModel.pth', map_location=args.device ))
        mpc_dm.dynamic_model.load_state_dict(torch.load( f'{mpc_location}/{str(args.seed)}_DynamicModel.pth', map_location=args.device ))
    else:
        print("##### Training MPC policy and Dynamic Model #####")
        batch_size = 128
        for i in range(args.MPC_pre_ep):
            loss_mpc, loss_dm = mpc_dm.update(batch_size, buffer)
            wandb.log({"mpc_dm episode": i, "train/loss_mpc": loss_mpc, "train/loss_dm": loss_dm,})
        torch.save(mpc_dm.mpc_policy_net.state_dict(), f'{mpc_location}/{str(args.seed)}_MPCModel.pth')
        torch.save(mpc_dm.dynamic_model.state_dict(), f'{mpc_location}/{str(args.seed)}_DynamicModel.pth')


    ##### 5 Training state decoder #####
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
    }
    CDPC(MPC(**params), train_set, mpc_location)