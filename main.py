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
wandb.init(project="cdpc", name = 'reacher v2 cdpc: mpc reacher_source_R')


def collect_target_data(args, agent_target, target_env):
    buffer_maxlen = 1000000
    buffer = ReplayBuffer(buffer_maxlen, args.device)
    train_set = ReplayBuffer_traj()

    expert_ratio = args.expert_ratio
    random_ratio = args.random_ratio

    env_target = gym.make(target_env)
    max_episode_steps = env_target.spec.max_episode_steps
    max_episode = args.targetData_ep 
    for episode in range(int(max_episode)):
        score = 0
        state, _ = env_target.reset()
        state_list = []
        next_state_list = []
        for _ in range(max_episode_steps):
            if episode < int(max_episode*expert_ratio):
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
    
    print("Amount of expert data : ", int(max_episode*expert_ratio))
    print("Amount of random data : ", int(max_episode*random_ratio))
    return train_set, buffer


def CDPC(mpc, train_set):
    Return_val = []
    # val state decoder
    total_reward = mpc.evaluate() 
    wandb.log({"validation_reward": total_reward, })

    Return_val.append(total_reward)
    print(f'episode: {0}, validation reward: {total_reward}')
    
    for j in range(1, args.decoder_ep+1):
        # train state decoder
        loss_tran_list, loss_pref_list, loss_rec_list = [], [], []
        for _ in range(1):
            mpc.decoder_net.train()
            loss_tran, loss_pref, loss_rec = mpc.learn(train_set)
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
        wandb.log({"validation_reward": total_reward, 
                  "tran loss": np.mean(loss_tran_list),
                  "pref loss": np.mean(loss_pref_list),
                  "rec loss": np.mean(loss_rec_list),
                  })
        # if not os.path.exists('./data/'): os.makedirs('./data/')
        # filename = './data/'+str(args.seed)+'_0.8_0.2.npz'
        # np.savez(filename, reward_val = Return_val)
    


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False        


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int, nargs='?', default=2)
    parser.add_argument("targetData_ep", type=int, nargs='?', default=10000) # 10000
    parser.add_argument("MPC_pre_ep", type=int, nargs='?', default=10000)
    parser.add_argument("expert_ratio", type=float, nargs='?', default=0.8)
    parser.add_argument("random_ratio", type=float, nargs='?', default=0.2)
    parser.add_argument("decoder_batch", type=int, nargs='?', default=32)
    parser.add_argument("decoder_ep", type=int, nargs='?', default=500)
    parser.add_argument("--device", type=str, nargs='?', default="cuda")
    parser.add_argument("--env", type=str, nargs='?', default="cheetah") # cheetah reacher
    
    args = parser.parse_args()
    seed_everything(args.seed)

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
    agent.load_state_dict(torch.load(f'./v2_models/{args.env}/{args.env}_source.pth'))


    ##### 2 Loading target domain expert policy #####
    print("##### Loading target domain expert policy #####")
    #agent_target = SAC.load(f'./experiments/{args.env}_target/models/final_model.zip', device=args.device) # SB3
    agent_target = PolicyNetwork(target_s_dim, target_a_dim, hidden_dim, action_range, args.device).to(args.device)
    agent_target.load_state_dict(torch.load(f'./v2_models/{args.env}/{args.env}_target.pth'))


    ##### 3 Collecting target domain data #####
    print("##### Collecting target domain data #####")
    train_set, buffer = collect_target_data(args, agent_target, target_env)


    ##### 4 Train or Loading MPC policy and Dynamic Model #####
    mpc_dm = MPC_DM(target_s_dim, target_a_dim, args.device)
    if not os.path.exists(f'./v2_models/{args.env}'): os.makedirs(f'./v2_models/{args.env}')
    if os.path.exists(f'./v2_models/{args.env}/{str(args.seed)}_MPCModel.pth'):
        print("##### Loading MPC policy and Dynamic Model #####")
        mpc_dm.mpc_policy_net.load_state_dict(torch.load( f'./v2_models/{args.env}/{str(args.seed)}_MPCModel.pth', map_location=args.device ))
        mpc_dm.dynamic_model.load_state_dict(torch.load( f'./v2_models/{args.env}/{str(args.seed)}_DynamicModel.pth', map_location=args.device ))
    else:
        print("##### Training MPC policy and Dynamic Model #####")
        batch_size = 128
        for i in range(args.MPC_pre_ep):
            mpc_dm.update(batch_size, buffer)
            torch.save(mpc_dm.mpc_policy_net.state_dict(), f'./v2_models/{args.env}/{str(args.seed)}_MPCModel.pth')
            torch.save(mpc_dm.dynamic_model.state_dict(), f'./v2_models/{args.env}/{str(args.seed)}_DynamicModel.pth')


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
    CDPC(MPC(**params), train_set)