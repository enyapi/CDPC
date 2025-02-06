import gymnasium as gym
import torch
import numpy as np
import wandb
import random
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sac_v2 import ReplayBuffer, PolicyNetwork

def collect_target_data(args, agent_target, env_target, buffer):
    expert_ratio = args.expert_ratio
    random_ratio = np.round(1.0 - expert_ratio, 2)

    max_episode_steps = env_target.spec.max_episode_steps
    max_episode = args.targetData_ep 
    for episode in range(int(max_episode)):
        score = 0
        state, _ = env_target.reset()
        for _ in range(max_episode_steps):
            if episode < int(max_episode*expert_ratio):
                action = agent_target.get_action(state, deterministic=True)
            else:
                action = np.random.uniform(low=-1, high=1, size=(env_target.action_space.shape[0],))
            next_state, reward, terminated, truncated, _ = env_target.step(action)
            done = truncated or terminated

            buffer.push(state, action, reward, next_state, done)

            score += reward
            if done: break
            state = next_state

        print("episode:{}, Return:{}, buffer_capacity:{}".format(episode, score, len(buffer)))
    env_target.close()
    
    print("Amount of expert data : ", int(max_episode*expert_ratio))
    print("Amount of random data : ", int(max_episode*random_ratio))


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
    parser.add_argument("domain", type=str, nargs='?', default="target")
    parser.add_argument("--seed", type=int, nargs='?', default=2)
    parser.add_argument("--targetData_ep", type=int, nargs='?', default=10000) # 1000/10000
    parser.add_argument("--expert_ratio", type=float, nargs='?', default=0.8) # random_ratio=1-expert_ratio
    parser.add_argument("--ep", type=int, nargs='?', default=500) # the same as cdpc
    parser.add_argument("--device", type=str, nargs='?', default="cuda")
    parser.add_argument("--env", type=str, nargs='?', default="cheetah")
    
    args = parser.parse_args()
    seed_everything(args.seed)
    device = args.device
    
    wandb.init(project="cdpc", name = f'baseline: SAC-Off-TR {str(args.seed)}_{args.env}')
    location = f'./baselines/sac_off_tr/{args.env}/seed_{str(args.seed)}'

    ##### Loading source domain policy #####
    print("##### training source domain policy #####")
    if args.env == "reacher":
        env = gym.make("Reacher-3joints")
    elif args.env == "cheetah":
        env = gym.make("HalfCheetah-3legs")
    

    # Params
    batch_size = 32 #300
    replay_buffer_size = 1e6
    replay_buffer = ReplayBuffer(replay_buffer_size)
    hidden_dim = 512
    action_range = 10.0 if args.env=="reacher" else 1.0
    DETERMINISTIC=False
    AUTO_ENTROPY=True
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]

    wandb.config = {
        "batch_size": batch_size
    }
    wandb.config.update() 

    agent = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range, args.device).to(args.device)

    if not os.path.exists(location): os.makedirs(location)
    if not os.path.exists(f'{location}/{str(args.seed)}_{args.env}_sac_off_tr.pth'):
        ## collect offline data
        collect_target_data(args, agent, env, replay_buffer)

        ## train sac_off_tr
        test_score = agent.evalute()
        wandb.log({"episode": 0, "test/score": test_score})
        for episode in range(1, args.ep+1):
            _ = agent.update(batch_size, reward_scale=10., auto_entropy=AUTO_ENTROPY, target_entropy=-1.*action_dim)

            eval_freq = 1
            if episode % eval_freq == 0: # plot and model saving interval
                test_score = agent.evalute()
                wandb.log({"episode": episode, "test/score": test_score})

        torch.save(agent.policy_net.state_dict(), f'{location}/{str(args.seed)}_{args.env}_sac_off_tr.pth')
