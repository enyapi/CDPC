import argparse
from experiments.common.setup_experiment import flush_logs, get_value_logger
from core.constraints import BaseConstraint, BoxConstraint
from core.flow.real_nvp import RealNvp
from core.flow.train_flow import update_flow_batch
from core.flow.constrained_distribution import ConstrainedDistribution
import torch as th
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import time
import random

import gymnasium as gym
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from sac_v2 import PolicyNetwork


def normalization(data):
    mean = th.mean(data, axis=0)
    std = th.std(data, axis=0)
    data  = (data-mean) / (std+1e-8)
    return data, mean, std
    

def data_collect(agent_target, env_target, seed, device):
    state_list = []
    sample_num = 60000

    for i in range(10000):
        obs, _ = env_target.reset(seed=seed)
        done = False
        rewards = 0
        
        while not done:
            action = agent_target.get_action(obs, deterministic=False)
            obs_next, reward, terminated, truncated, _ = env_target.step(action)
            done = truncated or terminated

            state_list.append(list(obs))

            obs = obs_next
            rewards += reward

            if(len(state_list) >= sample_num):
                break
        print(f'episode: {i}, rewards: {rewards}')
        if(len(state_list) >= sample_num):
            break

    np.save(f'./data/{args.env}/seed_{str(args.seed)}_state.npy', np.array(state_list))

    return th.from_numpy(np.array(state_list)).double().to(device)


def main(args, data, dim):
    log_dir = f'./logs/{args.env}/seed_{str(args.seed)}/'
    logger = get_value_logger(log_dir)

    # Get the constraint
    conditional_p_count = 0

    # Define the flow model
    flow = RealNvp(dim, args.transform_count, conditional_param_count=conditional_p_count, hidden_size=args.hidden_size).to(args.device)
    optimizer = th.optim.Adam([p for p in flow.parameters() if p.requires_grad == True], lr=args.lr)

    # Define the mollified uniform distribution
    box_l = th.full((dim, ), -1).double()
    box_h = th.full((dim, ), 1).double()
    uniform_constraint = BoxConstraint(dim, box_l, box_h).to(args.device)
    mollified_uniform_distribution = ConstrainedDistribution(uniform_constraint, args.mollifier_sigma)

    # Load dataset
    if args.train_sample_count + args.test_sample_count > len(data):
        raise ValueError("Not enough samples in the dataset")
    train_data = data[:args.train_sample_count]
    test_data = data[args.train_sample_count: args.test_sample_count+ args.train_sample_count]

    bound = np.zeros((2, dim))
    for i in range(len(train_data[0])):
        bound[0][i] = max(train_data[:, i])
        bound[1][i] = min(train_data[:, i])

    dataset = TensorDataset(train_data)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print("Staring experiment")

    start_time = time.time()
    for epoch in range(args.epochs):
        losses = []
        # Update flow for each batch
        for batch, in data_loader:
            loss = update_flow_batch(flow, mollified_uniform_distribution, batch, optimizer, gradient_clip_value=args.gradient_clip_value, take_log_again=args.take_log_again)
            losses.append(loss)

        print(f"Updated batch: {epoch}")
        if (epoch+1) % args.eval_freq == 0:
            # Evaluate
            with th.no_grad():
                # Calculate accuracy (z -(g)-> x)
                z_act = th.rand((len(test_data), dim)).double().to(args.device)*2-1 #[-1, 1]
                z = th.cat([z_act, test_data[:, dim:]], dim=1) # Inclue conditional variables
                generated_samples = flow.g(z)[0]

                valid_count = 0
                numpy_generated_samples = generated_samples.cpu().detach().numpy()
                for i in range(args.test_sample_count):
                    if((numpy_generated_samples[i] <= bound[0]).all() and (numpy_generated_samples[i] >= bound[1]).all()):
                        valid_count += 1
                accuracy = valid_count/args.test_sample_count
                
                # Calculate recall (x -(f)-> z)
                mapped_z = flow.f(test_data)[0][:, :dim]
                validity_z = th.all(mapped_z >= -1, dim=1) & th.all(mapped_z <= 1, dim=1)
                valid_z_count = validity_z.int().sum().item()
                recall = valid_z_count/len(validity_z)
            

            elapsed_time = time.time() - start_time
            logger.record("train/time_elapsed", elapsed_time)
            logger.record("train/mean_loss", np.mean(losses))
            logger.record("train/accuracy", accuracy)
            logger.record("train/recall", recall)



            print(f"Epoch: {epoch+1}: Mean loss {np.mean(losses):.4f}, Acc: {accuracy*100: .2f}%, Recall: {recall*100: .2f}%")
            logger.record("train/epoch", epoch+1)
            flush_logs()
            logger.dump(epoch)
    os.makedirs(f'./flow_models/{args.env}/', exist_ok=True)
    flow.save_module(f'./flow_models/{args.env}/flow_seed{str(args.seed)}.pt')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_sample_count", type=int, nargs='?', default=50000)
    parser.add_argument("--test_sample_count", type=int, nargs='?', default=10000)
    parser.add_argument("--epochs", type=int, nargs='?', default=5000)
    parser.add_argument("--eval_freq", type=int, nargs='?', default=1)
    parser.add_argument("--lr", type=float, nargs='?', default=1e-5)
    parser.add_argument("--batch_size", type=int, nargs='?', default=5000)
    parser.add_argument("--hidden_size", type=int, nargs='?', default=256)
    parser.add_argument("--transform_count", type=int, nargs='?', default=6) #num. of flow layers
    parser.add_argument("--mollifier_sigma", type=float, nargs='?', default=0.0001)
    parser.add_argument("--gradient_clip_value", type=float, nargs='?', default=0.1)
    parser.add_argument("--take_log_again", action="store_true")
    parser.add_argument("--seed", type=int, nargs='?', default=2)
    parser.add_argument("--device", type=str, nargs='?', default='cuda')
    parser.add_argument("--env", type=str, nargs='?', default='reacher')
    args = parser.parse_args()

    seed_everything(args.seed)
    seed = args.seed
    device = args.device

    # Env
    if args.env == "reacher":
        env = gym.make("Reacher-v4")
        dim = 11
    elif args.env == "cheetah":
        env = gym.make("HalfCheetah-v4", exclude_current_positions_from_observation=False)
        dim = 18

    os.makedirs(f'./data/{args.env}/', exist_ok=True)
    data_file = f'./data/{args.env}/seed_{str(args.seed)}_state.npy'
    if os.path.exists(data_file):
        data = th.from_numpy(np.load(data_file)).double().to(args.device)
        data, mean, std = normalization(data)
    else:
        # parameters
        hidden_dim = 512
        action_range = 10.0 if args.env=="reacher" else 1.0
        action_dim = env.action_space.shape[0]
        state_dim = env.observation_space.shape[0]

        # load expert policy
        trained_agent = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range, device).to(device)
        model_path = f'../models/{args.env}/seed_{str(seed)}/{str(seed)}_{args.env}_source.pth'
        trained_agent.load_state_dict(th.load( model_path, map_location=device ))

        # collect data
        data = data_collect(trained_agent, env, seed, device)
        data, mean, std = normalization(data)

    np.save(f'./data/{args.env}/seed_{str(args.seed)}_mean.npy', mean.cpu().detach().numpy())
    np.save(f'./data/{args.env}/seed_{str(args.seed)}_std.npy', std.cpu().detach().numpy())

    # Train flow forwad using generated samples
    main(args, data, dim)