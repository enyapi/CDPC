import argparse
from experiments.common.setup_experiment import setup_experiment, flush_logs, get_value_logger
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

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

def main():
    """
    Train flow forwad using generated samples from a file.
    """
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, nargs='?', default='./logs/Hopper_action/')
    parser.add_argument("--train_sample_count", type=int, nargs='?', default=25000)
    parser.add_argument("--test_sample_count", type=int, nargs='?', default=25000)
    parser.add_argument("--epochs", type=int, nargs='?', default=500)
    parser.add_argument("--eval_freq", type=int, nargs='?', default=1)
    parser.add_argument("--device", type=str, nargs='?', default='cuda:2')
    parser.add_argument("--lr", type=float, nargs='?', default=1e-5)
    parser.add_argument("--batch_size", type=int, nargs='?', default=256)
    parser.add_argument("--hidden_size", type=int, nargs='?', default=256)
    parser.add_argument("--transform_count", type=int, nargs='?', default=6) #num. of flow layers
    parser.add_argument("--mollifier_sigma", type=float, nargs='?', default=0.0001)
    parser.add_argument("--gradient_clip_value", type=float, nargs='?', default=0.1)
    parser.add_argument("--dim", type=int, nargs='?', default=3)
    parser.add_argument("--take_log_again", action="store_true")
    parser.add_argument("--seed", type=int, nargs='?', default=1)
    parser.add_argument("--folder", type=str, nargs='?', default='/home/')
    args = parser.parse_args()
    log_dir = './logs/Hopper_action/seed' + str(args.seed) + "/"
    logger = get_value_logger(log_dir)
    seed_everything(args.seed)
    # Get the constraint
    conditional_p_count = 0
    dim = args.dim
    data_file = args.folder + "source/stable-baselines3-1.7.0/data_action/seed" + str(args.seed) + "/action.npy"
    # Define the flow model
    flow = RealNvp(dim, args.transform_count, conditional_param_count=conditional_p_count, hidden_size=args.hidden_size).to(args.device)
    optimizer = th.optim.Adam([p for p in flow.parameters() if p.requires_grad == True], lr=args.lr)

    # Define the mollified uniform distribution
    box_l = th.full((dim, ), -1).double()
    box_h = th.full((dim, ), 1).double()
    uniform_constraint = BoxConstraint(dim, box_l, box_h).to(args.device)
    mollified_uniform_distribution = ConstrainedDistribution(uniform_constraint, args.mollifier_sigma)

    # Load dataset
    data = th.from_numpy(np.load(data_file)).double().to(args.device)
    if args.train_sample_count + args.test_sample_count > len(data):
        raise ValueError("Not enough samples in the dataset")
    train_data = data[:args.train_sample_count]
    bound = np.zeros((2, dim))
    for i in range(len(train_data[0])):
        bound[0][i] = max(train_data[:, i])
        bound[1][i] = min(train_data[:, i])
    test_data = data[args.train_sample_count: args.test_sample_count+ args.train_sample_count]

    dataset = TensorDataset(train_data)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print("Staring experiment")

    os.makedirs(log_dir + "/figures", exist_ok=True)
    # sinkhorn_loss_func = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
    np.save(f"{log_dir}/figures/test_data.npy", test_data.cpu().numpy())

    start_time = time.time()
    for epoch in range(args.epochs):
        losses = []
        # Update flow for each batch
        for batch, in data_loader:
            loss = update_flow_batch(flow, mollified_uniform_distribution, batch, optimizer, gradient_clip_value=args.gradient_clip_value, take_log_again=args.take_log_again)
            losses.append(loss)

        print(f"Updated batch: {epoch}")
        if (epoch+1)%args.eval_freq == 0:
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
            flow.save_module(f"{log_dir}/model.pt")
            logger.record("train/epoch", epoch+1)
            flush_logs()
            logger.dump(epoch)
    flow.save_module(args.folder + "source/stable-baselines3-1.7.0/model_action/flow_seed" + str(args.seed) + ".pt")

if __name__ == "__main__":
    main()