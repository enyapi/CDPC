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
    seed_everything(2)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, nargs='?', default='data/state.npy')
    parser.add_argument("--log_dir", type=str, nargs='?', default='logs/halfcheetah/')
    parser.add_argument("--train_sample_count", type=int, nargs='?', default=25000)
    parser.add_argument("--test_sample_count", type=int, nargs='?', default=25000)
    parser.add_argument("--epochs", type=int, nargs='?', default=500)
    parser.add_argument("--eval_freq", type=int, nargs='?', default=1)
    parser.add_argument("--device", type=str, nargs='?', default='cpu')
    parser.add_argument("--lr", type=float, nargs='?', default=1e-5)
    parser.add_argument("--batch_size", type=int, nargs='?', default=256)
    parser.add_argument("--hidden_size", type=int, nargs='?', default=256)
    parser.add_argument("--transform_count", type=int, nargs='?', default=6) #num. of flow layers
    parser.add_argument("--mollifier_sigma", type=float, nargs='?', default=0.0001)
    parser.add_argument("--gradient_clip_value", type=float, nargs='?', default=0.1)
    parser.add_argument("--take_log_again", action="store_true")
    args = parser.parse_args()

    logger = get_value_logger(args.log_dir)

    # Get the constraint
    conditional_p_count = 0
    dim = 17

    # Define the flow model
    flow = RealNvp.load_module("/media/hdd/penny644/flow0128/flowpg/logs/Halfcheetah/model.pt")
    #flow.load_module("/media/hdd/penny644/flow0128/flowpg/logs/Halfcheetah/model.pt")
    optimizer = th.optim.Adam([p for p in flow.parameters() if p.requires_grad == True], lr=args.lr)

    # Define the mollified uniform distribution
    box_l = th.full((dim, ), -1).double()
    box_h = th.full((dim, ), 1).double()
    uniform_constraint = BoxConstraint(dim, box_l, box_h).to(args.device)
    mollified_uniform_distribution = ConstrainedDistribution(uniform_constraint, args.mollifier_sigma)

    # Load dataset
    data = th.from_numpy(np.load(args.data_file)).double().to(args.device)
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

    os.makedirs(args.log_dir + "/figures", exist_ok=True)
    # sinkhorn_loss_func = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
    np.save(f"{args.log_dir}/figures/test_data.npy", test_data.cpu().numpy())

    start_time = time.time()
    for epoch in range(1):
        losses = []
        # Update flow for each batch

        print(f"Updated batch: {epoch}")
            # Evaluate
        with th.no_grad():
            # Calculate accuracy (z -(g)-> x)
            z_act = th.rand((len(test_data), dim)).double().to(args.device)*2-1 #[-1, 1]
            z = th.concat([z_act, test_data[:, dim:]], dim=1) # Inclue conditional variables
            generated_samples = flow.g(z)[0]
            valid_count = 0
            numpy_generated_samples = generated_samples.cpu().detach().numpy()
            for i in range(args.test_sample_count):
                if((numpy_generated_samples[i] <= bound[0]).all() and (numpy_generated_samples[i] >= bound[1]).all()):
                    valid_count += 1
            accuracy = valid_count/args.test_sample_count
            print('Bound')
            print(bound)
            print('generated samples')
            print(numpy_generated_samples)
            print(f'accuracy: {accuracy}')
              
 

if __name__ == "__main__":
    main()