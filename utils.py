import random
import numpy as np
import os
import torch
import pickle


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False     

def save_buffer(buffer, filename):
    with open(filename, 'wb') as f:
        pickle.dump(buffer, f)

def load_buffer(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)  