import os
import gymnasium as gym
import argparse
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import SAC
import torch
from collections import deque
import random
import gym
import numpy as np
import shelve

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, nargs='?', default=1)
parser.add_argument("--folder", type=str, nargs='?', default='/home/')
parser.add_argument("--env", type=str, nargs='?', default='Hopper-v3')
args = parser.parse_args()

log_folder = args.folder + "source/stable-baselines3-1.7.0/logs/source/seed" + str(args.seed) + "/"

seed_everything(args.seed)
#warnings.filterwarnings("ignore")
vec_env = DummyVecEnv([lambda: gym.make(args.env)])
vec_env.seed(seed=args.seed)
set_random_seed(seed = args.seed)
final_buffer = deque()
state_list = []
action_list = []
next_state_list = []
flatten_list = []
model = SAC.load(log_folder + "best_model")
sample_num = 60000

for i in range(10000):
    obs = vec_env.reset()
    dones = False
    reward = 0
    
    while not dones:
        action, _states = model.predict(obs)
        #flatten = vec_env.sim.get_state().flatten()
        state_list.append(list(obs[0]))
        action_list.append(action)
        obs_next, rewards, dones, info = vec_env.step(action)
        next_state_list.append(obs)
        final_buffer.append((obs[0], action, rewards, obs_next[0], dones))
        obs = obs_next
        reward += rewards
        #vec_env.render("human")
        if(len(state_list) >= sample_num):
            break
    print(f'episode: {i}, reward: {reward}')
    if(len(state_list) >= sample_num):
        break
np.save('data/seed' + str(args.seed) + '/state.npy', np.array(state_list))
file = shelve.open('buffer/buffer_seed' + str(args.seed), writeback=True)
file['buffer'] = final_buffer
file.close()