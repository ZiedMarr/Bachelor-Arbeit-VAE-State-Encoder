import os
from PPO_cartpole.train_ppo_cartpole import train_ppo_cartpole
from config import EVAL_SEED


# Get the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

#name of log directory
batch = "batch_20000_timesteps_rand_env"

#define batch size
batch_size = 5

#define number of total training time steps :
total_timesteps = 50000

log_batch_dir = os.path.join(script_dir,"logs", "explore_rand_env" , batch)

seeds = EVAL_SEED

for i in range(batch_size):
    train_ppo_cartpole(log_batch_dir=log_batch_dir, total_timesteps=total_timesteps, seed=seeds[i])