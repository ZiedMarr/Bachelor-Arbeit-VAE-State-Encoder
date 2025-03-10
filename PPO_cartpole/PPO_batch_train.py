import os
from PPO_cartpole.train_ppo_cartpole import train_ppo_cartpole


# Get the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

#name of log directory
batch = "batch_20000_timesteps"

#define batch size
batch_size = 2

#define number of total training time steps :
total_timesteps = 20000

log_batch_dir = os.path.join(script_dir,"logs", "explore" , batch)

seeds = [19 , 323]

for i in range(batch_size):
    train_ppo_cartpole(log_batch_dir=log_batch_dir, total_timesteps=total_timesteps, seed=seeds[i])