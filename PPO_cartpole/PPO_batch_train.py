import os
from PPO_cartpole.train_ppo_cartpole import train_ppo_cartpole
from configs import eval_config
from configs.save_config import save_eval_config

# Get the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

#name of log directory
batch = "batch_20000_timesteps_rand_env_evalconfig2"

#define batch size
batch_size = eval_config.BATCH_SIZE

#define number of total training time steps :
total_timesteps = eval_config.TOTAL_TIMESTEPS

log_batch_dir = os.path.join(script_dir,"logs", "explore_rand_env" , batch)
#save eval configs :
save_eval_config(log_batch_dir)

seeds = eval_config.EVAL_SEED

for i in range(batch_size):
    train_ppo_cartpole(log_batch_dir=log_batch_dir, total_timesteps=total_timesteps, seed=seeds[i])