
import os

from VAE_PPO_train.train import train
from config import EVAL_SEED

# Get the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

#name of log directory
batch = "batch_explore_53_kl0,1_rand_env_50k"

#define batch size
batch_size = 5

#define number of total training time steps :
total_timesteps = 50000

#define vae model path
vae_model_path = os.path.join(script_dir,"..","VAE_pretrain","pretrained_vae","5_3","rand_0,1_100k","vae_rand_1")
#VAE_pretrain/pretrained_vae/5_in_2_out/vae_explore_17
vae_model_name = os.path.basename(vae_model_path)  # Get the last element of the path

#define vae model save path after training
vae_save_folder = os.path.join(script_dir,"trained_vae",batch)

log_batch_dir = os.path.join(script_dir,"logs" , batch)

seeds = EVAL_SEED


for i in range(batch_size) :
    train(vae_model_path=vae_model_path, vae_save_folder=vae_save_folder,log_batch_dir=log_batch_dir, total_timesteps=total_timesteps, seed=seeds[i])
