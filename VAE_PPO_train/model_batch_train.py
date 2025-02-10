
import os
from configs import eval_config
from configs.save_config import save_eval_config

from VAE_PPO_train.train import train


# Get the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

#name of log directory
batch = "batch_V3.12_kl=0.002_evalconfig2"

#define batch size
batch_size = eval_config.BATCH_SIZE

#define number of total training time steps :
total_timesteps = eval_config.TOTAL_TIMESTEPS

#define vae model path
vae_model_path = os.path.join(script_dir,"..","VAE_pretrain","pretrained_vae","VAE_Version_3.12","4_4","KL-D_0.002","vae_rand_100k")
#VAE_pretrain/pretrained_vae/5_in_2_out/vae_explore_17
vae_model_name = os.path.basename(vae_model_path)  # Get the last element of the path

#define vae model save path after training
vae_save_folder = os.path.join(script_dir,"trained_vae",batch)

log_batch_dir = os.path.join(script_dir,"logs" , batch)
#save current evaluation config :
save_eval_config(log_batch_dir)

seeds = eval_config.EVAL_SEED


for i in range(batch_size) :
    train(vae_model_path=vae_model_path, vae_save_folder=vae_save_folder,log_batch_dir=log_batch_dir, total_timesteps=total_timesteps, seed=seeds[i])
