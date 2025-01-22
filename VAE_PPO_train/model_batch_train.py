
import os

from VAE_PPO_train.train import train

batch = "batch2"

#define number of total training time steps :
total_timesteps = 20000

#define vae model path
vae_model_path = "./trained_vae/vae_15000_vae_offline_expert_20250114_165817"

vae_model_name = os.path.basename(vae_model_path)  # Get the last element of the path

#define vae model save path after training
vae_save_folder = f"./trained_vae/{batch}"

log_batch_dir = f"logs/{batch}"

for i in range(10) :
    train(vae_model_path=vae_model_path, vae_save_folder=vae_save_folder,log_batch_dir=log_batch_dir, total_timesteps=total_timesteps)
