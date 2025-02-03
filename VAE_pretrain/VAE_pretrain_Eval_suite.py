from VAE_pretrain.VAE_offline_pretrain import call_pretrain
from VAE_Latent_Space_Eval.Multi_Data_Latent_Plot import call_latent
from VAE_Latent_Space_Eval.VAE_reconstructions import call_reconstruction

vae_name = "vae_rand_100k"
train_data = '../Data_Collection/collected data/rand_pol_rand_env/random_100000_20250130_114306.npz'

call_pretrain(vae_name=vae_name,data_dir=train_data)
call_latent(vae_name=vae_name, show= False)
call_reconstruction(vae_name)