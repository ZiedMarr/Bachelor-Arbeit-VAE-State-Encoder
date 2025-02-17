import os
from configs.save_config import save_config, save_vae_code
from configs import config
from configs.suite_configs import SUITE_CONFIGS

# Assuming base_dir is defined as the directory of the current script
base_dir = os.path.dirname(os.path.abspath(__file__))

from VAE_pretrain.VAE_offline_pretrain import call_pretrain
from VAE_Latent_Space_Eval.VAE_reconstructions_cartpole import call_reconstruction
from VAE_Latent_Space_Eval.VAE_score_Eval import vae_score_call
from VAE_Latent_Space_Eval.Multi_Data_Latent_Colored_Plots import call_latent_colored


def apply_config(new_config):
    # Update the global constants from config.py.
    config.INPUT_STATE_SIZE = new_config.get("INPUT_STATE_SIZE", config.INPUT_STATE_SIZE)
    config.OUTPUT_STATE_SIZE = new_config.get("OUTPUT_STATE_SIZE", config.OUTPUT_STATE_SIZE)
    config.LATENT_DIM = new_config.get("LATENT_DIM", config.LATENT_DIM)
    config.ENCODER_HIDDEN = new_config.get("ENCODER_HIDDEN", config.ENCODER_HIDDEN)
    config.ENCODER_HIDDEN2 = new_config.get("ENCODER_HIDDEN2", config.ENCODER_HIDDEN2)
    config.ENCODER_HIDDEN3 = new_config.get("ENCODER_HIDDEN3", config.ENCODER_HIDDEN3)
    config.DECODER_HIDDEN = new_config.get("DECODER_HIDDEN", config.DECODER_HIDDEN)
    config.DECODER_HIDDEN2 = new_config.get("DECODER_HIDDEN2", config.DECODER_HIDDEN2)
    config.DECODER_HIDDEN3 = new_config.get("DECODER_HIDDEN3", config.DECODER_HIDDEN3)
    config.BETA_KL_DIV = new_config.get("BETA_KL_DIV", config.BETA_KL_DIV)
    config.VAE_Version = new_config.get("VAE_Version", config.VAE_Version)

    # Recalculate derived constants if needed.
    config.INPUT_DIMENSION = config.INPUT_STATE_SIZE * 4
    config.OUTPUT_DIMENSION = config.OUTPUT_STATE_SIZE * 4

def train_suite() :
    vae_name = "vae_rand_100k"
    train_data = os.path.join(base_dir, "..", "Data_Collection", "collected_data", "rand_pol_rand_env","wrapper2", "random_100000_20250211_144748.npz")

    call_pretrain(vae_name=vae_name,data_dir=train_data)
    #call_latent(vae_name=vae_name, show= False)
    call_latent_colored(vae_name=vae_name, show=False)
    call_reconstruction(vae_name)
    vae_score_call(data_path=os.path.join(base_dir, "..", "Data_Collection", "collected_data", "eval","random_10000_20250211_150118.npz"),vae_name=vae_name)
    save_config()
    save_vae_code()


    vae_name = "vae_rand_500k"
    train_data = os.path.join(base_dir, "..", "Data_Collection", "collected_data", "rand_pol_rand_env","wrapper2", "random_500000_20250211_144600.npz")

    call_pretrain(vae_name=vae_name,data_dir=train_data)
    call_latent_colored(vae_name=vae_name, show=False)
    #call_latent(vae_name=vae_name, show= False)
    call_reconstruction(vae_name)
    vae_score_call(data_path=os.path.join(base_dir, "..", "Data_Collection", "collected_data", "eval","random_10000_20250211_150118.npz"),vae_name=vae_name)




    vae_name = "vae_rand_1M"
    train_data = os.path.join(base_dir, "..", "Data_Collection", "collected_data", "rand_pol_rand_env","wrapper2",
                              "random_1000000_20250211_144226.npz")

    call_pretrain(vae_name=vae_name, data_dir=train_data)
    call_latent_colored(vae_name=vae_name, show=False)
    call_reconstruction(vae_name)
    vae_score_call(data_path=os.path.join(base_dir, "..", "Data_Collection", "collected_data", "eval","random_10000_20250211_150118.npz"),vae_name=vae_name)


def main(suite=False) :
    if suite :
        for config_name, new_config in SUITE_CONFIGS.items():
            print(f"\nApplying configuration {config_name}: {new_config}")
            apply_config(new_config)
            train_suite()
    else :
        train_suite()

if __name__ == "__main__" :
    main(True)
