import os
import torch
import torch.multiprocessing as mp

import configs.config
from configs.save_config import save_config, save_vae_code
from configs import config as config_module
from configs.suite_configs import SUITE_CONFIGS
import psutil
from typing import Dict, Any
from datetime import datetime

from VAE_pretrain.VAE_offline_pretrain import call_pretrain
from VAE_Latent_Space_Eval.VAE_reconstructions import call_reconstruction
from VAE_Latent_Space_Eval.VAE_score_Eval import vae_score_call
from VAE_Latent_Space_Eval.Multi_Data_Latent_Colored_Plots import call_latent_colored


class Config:
    """Configuration class to hold all parameters"""

    def __init__(self, config_dict):
        self.INPUT_STATE_SIZE = config_dict.get("INPUT_STATE_SIZE", 5)
        self.OUTPUT_STATE_SIZE = config_dict.get("OUTPUT_STATE_SIZE", 2)
        self.LATENT_DIM = config_dict.get("LATENT_DIM", 2)
        self.ENCODER_HIDDEN = config_dict.get("ENCODER_HIDDEN", 64)
        self.ENCODER_HIDDEN2 = config_dict.get("ENCODER_HIDDEN2", 32)
        self.ENCODER_HIDDEN3 = config_dict.get("ENCODER_HIDDEN3", 16)
        self.DECODER_HIDDEN = config_dict.get("DECODER_HIDDEN", 16)
        self.DECODER_HIDDEN2 = config_dict.get("DECODER_HIDDEN2", 32)
        self.DECODER_HIDDEN3 = config_dict.get("DECODER_HIDDEN3", 64)
        self.BETA_KL_DIV = config_dict.get("BETA_KL_DIV", 0.001)
        self.VAE_Version = config_dict.get("VAE_Version", "3.13")

        # Calculate derived values
        self.INPUT_DIMENSION = self.INPUT_STATE_SIZE * 8
        self.OUTPUT_DIMENSION = self.OUTPUT_STATE_SIZE * 8


def get_base_config() -> Dict[str, Any]:
    """Get the base configuration as a dictionary"""
    return {
        "INPUT_STATE_SIZE": config_module.INPUT_STATE_SIZE,
        "OUTPUT_STATE_SIZE": config_module.OUTPUT_STATE_SIZE,
        "LATENT_DIM": config_module.LATENT_DIM,
        "ENCODER_HIDDEN": config_module.ENCODER_HIDDEN,
        "ENCODER_HIDDEN2": config_module.ENCODER_HIDDEN2,
        "ENCODER_HIDDEN3": config_module.ENCODER_HIDDEN3,
        "DECODER_HIDDEN": config_module.DECODER_HIDDEN,
        "DECODER_HIDDEN2": config_module.DECODER_HIDDEN2,
        "DECODER_HIDDEN3": config_module.DECODER_HIDDEN3,
        "BETA_KL_DIV": config_module.BETA_KL_DIV,
        "VAE_Version": config_module.VAE_Version
    }


def worker(process_id: int,
           config_name: str,
           config_dict: Dict[str, Any],
           base_dir: str):
    """Worker function for training a single VAE configuration."""
    try:
        # Force CPU usage and limit threads per process
        torch.set_num_threads(1)

        # Set process name for monitoring
        process_name = f"vae_worker_{process_id}_{config_name}"
        try:
            import setproctitle
            setproctitle.setproctitle(process_name)
        except ImportError:
            pass

        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create configuration object
        base_config_dict = get_base_config()
        base_config_dict.update(config_dict)  # Override with new config values
        current_config = Config(base_config_dict)

        # Update global config module values for compatibility
        for key, value in vars(current_config).items():
            setattr(config_module, key, value)

        # Define evaluation data path
        eval_data = os.path.join(base_dir, "..", "Data_Collection", "collected_data",
                                 "eval", "rand_pol_rand_env","random_5000_20250218_162835.npz")

        # Training datasets and their corresponding VAE names
        datasets = [
            ("vae_ppo_noisy_100ep", "random_10000_20250218_160804.npz")
        ]

        # Process each dataset size
        for vae_name_base, data_file in datasets:
            vae_name = f"{vae_name_base}_{config_name}_{configs.config.EPOCHS}"
            train_data = os.path.join(base_dir, "..", "Data_Collection", "collected_data",
                                      "train","rand_pol_rand_env", data_file)

            print(f"Process {process_id} starting training for {vae_name}")

            # Run training and evaluation pipeline
            call_pretrain(vae_name=vae_name, data_dir=train_data)
            call_latent_colored(vae_name=vae_name, show=False, data_path=eval_data)
            call_reconstruction(vae_name,data_path=eval_data)
            vae_score_call(data_path=eval_data, vae_name=vae_name)

        # Save configurations
        save_config()
        save_vae_code()

        print(f"Process {process_id} completed configuration {config_name}")

    except Exception as e:
        print(f"Error in process {process_id} with config {config_name}: {str(e)}")
        raise e


def get_optimal_process_count() -> int:
    """Determine optimal number of processes based on CPU cores."""
    cpu_count = psutil.cpu_count(logical=False)
    return max(1, int(cpu_count * 0.75))


def main():
    # Get base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Initialize multiprocessing
    mp.set_start_method('spawn', force=True)

    # Determine optimal number of parallel processes
    num_processes = get_optimal_process_count()
    print(f"Running with {num_processes} parallel processes")

    # Create process pools
    processes = []

    try:
        # Launch processes for each configuration
        for i, (config_name, config_dict) in enumerate(SUITE_CONFIGS.items()):
            print(f"\nStarting configuration {config_name}")

            # Create and start process
            p = mp.Process(
                target=worker,
                args=(i, config_name, config_dict, base_dir)
            )
            p.start()
            processes.append(p)

            # Wait if we've reached the process limit
            if len(processes) >= num_processes:
                for p in processes:
                    p.join()
                processes = []

        # Wait for any remaining processes
        for p in processes:
            p.join()

    except KeyboardInterrupt:
        print("Detected keyboard interrupt, terminating processes...")
        for p in processes:
            p.terminate()
            p.join()
        raise

    except Exception as e:
        print(f"Error in main process: {str(e)}")
        for p in processes:
            p.terminate()
            p.join()
        raise

    print("All VAE configurations completed successfully!")


if __name__ == "__main__":
    main()