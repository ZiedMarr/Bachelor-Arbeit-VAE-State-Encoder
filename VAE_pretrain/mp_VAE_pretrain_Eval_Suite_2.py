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
import importlib
import copy
from typing import Dict, Any


def get_config_with_updates(config_name: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a configuration dictionary by:
    1. Starting with all values from configs.config
    2. Adding any new keys from updates
    3. Overriding any existing keys with values from updates

    Args:
        config_name: Name of this configuration (for logging purposes)
        updates: Dictionary of config values to update or add

    Returns:
        Complete configuration dictionary with all needed values
    """
    # Force reload the config module to get fresh values
    import configs.config
    importlib.reload(configs.config)

    # Get all attributes from the config module
    config_dict = {}
    for attr in dir(configs.config):
        # Skip private/special attributes
        if not attr.startswith('__'):
            config_dict[attr] = getattr(configs.config, attr)

    # Make a deep copy to avoid modifying the original
    config_dict = copy.deepcopy(config_dict)

    # Add/update with new values
    config_dict.update(updates)

    print(f"Configuration '{config_name}' created with {len(updates)} custom values")
    return config_dict


def apply_config_to_module(config_dict: Dict[str, Any]) -> None:
    """
    Updates the global configs.config module with values from config_dict.
    This maintains compatibility with existing code that imports from configs.config.

    Args:
        config_dict: Dictionary of configuration values to apply
    """
    import configs.config

    # Update each attribute in the module
    for key, value in config_dict.items():
        setattr(configs.config, key, value)

    print(f"Applied {len(config_dict)} configuration values to configs.config module")


def worker(process_id: int, config_name: str, config_updates: Dict[str, Any], base_dir: str):
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

        # Create complete config and apply to global module
        config_dict = get_config_with_updates(config_name, config_updates)
        apply_config_to_module(config_dict)

        # Now all functions that import from configs.config will see the updated values

        # Define evaluation data path
        eval_data = os.path.join(base_dir, "..", "Data_Collection", "collected_data",
                                 "eval", "rand_pol_rand_env", "random_5000_20250218_162835.npz")

        # Training datasets and their corresponding VAE names
        datasets = [
            ("vae_ppo_noisy_100ep", "random_10000_20250218_160804.npz")
        ]

        # Process each dataset size
        for vae_name_base, data_file in datasets:
            vae_name = f"{vae_name_base}_{config_name}_{configs.config.EPOCHS}"
            train_data = os.path.join(base_dir, "..", "Data_Collection", "collected_data",
                                      "train", "rand_pol_rand_env", data_file)

            print(f"Process {process_id} starting training for {vae_name}")

            # Run training and evaluation pipeline
            call_pretrain(vae_name=vae_name, data_dir=train_data)
            call_latent_colored(vae_name=vae_name, show=False, data_path=eval_data)
            call_reconstruction(vae_name, data_path=eval_data)
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