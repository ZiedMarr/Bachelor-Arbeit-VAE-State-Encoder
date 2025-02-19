import os
import torch
import torch.multiprocessing as mp
import importlib
import sys
from typing import Dict, Any
from datetime import datetime
import psutil

from configs.suite_configs import SUITE_CONFIGS
from configs.save_config import save_config, save_vae_code


def update_config_module(config_updates: Dict[str, Any]) -> None:
    """
    Updates the configs.config module with values from config_updates.
    Forces module reload to ensure changes take effect.

    Args:
        config_updates: Dictionary of configuration values to update or add
    """
    # First, import the module
    import configs.config as config_module

    # Update each attribute in the module
    for key, value in config_updates.items():
        setattr(config_module, key, value)

    # Make sure any modules that already imported config get the updates
    # This is critical for multiprocessing
    if 'configs.config' in sys.modules:
        importlib.reload(sys.modules['configs.config'])

    # Verify changes were applied
    success = True
    for key, value in config_updates.items():
        current_value = getattr(config_module, key, None)
        if current_value != value:
            success = False
            print(f"Failed to update {key}: expected {value}, got {current_value}")

    if success:
        print(f"Successfully applied {len(config_updates)} configuration values")
    else:
        print("Some configuration updates failed")


def worker(process_id: int,
           config_name: str,
           config_updates: Dict[str, Any],
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

        print(f"Process {process_id} applying configuration {config_name}")

        # Update configuration values
        update_config_module(config_updates)

        # Import modules that use config AFTER updating it
        from VAE_pretrain.VAE_offline_pretrain import call_pretrain
        from VAE_Latent_Space_Eval.VAE_reconstructions import call_reconstruction
        from VAE_Latent_Space_Eval.VAE_score_Eval import vae_score_call
        from VAE_Latent_Space_Eval.Multi_Data_Latent_Colored_Plots import call_latent_colored

        # Verify configuration values to ensure they were updated correctly
        import configs.config as config_module
        print(f"Configuration verification for {config_name}:")
        for key, expected_value in config_updates.items():
            actual_value = getattr(config_module, key)
            print(f"  {key}: {actual_value} (Expected: {expected_value})")

        # Define evaluation data path
        eval_data = os.path.join(base_dir, "..", "Data_Collection", "collected_data",
                                 "eval", "rand_pol_rand_env", "random_5000_20250218_162835.npz")

        # Training datasets and their corresponding VAE names
        datasets = [
            ("vae_ppo_noisy_100ep", "random_10000_20250218_160804.npz")
        ]

        # Process each dataset
        for vae_name_base, data_file in datasets:
            # Import config module here to ensure we get the updated values
            import configs.config as config_module
            vae_name = f"{vae_name_base}_{config_name}_{config_module.EPOCHS}"
            train_data = os.path.join(base_dir, "..", "Data_Collection", "collected_data",
                                      "train", "rand_pol_rand_env", data_file)

            print(f"Process {process_id} starting training for {vae_name}")
            print(f"Using configuration: {config_name}")

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

    # Initialize multiprocessing with spawn method
    # 'spawn' creates a completely new Python interpreter process
    # which is crucial for isolating configurations between processes
    mp.set_start_method('spawn', force=True)

    # Determine optimal number of parallel processes
    num_processes = get_optimal_process_count()
    print(f"Running with {num_processes} parallel processes")

    # Create process pools
    processes = []

    try:
        # Launch processes for each configuration
        for i, (config_name, config_updates) in enumerate(SUITE_CONFIGS.items()):
            print(f"\nStarting configuration {config_name}")
            print(f"Configuration updates: {config_updates}")

            # Create and start process
            p = mp.Process(
                target=worker,
                args=(i, config_name, config_updates, base_dir)
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