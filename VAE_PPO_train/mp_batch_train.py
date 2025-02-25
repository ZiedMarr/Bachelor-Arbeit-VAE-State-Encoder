import os
import torch
import torch.multiprocessing as mp

#from VAE_PPO_train.model_batch_train import vae_model_path
from configs import eval_config
from configs.save_config import save_eval_config
from VAE_PPO_train.train import train
import psutil
from configs import config as config_module
from typing import Optional
from model_eval.visualize_averaged_reward import call_visualize_combined

# Get the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# List to keep track of results
success_list = []
error_list = []


def load_config_from_file(file_path):
    """Load configuration from a text file into a dictionary."""
    config_dict = {}

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and "=" in line:  # Ignore empty lines
                key, value = map(str.strip, line.split("=", 1))

                # Convert numerical values if possible
                try:
                    value = eval(value)  # Convert numbers & tuples (be careful with eval)
                except:
                    pass  # Keep as string if conversion fails

                config_dict[key] = value

    return config_dict

class Config:
    """Dynamically loads attributes from a dictionary."""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)  # Dynamically add attributes

def update_config(config_path) :
    # Load config dictionary from file
    config_dict = load_config_from_file(config_path)
    # Create a Config object with loaded values
    config = Config(config_dict)
    #update config_module
    # Update global config module values for compatibility
    for key, value in vars(config).items():
        setattr(config_module, key, value)

def worker(process_id: int,
           vae_model_path: str,
           vae_save_folder: str,
           log_batch_dir: str,
           total_timesteps: int,
           seed: int,
           vae_config : str):
    """Worker function for each training process."""
    try:
        # Force CPU usage
        torch.set_num_threads(1)  # Important: limit threads per process

        # Set process name for better monitoring
        process_name = f"train_worker_{process_id}"
        try:
            import setproctitle
            setproctitle.setproctitle(process_name)
        except ImportError:
            pass

        # Create unique log directory for this process
        process_log_dir = os.path.join(log_batch_dir, f"process_{process_id}")
        os.makedirs(process_log_dir, exist_ok=True)

        #modifiy config_module
        update_config(vae_config)

        # Run training
        train(
            vae_model_path=vae_model_path,
            vae_save_folder=vae_save_folder,
            log_batch_dir=process_log_dir,
            total_timesteps=total_timesteps,
            seed=seed
        )

    except Exception as e:
        print(f"Error in process {process_id}: {str(e)}")
        raise e


def get_optimal_process_count() -> int:
    """Determine optimal number of processes based on CPU cores."""
    cpu_count = psutil.cpu_count(logical=False)  # Physical CPU cores

    # Use 75% of available cores (rounded down) to leave some headroom
    # but at least 1 process
    return max(1, int(cpu_count * 0.75))


def main(vae_config, batch = "batch_V3.13_kl=0.002_evalconfig3_100k" ,   vae_model_path = os.path.join(script_dir, "..", "VAE_pretrain", "pretrained_vae","VAE_Version_3.13", "2_2", "KL-D_0.002", "vae_rand_500k")):


    # Setup batch configuration

    batch_size = eval_config.BATCH_SIZE
    total_timesteps = eval_config.TOTAL_TIMESTEPS

    # Setup paths

    vae_save_folder = os.path.join(script_dir, "trained_vae", batch)
    log_batch_dir = os.path.join(script_dir, "logs", batch)

    # Save evaluation config
    save_eval_config(log_batch_dir)

    # Create necessary directories
    os.makedirs(vae_save_folder, exist_ok=True)
    os.makedirs(log_batch_dir, exist_ok=True)

    # Get seeds
    seeds = eval_config.EVAL_SEED

    # Determine optimal number of parallel processes
    num_processes = min(get_optimal_process_count(), batch_size)
    print(f"Running training with {num_processes} parallel processes")

    # Initialize multiprocessing method
    mp.set_start_method('spawn', force=True)

    # Create process pools
    processes = []

    try:
        # Launch processes
        for i in range(batch_size):
            # Create and start process
            p = mp.Process(
                target=worker,
                args=(i, vae_model_path, vae_save_folder, log_batch_dir,
                      total_timesteps, seeds[i], vae_config)
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

    print("All training processes completed successfully!")

def batch_train_module(  vae_version,vae_name, in_out , kl , vae_config ,vae_path=os.path.join("..", "VAE_pretrain", "pretrained_vae","VAE_Version_2.1", "4_2", "KL-D_0.0008") ) :
    vae_config_path = os.path.join(script_dir, vae_path, vae_config)


    #train :
    vae_model_path = os.path.join(script_dir, vae_path, vae_name)
    batch = f"batch_1M_{vae_version}_{vae_name}"
    main(batch = batch, vae_model_path = vae_model_path, vae_config=vae_config_path)
    #visualize :

    call_visualize_combined(vae_batch=batch, vae_version=vae_version, in_out=in_out , kl=kl)

def safe_batch_train(vae_name, vae_config, vae_path, vae_version, in_out , kl):
    """Wrapper for batch_train_module that handles errors and logs successes."""
    try:
        batch_train_module(vae_name=vae_name, vae_config=vae_config, vae_path=vae_path, vae_version=vae_version, in_out=in_out , kl=kl)
        success_list.append(f"✅ SUCCESS: {vae_name} with {vae_config}")
    except Exception as e:
        error_list.append(f"❌ ERROR: {vae_name} with {vae_config} → {str(e)}")


if __name__ == "__main__":
    # First batch of VAE trainings from VAE_Version_2.1
    vae_version = "VAE_Version_2.1"
    in_out = "4_2"
    kl = "KL-D_0.0008"
    vae_path = os.path.join("..", "VAE_pretrain", "pretrained_vae", vae_version, in_out, kl)
    safe_batch_train(vae_name="vae_random_100ep_config_H2_2", vae_config="VAE_config_config_H2.txt", vae_path=vae_path,
                       vae_version=vae_version, in_out=in_out , kl=kl)

    vae_path = os.path.join("..", "VAE_pretrain", "pretrained_vae", vae_version, "2_2", "KL-D_0.001")
    safe_batch_train(vae_name="vae_random_100ep_config_A_2", vae_config="VAE_config_config_A.txt",
                       vae_path=vae_path, vae_version=vae_version)

    vae_path = os.path.join("..", "VAE_pretrain", "pretrained_vae", vae_version, "4_2", "KL-D_0.0009")
    safe_batch_train(vae_name="vae_random_100ep_config_K_2", vae_config="VAE_config_config_K.txt",
                       vae_path=vae_path, vae_version=vae_version)

    vae_path = os.path.join("..", "VAE_pretrain", "pretrained_vae", vae_version, "5_2", "KL-D_0.001")
    safe_batch_train(vae_name="vae_random_100ep_config_M_2", vae_config="VAE_config_config_M.txt",
                       vae_path=vae_path, vae_version=vae_version)

    # Second batch from VAE_Version_2.2
    vae_version = "VAE_Version_2.2"
    vae_path = os.path.join("..", "VAE_pretrain", "pretrained_vae", vae_version, "2_2", "KL-D_0.0008")
    safe_batch_train(vae_name="vae_random_100ep_config_B_2", vae_config="VAE_config_config_B.txt",
                       vae_path=vae_path, vae_version=vae_version)

    vae_path = os.path.join("..", "VAE_pretrain", "pretrained_vae", vae_version, "2_2", "KL-D_0.0009")
    safe_batch_train(vae_name="vae_random_100ep_config_C_2", vae_config="VAE_config_config_C.txt",
                       vae_path=vae_path, vae_version=vae_version)

    vae_path = os.path.join("..", "VAE_pretrain", "pretrained_vae", vae_version, "2_2", "KL-D_0.001")
    safe_batch_train(vae_name="vae_random_100ep_config_A_2", vae_config="VAE_config_config_A.txt",
                       vae_path=vae_path, vae_version=vae_version)

    vae_path = os.path.join("..", "VAE_pretrain", "pretrained_vae", vae_version, "4_2", "KL-D_0.0008")
    safe_batch_train(vae_name="vae_random_100ep_config_D_2", vae_config="VAE_config_config_D.txt",
                       vae_path=vae_path, vae_version=vae_version)

    vae_path = os.path.join("..", "VAE_pretrain", "pretrained_vae", vae_version, "4_2", "KL-D_0.001")
    safe_batch_train(vae_name="vae_random_100ep_config_E_2", vae_config="VAE_config_config_E.txt",
                       vae_path=vae_path, vae_version=vae_version)

    # Last batch from VAE_Version_2.1 (another unique one)
    vae_version = "VAE_Version_2.1"
    vae_path = os.path.join("..", "VAE_pretrain", "pretrained_vae", vae_version, "4_2", "KL-D_0.001")
    safe_batch_train(vae_name="vae_random_100ep_config_G_2", vae_config="VAE_config_config_G.txt",
                       vae_path=vae_path, vae_version=vae_version)

    # Print results
    print("\n=== SUMMARY REPORT ===")
    print("\n✔ Successful Runs:")
    for success in success_list:
        print(success)

    print("\n❌ Errors Encountered:")
    for error in error_list:
        print(error)
