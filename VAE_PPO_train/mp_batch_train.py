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
    '''
    # 1st VAE
    vae_version = "VAE_Version_4.1"
    in_out = "2_2"
    kl = "KL-D_0.001"
    vae_path = os.path.join("..", "VAE_pretrain", "pretrained_vae", vae_version, in_out, kl)
    safe_batch_train(vae_name="vae_random_100ep_config_compact_2", vae_config="VAE_config_config_compact.txt",
                     vae_path=vae_path, vae_version=vae_version, in_out=in_out, kl=kl)

    # 2nd VAE
    vae_version = "VAE_Version_4.2"
    in_out = "2_2"
    kl = "KL-D_0.0008"
    vae_path = os.path.join("..", "VAE_pretrain", "pretrained_vae", vae_version, in_out, kl)
    safe_batch_train(vae_name="vae_random_100ep_config_bottleneck_2", vae_config="VAE_config_config_bottleneck.txt",
                     vae_path=vae_path, vae_version=vae_version, in_out=in_out, kl=kl)

    # 3rd VAE
    vae_version = "VAE_Version_4.3"
    in_out = "3_3"
    kl = "KL-D_0.0005"
    vae_path = os.path.join("..", "VAE_pretrain", "pretrained_vae", vae_version, in_out, kl)
    safe_batch_train(vae_name="vae_random_100ep_config_wide_2", vae_config="VAE_config_config_wide.txt",
                     vae_path=vae_path, vae_version=vae_version, in_out=in_out, kl=kl)

    # 4th VAE
    vae_version = "VAE_Version_4.4"
    in_out = "1_1"
    kl = "KL-D_0.0003"
    vae_path = os.path.join("..", "VAE_pretrain", "pretrained_vae", vae_version, in_out, kl)
    safe_batch_train(vae_name="vae_random_100ep_config_minimal_2", vae_config="VAE_config_config_minimal.txt",
                     vae_path=vae_path, vae_version=vae_version, in_out=in_out, kl=kl)

    # 5th VAE
    vae_version = "VAE_Version_4.5"
    in_out = "2_2"
    kl = "KL-D_0.0009"
    vae_path = os.path.join("..", "VAE_pretrain", "pretrained_vae", vae_version, in_out, kl)
    safe_batch_train(vae_name="vae_random_100ep_config_latent_rich_2", vae_config="VAE_config_config_latent_rich.txt",
                     vae_path=vae_path, vae_version=vae_version, in_out=in_out, kl=kl)
    '''
    # 6th VAE
    vae_version = "VAE_Version_4.6"
    in_out = "2_2"
    kl = "KL-D_0.001"
    vae_path = os.path.join("..", "VAE_pretrain", "pretrained_vae", vae_version, in_out, kl)
    safe_batch_train(vae_name="vae_random_100ep_config_dense_2", vae_config="VAE_config_config_dense.txt",
                     vae_path=vae_path, vae_version=vae_version, in_out=in_out, kl=kl)

    # 7th VAE
    vae_version = "VAE_Version_4.7"
    in_out = "3_2"
    kl = "KL-D_0.0005"
    vae_path = os.path.join("..", "VAE_pretrain", "pretrained_vae", vae_version, in_out, kl)
    safe_batch_train(vae_name="vae_random_100ep_config_asymmetric_2", vae_config="VAE_config_config_asymmetric.txt",
                     vae_path=vae_path, vae_version=vae_version, in_out=in_out, kl=kl)

    # 8th VAE
    vae_version = "VAE_Version_4.8"
    in_out = "1_1"
    kl = "KL-D_0.0008"
    vae_path = os.path.join("..", "VAE_pretrain", "pretrained_vae", vae_version, in_out, kl)
    safe_batch_train(vae_name="vae_random_100ep_config_micro_2", vae_config="VAE_config_config_micro.txt",
                     vae_path=vae_path, vae_version=vae_version, in_out=in_out, kl=kl)

    # 9th VAE
    vae_version = "VAE_Version_4.9"
    in_out = "4_4"
    kl = "KL-D_0.0003"
    vae_path = os.path.join("..", "VAE_pretrain", "pretrained_vae", vae_version, in_out, kl)
    safe_batch_train(vae_name="vae_random_100ep_config_complex_2", vae_config="VAE_config_config_complex.txt",
                     vae_path=vae_path, vae_version=vae_version, in_out=in_out, kl=kl)

    # 10th VAE
    vae_version = "VAE_Version_4.10"
    in_out = "2_2"
    kl = "KL-D_0.001"
    vae_path = os.path.join("..", "VAE_pretrain", "pretrained_vae", vae_version, in_out, kl)
    safe_batch_train(vae_name="vae_random_100ep_config_balanced_2", vae_config="VAE_config_config_balanced.txt",
                     vae_path=vae_path, vae_version=vae_version, in_out=in_out, kl=kl)

    # 11th VAE
    vae_version = "VAE_Version_4.11"
    in_out = "2_2"
    kl = "KL-D_0.0009"
    vae_path = os.path.join("..", "VAE_pretrain", "pretrained_vae", vae_version, in_out, kl)
    safe_batch_train(vae_name="vae_random_100ep_config_narrow_2", vae_config="VAE_config_config_narrow.txt",
                     vae_path=vae_path, vae_version=vae_version, in_out=in_out, kl=kl)

    # 12th VAE
    vae_version = "VAE_Version_4.12"
    in_out = "2_2"
    kl = "KL-D_0.0005"
    vae_path = os.path.join("..", "VAE_pretrain", "pretrained_vae", vae_version, in_out, kl)
    safe_batch_train(vae_name="vae_random_100ep_config_exp_decay_2", vae_config="VAE_config_config_exp_decay.txt",
                     vae_path=vae_path, vae_version=vae_version, in_out=in_out, kl=kl)

    # 13th VAE
    vae_version = "VAE_Version_4.13"
    in_out = "2_2"
    kl = "KL-D_0.0003"
    vae_path = os.path.join("..", "VAE_pretrain", "pretrained_vae", vae_version, in_out, kl)
    safe_batch_train(vae_name="vae_random_100ep_config_deeper_latent_2",
                     vae_config="VAE_config_config_deeper_latent.txt",
                     vae_path=vae_path, vae_version=vae_version, in_out=in_out, kl=kl)

    # 14th VAE
    vae_version = "VAE_Version_4.14"
    in_out = "5_5"
    kl = "KL-D_0.0008"
    vae_path = os.path.join("..", "VAE_pretrain", "pretrained_vae", vae_version, in_out, kl)
    safe_batch_train(vae_name="vae_random_100ep_config_wider_io_2", vae_config="VAE_config_config_wider_io.txt",
                     vae_path=vae_path, vae_version=vae_version, in_out=in_out, kl=kl)

    # 15th VAE
    vae_version = "VAE_Version_4.15"
    in_out = "3_3"
    kl = "KL-D_0.001"
    vae_path = os.path.join("..", "VAE_pretrain", "pretrained_vae", vae_version, in_out, kl)
    safe_batch_train(vae_name="vae_random_100ep_config_pyramid_2", vae_config="VAE_config_config_pyramid.txt",
                     vae_path=vae_path, vae_version=vae_version, in_out=in_out, kl=kl)
                     


    # Define log file path
    log_dir = os.path.join(script_dir, "training_log")
    os.makedirs(log_dir, exist_ok=True)  # Ensure the directory exists
    log_file_path = os.path.join(log_dir, "training_summary.log")

    # Open the file and write the logs
    with open(log_file_path, "w") as log_file:
        log_file.write("=== SUMMARY REPORT ===\n\n")

        log_file.write("✔ Successful Runs:\n")
        for success in success_list:
            log_file.write(success + "\n")

        log_file.write("\n❌ Errors Encountered:\n")
        for error in error_list:
            log_file.write(error + "\n")

    # Print confirmation
    print(f"\nSummary log saved at: {log_file_path}")