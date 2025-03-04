import os
import torch
import torch.multiprocessing as mp
from PPO.train_ppo import train_ppo
from configs import eval_config
from configs.save_config import save_eval_config
from Wrappers.RandomStartLunarLander import  RandomStartLunarLander
import psutil
from typing import Optional
import time
import json
from datetime import datetime, timedelta

def worker(process_id: int,
           log_batch_dir: str,
           total_timesteps: int,
           seed: int,
           env_wrapper):
    """Worker function for each training process."""
    try:
        # Force CPU usage and limit threads per process
        torch.set_num_threads(1)

        # Set process name for better monitoring
        process_name = f"ppo_worker_{process_id}"
        try:
            import setproctitle
            setproctitle.setproctitle(process_name)
        except ImportError:
            pass

        # Create unique log directory for this process
        process_log_dir = os.path.join(log_batch_dir, f"process_{process_id}")
        os.makedirs(process_log_dir, exist_ok=True)

        # Start timing
        start_time = time.time()

        # Run training
        train_ppo(
            log_batch_dir=process_log_dir,
            total_timesteps=total_timesteps,
            seed=seed,
            env_wrapper=env_wrapper
        )

        # End timing
        end_time = time.time()
        training_time = end_time - start_time

        # Save timing data to a JSON file in the process directory
        timing_info = {
            'process_id': process_id,
            'start_time': datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S"),
            'end_time': datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S"),
            'training_time_seconds': training_time,
            'training_time_formatted': str(timedelta(seconds=int(training_time))),
        }

        # Write timing data to process directory
        timing_file = os.path.join(process_log_dir, "training_time.json")
        with open(timing_file, 'w') as f:
            json.dump(timing_info, f, indent=4)


    except Exception as e:
        print(f"Error in process {process_id}: {str(e)}")
        raise e


def get_optimal_process_count() -> int:
    """Determine optimal number of processes based on CPU cores."""
    cpu_count = psutil.cpu_count(logical=False)  # Physical CPU cores
    return max(1, int(cpu_count * 0.75))  # Use 75% of available cores


def main(batch = "batch_evalconfig3_100k"):
    # Get the current script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Setup batch configuration

    batch_size = eval_config.BATCH_SIZE
    total_timesteps = eval_config.TOTAL_TIMESTEPS

    # Setup log directory
    log_batch_dir = os.path.join(script_dir, "logs", "explore", batch)

    # Save evaluation config
    save_eval_config(log_batch_dir)

    # Create necessary directories
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
                args=(i, log_batch_dir, total_timesteps, seeds[i], RandomStartLunarLander)
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


if __name__ == "__main__":
    main(batch = "batch_10_200k")