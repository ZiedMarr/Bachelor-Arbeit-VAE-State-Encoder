import numpy as np
import os

# Define the base directory (directory of the current script)

base_dir = os.path.dirname(os.path.abspath(__file__))

def vae_ppo_average (output_file,base_log_dir = os.path.join(base_dir, "..", "VAE_PPO_train", "logs", "batch2")) :
    # Recursively find all "eval" directories
    log_dirs = []
    for root, dirs, files in os.walk(base_log_dir):
        if "eval" in dirs:  # If "eval" is a subdirectory at this level
            log_dirs.append(os.path.join(root, "eval"))  # Append the full path
    #log_dirs = [
    #    "../VAE_PPO_train/logs/logs_20000_vae_15000_vae_offline_expert_20250114_165817_20250114_172631/eval/",
    #    "../VAE_PPO_train/logs/logs_20000_vae_offline_expert_20250114_160136/eval/",
    #    "../VAE_PPO_train/logs/logs_20000_vae_offline_expert_20250114_165649/eval/",
        # Add more paths as needed
    #]

    # Initialize storage for rewards and timesteps
    all_rewards = []
    timesteps = None

    # Load rewards from each log
    for log_dir in log_dirs:
        eval_file = os.path.join(log_dir, "evaluations.npz")
        data = np.load(eval_file)

        # Timesteps are the same for all logs
        if timesteps is None:
            timesteps = data["timesteps"]

        # Collect rewards
        all_rewards.append(data["results"])  # Shape: (n_evals, n_episodes)

    # Convert to numpy array
    all_rewards = np.array(all_rewards)  # Shape: (n_runs, n_evals, n_episodes)

    # Compute average and std across runs
    mean_rewards = all_rewards.mean(axis=(0, 2))  # Average across runs and episodes
    std_rewards = all_rewards.std(axis=(0, 2))  # Standard deviation

    # Save the aggregated log
    #output_file = os.path.join("logs", "VAE_PPO" , "averaged_evaluation_batch2.npz")
    np.savez(output_file, timesteps=timesteps, mean_rewards=mean_rewards, std_rewards=std_rewards)
    print(f"Averaged log saved to: {output_file}")


def ppo_average(output_file,base_log_dir = os.path.join(base_dir,"..", "PPO", "logs" , "batch2")) :# Define the base directory (directory of the current script)

    # Automatically populate log_dirs with subdirectories containing logs
    # Recursively find all "eval" directories
    log_dirs = []
    for root, dirs, files in os.walk(base_log_dir):
        if "eval" in dirs:  # If "eval" is a subdirectory at this level
            log_dirs.append(os.path.join(root, "eval"))  # Append the full path
    #log_dirs = [
    #    "../VAE_PPO_train/logs/logs_20000_vae_15000_vae_offline_expert_20250114_165817_20250114_172631/eval/",
    #    "../VAE_PPO_train/logs/logs_20000_vae_offline_expert_20250114_160136/eval/",
    #    "../VAE_PPO_train/logs/logs_20000_vae_offline_expert_20250114_165649/eval/",
        # Add more paths as needed
    #]
    # Initialize storage for rewards and timesteps
    all_rewards = []
    timesteps = None

    # Load rewards from each log
    for log_dir in log_dirs:
        eval_file = os.path.join(log_dir, "evaluations.npz")
        data = np.load(eval_file)

        # Timesteps are the same for all logs
        if timesteps is None:
            timesteps = data["timesteps"]

        # Collect rewards
        all_rewards.append(data["results"])  # Shape: (n_evals, n_episodes)

    # Convert to numpy array
    all_rewards = np.array(all_rewards)  # Shape: (n_runs, n_evals, n_episodes)
    print("Shape of all_rewards:", all_rewards.shape)

    # Compute average and std across runs
    mean_rewards = all_rewards.mean(axis=(0, 2))  # Average across runs and episodes
    std_rewards = all_rewards.std(axis=(0, 2))  # Standard deviation

    # Save the aggregated log
    #output_file = os.path.join("logs", "PPO" , "averaged_evaluation_batch2.npz")
    np.savez(output_file, timesteps=timesteps, mean_rewards=mean_rewards, std_rewards=std_rewards)
    print(f"Averaged log saved to: {output_file}")


if __name__ == "__main__":
    ppo_average(output_file=os.path.join(base_dir,"logs", "PPO" , "averaged_evaluation_rand_env_seed10_100k_3.npz" ),base_log_dir=os.path.join(base_dir,"..", "PPO", "logs" , "eval", "batch_eval_100k__3"))
    #vae_ppo_average(output_file= os.path.join("logs", "VAE_PPO" ,"V3.12", "averaged_evaluation_batch_V3.12_kl=0.002_100k_seed10.npz") ,base_log_dir=os.path.join(base_dir,"..", "VAE_PPO_train", "logs" , "batch_V3.12_kl=0.002_100k"))