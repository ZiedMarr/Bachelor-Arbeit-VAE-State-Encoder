import numpy as np
import os


base_log_dir = "../VAE_PPO_train/logs/batch1"
# Paths to the evaluation logs
# Automatically populate log_dirs with subdirectories containing logs
log_dirs = [
    os.path.join(base_log_dir, folder, "eval")
    for folder in os.listdir(base_log_dir)
    if os.path.isdir(os.path.join(base_log_dir, folder, "eval"))
]
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
output_file = "logs/averaged_evaluation_batch1.npz"
np.savez(output_file, timesteps=timesteps, mean_rewards=mean_rewards, std_rewards=std_rewards)
print(f"Averaged log saved to: {output_file}")
