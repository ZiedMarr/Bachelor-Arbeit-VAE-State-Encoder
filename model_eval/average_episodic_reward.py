import gymnasium as gym
import torch
import numpy as np
from stable_baselines3 import PPO
import os

import json
from datetime import datetime

from Wrappers.Wrapped_environment import VAEWrapperWithHistory
from VAE import VAE
from configs import config

# Get the current working directory
current_dir = os.getcwd()
def evaluate_ppo_model(ppo_model_path, n_episodes=100, seed=42, save_path=None):
    """
    Load PPO model and evaluate it over n episodes.

    Args:
        ppo_model_path: Path to the trained PPO model
        n_episodes: Number of evaluation episodes
        seed: Random seed for reproducibility
        save_path: Directory to save results

    Returns:
        mean_reward: Mean reward across all episodes
        std_reward: Standard deviation of rewards
        all_rewards: List of rewards for each episode
    """
    print(f"Evaluating PPO from: {ppo_model_path}")
    print(f"Running {n_episodes} evaluation episodes...")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create environment
    env = gym.make("LunarLander-v3")
    env.reset(seed=seed)
    env.observation_space.seed(seed)

    # Load PPO model
    ppo_model = PPO.load(ppo_model_path, device=device)

    # Run evaluation episodes
    all_rewards = []

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode)  # Different seed for each episode
        done = False
        truncated = False
        total_reward = 0

        while not (done or truncated):
            action, _ = ppo_model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward

        all_rewards.append(total_reward)
        print(f"Episode {episode + 1}/{n_episodes}: Reward = {total_reward:.2f}")

    # Calculate statistics
    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    min_reward = np.min(all_rewards)
    max_reward = np.max(all_rewards)
    median_reward = np.median(all_rewards)

    # Print results
    print("-" * 50)
    print(f"Evaluation results over {n_episodes} episodes:")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Median reward: {median_reward:.2f}")
    print(f"Min reward: {min_reward:.2f}")
    print(f"Max reward: {max_reward:.2f}")
    print("-" * 50)

    # Save results if path is provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Extract model name for the filename
        ppo_name = os.path.basename(ppo_model_path).replace(".zip", "")

        # Save individual rewards to a NumPy file
        rewards_np_path = os.path.join(save_path, f"rewards_{ppo_name}_{timestamp}.npy")
        np.save(rewards_np_path, np.array(all_rewards))

        # Save rewards to a text file
        rewards_txt_path = os.path.join(save_path, f"rewards_{ppo_name}_{timestamp}.txt")
        with open(rewards_txt_path, 'w') as f:
            f.write(f"Evaluation of PPO: {ppo_model_path}\n")
            f.write(f"Number of episodes: {n_episodes}\n")
            f.write(f"Seed: {seed}\n\n")
            f.write("-" * 50 + "\n")
            f.write(f"Mean reward: {mean_reward:.4f}\n")
            f.write(f"Standard deviation: {std_reward:.4f}\n")
            f.write(f"Median reward: {median_reward:.4f}\n")
            f.write(f"Min reward: {min_reward:.4f}\n")
            f.write(f"Max reward: {max_reward:.4f}\n")
            f.write("-" * 50 + "\n\n")
            f.write("Individual episode rewards:\n")
            for i, reward in enumerate(all_rewards):
                f.write(f"Episode {i + 1}: {reward:.4f}\n")

        # Save summary in JSON format
        summary_path = os.path.join(save_path, f"summary_{ppo_name}_{timestamp}.json")
        summary = {
            "ppo_model": ppo_model_path,
            "episodes": n_episodes,
            "seed": seed,
            "statistics": {
                "mean": float(mean_reward),
                "std": float(std_reward),
                "median": float(median_reward),
                "min": float(min_reward),
                "max": float(max_reward)
            },
            "timestamp": timestamp
        }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)

        print(f"Results saved to:")
        print(f"  - {rewards_txt_path}")
        print(f"  - {rewards_np_path}")
        print(f"  - {summary_path}")

    return mean_reward, std_reward, all_rewards



def evaluate_model(vae_model_path, ppo_model_path, n_episodes=100, seed=42, save_path=None):
    """
    Load VAE and PPO models and evaluate them over n episodes.

    Args:
        vae_model_path: Path to the trained VAE model
        ppo_model_path: Path to the trained PPO model
        n_episodes: Number of evaluation episodes
        seed: Random seed for reproducibility
        save_path: Directory to save results

    Returns:
        mean_reward: Mean reward across all episodes
        std_reward: Standard deviation of rewards
        all_rewards: List of rewards for each episode
    """
    print(f"Evaluating VAE from: {vae_model_path}")
    print(f"Evaluating PPO from: {ppo_model_path}")
    print(f"Running {n_episodes} evaluation episodes...")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize VAE
    vae = VAE(input_dim=config.INPUT_DIMENSION,
              latent_dim=config.LATENT_DIM,
              output_dim=config.OUTPUT_DIMENSION)
    vae.load_state_dict(torch.load(vae_model_path, map_location=device))
    vae.to(device)
    vae.eval()  # Set to evaluation mode

    # Create optimizer (needed for environment wrapper but won't be used in eval)
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

    # Create environment with VAE wrapper
    env = gym.make("LunarLander-v3")
    env.reset(seed=seed)
    env.observation_space.seed(seed)

    # Create wrapped environment
    n = config.INPUT_STATE_SIZE
    m = config.OUTPUT_STATE_SIZE
    wrapped_env = VAEWrapperWithHistory(env, vae, n=n, m=m, vae_optimizer=vae_optimizer)

    # Load PPO model
    ppo_model = PPO.load(ppo_model_path, device='cpu')

    # Run evaluation episodes
    all_rewards = []

    for episode in range(n_episodes):  # Removed tqdm to fix potential dependency issues
        obs, _ = wrapped_env.reset(seed=seed + episode)  # Different seed for each episode
        done = False
        truncated = False
        total_reward = 0

        while not (done or truncated):
            action, _ = ppo_model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = wrapped_env.step(action)
            total_reward += reward

        all_rewards.append(total_reward)
        print(f"Episode {episode + 1}/{n_episodes}: Reward = {total_reward:.2f}")

    return all_rewards

def compute_and_save_results(all_rewards, output_path,batch_dir):
    """
    Computes mean, std, min, max, median from multiple PPO evaluations and saves results.

    Args:
        all_rewards: List of lists, where each inner list contains episode rewards from one model
        output_path: Path where the results should be saved
    """
    all_rewards = np.array(all_rewards)  # Shape: (num_models, num_episodes)

    # Compute mean and std across models
    mean_rewards = np.mean(all_rewards, axis=0)  # Mean per episode
    std_rewards = np.std(all_rewards, axis=0)  # Standard deviation per episode

    # Compute overall statistics
    overall_mean = np.mean(mean_rewards)
    overall_std = np.mean(std_rewards)
    overall_min = np.min(all_rewards)
    overall_max = np.max(all_rewards)
    overall_median = np.median(all_rewards)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save numpy array of all results
    np.save(os.path.join(output_path, f"batch_rewards_{timestamp}.npy"), all_rewards)

    # Save summary as text file
    summary_file = os.path.join(output_path, f"summary_batch_{timestamp}.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Batch Evaluation of PPO models from {batch_dir}\n")
        f.write(f"Number of models: {len(all_rewards)}\n")
        f.write(f"Number of episodes per model: {n_episodes}\n")
        f.write(f"Seed: {seed}\n\n")
        f.write("-" * 50 + "\n")
        f.write(f"Overall Mean reward: {overall_mean:.4f}\n")
        f.write(f"Overall Std reward: {overall_std:.4f}\n")
        f.write(f"Overall Min reward: {overall_min:.4f}\n")
        f.write(f"Overall Max reward: {overall_max:.4f}\n")
        f.write(f"Overall Median reward: {overall_median:.4f}\n")
        f.write("-" * 50 + "\n\n")
        f.write("Mean reward per episode:\n")
        for i, reward in enumerate(mean_rewards):
            f.write(f"Episode {i + 1}: {reward:.4f} ± {std_rewards[i]:.4f}\n")

    print(f"Batch evaluation completed. Results saved to: {summary_file}")

# Set your parameters directly in the script using os.path.join
if __name__ == "__main__":
    ###########################VAE-PPO BLOCK ######################################

    # Model paths using os.path.join
    vae_model_path = '../VAE_PPO_train/trained_vae/batch_1M_VAE_Version_2.2_vae_mix_10ep_config_B_2/500000_vae_mix_10ep_config_B_2_20250307_143717'

    ppo_model_batch = '../VAE_PPO_train/logs/batch_1M_VAE_Version_2.2_vae_mix_10ep_config_B_2'

    model_paths = []
    for root, _, files in os.walk(ppo_model_batch):
        if "best_model.zip" in files:
            model_paths.append(os.path.join(root, "best_model.zip"))

    print(f"Found {len(model_paths)} models.")

    # Evaluation parameters
    n_episodes = 10  # Number of episodes to evaluate
    seed = 120
    save_path = os.path.join(".", "average_episodic_rewards", "6_2_B2")  # Directory to save results
    os.makedirs(save_path, exist_ok=True)

    # Run evaluation
    all_rewards =[evaluate_model(
        vae_model_path,
        ppo_model_path,
        n_episodes=n_episodes,
        seed=seed,
        save_path=save_path
    ) for ppo_model_path in model_paths]

    # Compute and save final results
    compute_and_save_results(all_rewards, save_path, batch_dir=ppo_model_batch)
    '''
    ###########################VAE-PPO BLOCK ######################################
    # PPO Model path
    ppo_model_path = '../PPO/logs/eval/batch_20_50k/process_18/logs_1000000/best_model/best_model.zip'

    # Evaluation parameters
    n_episodes = 100  # Number of episodes to evaluate
    seed = 43
    save_path = os.path.join(".", "average_episodic_rewards", "PPO")  # Directory to save results
    
    
    # Run evaluation
    mean_reward, std_reward, all_rewards = evaluate_ppo_model(
        ppo_model_path,
        n_episodes=n_episodes,
        seed=seed,
        save_path=save_path
    )
  '''
