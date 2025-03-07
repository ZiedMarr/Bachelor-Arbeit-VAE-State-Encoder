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


def evaluate_model(vae_model_path, ppo_model_path, n_episodes=100, seed=43, save_path=None):
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
    ppo_model = PPO.load(ppo_model_path, env=wrapped_env)

    # Run evaluation episodes
    all_rewards = []

    for episode in range(n_episodes):
        obs, _ = wrapped_env.reset(seed=seed + episode)  # Different seed for each episode
        done = False
        truncated = False
        total_reward = 0

        while not (done or truncated):
            action, _ = ppo_model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = wrapped_env.step(action)
            total_reward += reward

        all_rewards.append(total_reward)

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

        # Extract model names for the filename
        vae_name = os.path.basename(vae_model_path)
        ppo_name = os.path.basename(ppo_model_path)

        # Save individual rewards to numpy file
        rewards_np_path = os.path.join(save_path, f"rewards_{vae_name}_{ppo_name}_{timestamp}.npy")
        np.save(rewards_np_path, np.array(all_rewards))

        # Save rewards to text file
        rewards_txt_path = os.path.join(save_path, f"rewards_{vae_name}_{ppo_name}_{timestamp}.txt")
        with open(rewards_txt_path, 'w') as f:
            f.write(f"Evaluation of VAE: {vae_model_path}\n")
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
        summary_path = os.path.join(save_path, f"summary_{vae_name}_{ppo_name}_{timestamp}.json")
        summary = {
            "vae_model": vae_model_path,
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


# Set your parameters directly in the script
if __name__ == "__main__":
    # Model paths
    vae_model_path = "../VAE_PPO_train/trained_vae/batch_V2/1000000_vae_ppo_noisy_100ep_config_D_5_20250219_143544"  # Path to your VAE model
    ppo_model_path = "../VAE_PPO_train/logs/batch_V2/process_19/logs_1000000_vae_ppo_noisy_100ep_config_D_5/best_model/best_model.zip"  # Path to your PPO model

    # Evaluation parameters
    n_episodes = 20  # Number of episodes to evaluate
    seed = 43
    save_path = "./average_episodic_rewards/2_2_D5"  # Directory to save results

    # Run evaluation
    mean_reward, std_reward, all_rewards = evaluate_model(
        vae_model_path,
        ppo_model_path,
        n_episodes=n_episodes,
        seed=seed,
        save_path=save_path
    )