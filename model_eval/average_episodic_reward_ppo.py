import os
import numpy as np
from stable_baselines3 import PPO
import gymnasium as gym
from datetime import datetime

# Set base directories
batch_dir = "../PPO/logs/eval/batch_20_50k"  # Change to your batch directory
env_name = "LunarLander-v3"  # Change if needed
n_episodes = 10  # Number of episodes per model
seed = 120  # Random seed for consistency
save_path = os.path.join(".", "average_episodic_rewards", "PPO", "batch_result")  # Where to save results

# Ensure save directory exists
os.makedirs(save_path, exist_ok=True)

# Find all best_model.zip files inside batch directory
model_paths = []
for root, _, files in os.walk(batch_dir):
    if "best_model.zip" in files:
        model_paths.append(os.path.join(root, "best_model.zip"))

print(f"Found {len(model_paths)} models.")

def evaluate_model(ppo_model_path,  n_episodes, seed, env_name="LunarLander-v3"):
    """
    Evaluates a PPO model over multiple episodes.

    Args:
        ppo_model_path: Path to the PPO model
        env_name: Gym environment name
        n_episodes: Number of evaluation episodes
        seed: Random seed

    Returns:
        rewards: List of episode rewards
    """
    print(f"Evaluating PPO model: {ppo_model_path}")

    # Load PPO model
    model = PPO.load(ppo_model_path, device='cpu')

    # Create environment
    env = gym.make("LunarLander-v3")
    env.reset(seed=seed)
    env.observation_space.seed(seed)

    # Run evaluation episodes
    episode_rewards = []
    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        truncated = False
        total_reward = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward

        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}/{n_episodes}: Reward = {total_reward:.2f}")

    return episode_rewards

def compute_and_save_results(all_rewards, output_path):
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

# Evaluate all models and store results
all_rewards = [evaluate_model(model_path, n_episodes, seed , env_name=env_name) for model_path in model_paths]

# Compute and save final results
compute_and_save_results(all_rewards, save_path)
