import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from average_eval import ppo_average, vae_ppo_average



# Define the base directory (directory of the current script)
base_dir = os.path.dirname(os.path.abspath(__file__))

def visualize(file_path = os.path.join(base_dir, "logs", "PPO" ,"averaged_evaluation_batch2.npz")):

    data = np.load(file_path)

    # Extract keys from the .npz file
    timesteps = data["timesteps"]  # Array of timesteps
    mean_rewards = data["mean_rewards"]  # Array of mean rewards
    std_rewards = data["std_rewards"]  # Array of standard deviations of rewards

    # Convert to a pandas DataFrame
    df = pd.DataFrame({
        "Timesteps": timesteps,
        "Mean Rewards": mean_rewards,
        "Std Dev": std_rewards
    })

    # Plot using Seaborn
    plt.figure(figsize=(10, 6))
    sns.lineplot(x="Timesteps", y="Mean Rewards", data=df, label="Mean Reward")
    plt.fill_between(
        df["Timesteps"],
        df["Mean Rewards"] - df["Std Dev"],
        df["Mean Rewards"] + df["Std Dev"],
        alpha=0.2,
        label="Standard Deviation"
    )

    # Customize the plot
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.title("Averaged Rewards Over Time")
    plt.legend()
    plt.grid()
    plt.show()

def visualize_2graphs(ax, file_path, title):
    """
    Visualize data from a given .npz file on a specific Axes object.

    Args:
        ax (matplotlib.axes.Axes): The axes to plot on.
        file_path (str): Path to the .npz file.
        title (str): Title for the subplot.
    """
    data = np.load(file_path)

    # Extract keys from the .npz file
    timesteps = data["timesteps"]  # Array of timesteps
    mean_rewards = data["mean_rewards"]  # Array of mean rewards
    std_rewards = data["std_rewards"]  # Array of standard deviations of rewards

    # Convert to a pandas DataFrame
    df = pd.DataFrame({
        "Timesteps": timesteps,
        "Mean Rewards": mean_rewards,
        "Std Dev": std_rewards
    })

    # Plot using Seaborn
    sns.lineplot(x="Timesteps", y="Mean Rewards", data=df, label="Mean Reward", ax=ax)
    ax.fill_between(
        df["Timesteps"],
        df["Mean Rewards"] - df["Std Dev"],
        df["Mean Rewards"] + df["Std Dev"],
        alpha=0.2,
        label="Standard Deviation"
    )

    # Customize the plot
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Reward")
    ax.set_title(title)
    ax.legend()
    ax.grid()


def two_graphs(ppo_file, vae_ppo_file):


    # Create a single figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # 1 row, 2 columns

    # Visualize PPO data
    visualize_2graphs(axes[0], ppo_file, title="PPO: Averaged Rewards Over Time")

    # Visualize VAE-PPO data
    visualize_2graphs(axes[1], vae_ppo_file, title="VAE-PPO: Averaged Rewards Over Time")

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()

def visualize_combined(ppo_file, vae_ppo_file):
    """
    Plot PPO and VAE-PPO rewards on the same graph with different colors.

    Args:
        ppo_file (str): Path to the PPO results file.
        vae_ppo_file (str): Path to the VAE-PPO results file.
    """
    # Load PPO data
    ppo_data = np.load(ppo_file)
    ppo_timesteps = ppo_data["timesteps"]
    ppo_mean_rewards = ppo_data["mean_rewards"]
    ppo_std_rewards = ppo_data["std_rewards"]

    # Load VAE-PPO data
    vae_ppo_data = np.load(vae_ppo_file)
    vae_ppo_timesteps = vae_ppo_data["timesteps"]
    vae_ppo_mean_rewards = vae_ppo_data["mean_rewards"]
    vae_ppo_std_rewards = vae_ppo_data["std_rewards"]

    # Create a single figure
    plt.figure(figsize=(10, 6))

    # Plot PPO with color blue
    sns.lineplot(x=ppo_timesteps, y=ppo_mean_rewards, label="PPO Mean Reward", color="blue")
    plt.fill_between(
        ppo_timesteps,
        ppo_mean_rewards - ppo_std_rewards,
        ppo_mean_rewards + ppo_std_rewards,
        alpha=0.2,
        color="blue",
        label="PPO Std Dev"
    )

    # Plot VAE-PPO with color red
    sns.lineplot(x=vae_ppo_timesteps, y=vae_ppo_mean_rewards, label="VAE-PPO Mean Reward", color="red")
    plt.fill_between(
        vae_ppo_timesteps,
        vae_ppo_mean_rewards - vae_ppo_std_rewards,
        vae_ppo_mean_rewards + vae_ppo_std_rewards,
        alpha=0.2,
        color="red",
        label="VAE-PPO Std Dev"
    )

    # Customize the plot
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.title("Comparison of PPO and VAE-PPO: Averaged Rewards Over Time")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__" :

    # define averaged files :
    ppo_average_dir = os.path.join(base_dir, "logs", "PPO", "rand_env_200k")
    #vae_ppo_average_dir = os.path.join("logs", "VAE_PPO", "V2", "rand_env_config1_1M")
    os.makedirs(ppo_average_dir, exist_ok=True)
    #os.makedirs(vae_ppo_average_dir, exist_ok=True)

    # average the rewards :
    ppo_average(output_file=os.path.join(ppo_average_dir, "batch_size_20.npz"),
                base_log_dir=os.path.join(base_dir, "..", "PPO", "logs", "explore", "batch_10_200k"))
    #vae_ppo_average(
    #    output_file=os.path.join(vae_ppo_average_dir, "batch_size_20.npz"),
    #    base_log_dir=os.path.join(base_dir, "..", "VAE_PPO_train", "logs", "batch_V2"))
    # Define file paths
    ppo_file = os.path.join(ppo_average_dir, "batch_size_20.npz")
    #vae_ppo_file = os.path.join(vae_ppo_average_dir, "batch_size_20.npz")
    visualize_combined(ppo_file , ppo_file)
