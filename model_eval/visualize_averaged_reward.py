import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


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


if __name__ == "__main__":
    #visualize(os.path.join(base_dir, "logs", "PPO" ,"averaged_evaluation_batch2.npz"))
    #visualize(os.path.join(base_dir, "logs", "VAE_PPO" ,"averaged_evaluation_batch2.npz"))
    # Define file paths
    ppo_file = os.path.join(base_dir, "logs", "PPO", "averaged_evaluation_batch2.npz")
    vae_ppo_file = os.path.join(base_dir, "logs", "VAE_PPO" ,"averaged_evaluation_batch_explore.npz")

    # Create a single figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # 1 row, 2 columns

    # Visualize PPO data
    visualize_2graphs(axes[0], ppo_file, title="PPO: Averaged Rewards Over Time")

    # Visualize VAE-PPO data
    visualize_2graphs(axes[1], vae_ppo_file, title="VAE-PPO: Averaged Rewards Over Time")

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()