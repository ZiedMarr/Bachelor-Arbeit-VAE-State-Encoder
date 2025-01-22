import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data from the .npz file
file_path = "logs/averaged_evaluation_batch1.npz"  # Path to your .npz file
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
