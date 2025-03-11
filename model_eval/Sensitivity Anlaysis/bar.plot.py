import matplotlib.pyplot as plt
import numpy as np
import os


def create_reward_bar_plot(rewards, x_labels, x_title="Algorithms", y_title="Average Episodic Rewards",
                           title="Average Episodic Rewards", save_path=None):
    """
    Create a bar plot for average episodic rewards.

    Parameters:
    -----------
    rewards : list or array
        List of 3 reward values to plot
    x_labels : list or array
        List of 3 labels for the x-axis
    x_title : str, optional
        Title for the x-axis
    y_title : str, optional
        Title for the y-axis
    title : str, optional
        Title of the plot
    save_path : str, optional
        If provided, save the figure to this path

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Ensure we have exactly 3 values and labels
    if len(rewards) != 3 or len(x_labels) != 3:
        raise ValueError("Please provide exactly 3 reward values and 3 x-axis labels")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar positions
    x_pos = np.arange(len(rewards))

    # Create bars
    bars = ax.bar(x_pos, rewards, width=0.6, edgecolor='black', linewidth=1.2)

    # Add colors to bars
    colors = ['#3274A1', '#E1812C', '#3A923A']
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    # Customize plot
    ax.set_xlabel(x_title, fontsize=12, fontweight='bold')
    ax.set_ylabel(y_title, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=10)

    # Add grid on y-axis
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add values on top of bars
    for i, v in enumerate(rewards):
        ax.text(i, v + max(rewards) * 0.02, f'{v:.2f}', ha='center', fontsize=10)

    # Adjust layout
    plt.tight_layout()

    # Save figure if path is provided
    if save_path:
        # Ensure the directory exists
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {os.path.abspath(save_path)}")

    return fig, ax


# Example usage
if __name__ == "__main__":
    # Input your data here
    rewards = [477.94, 470.46 , 420.09 ]  # Your three reward values
    x_labels = ["0.0001", "0.00097", "0.01"]  # Your x-axis labels

    # Set your axis titles here
    x_axis_title = "KL-Divergence Coefficient"  # Change this to describe what your x-axis represents
    y_axis_title = "Average Episodic Rewards"

    # Set your plot title
    plot_title = "Performance Comparison of 10 VAE-PPO agents over 10 episodes"

    # Set your save path here
    save_location = "./KL-D.png"  # Change this to your desired location

    # Create and save the plot
    fig, ax = create_reward_bar_plot(
        rewards=rewards,
        x_labels=x_labels,
        x_title=x_axis_title,
        y_title=y_axis_title,
        title=plot_title,
        save_path=save_location
    )

    # Show the plot (optional)
    plt.show()