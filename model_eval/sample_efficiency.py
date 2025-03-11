import numpy as np
import os

from pygame.examples.midi import output_main


def calculate_timesteps_to_reward(base_log_dir, target_reward, output_path=None):
    """
    Calculate the average number of timesteps needed to reach a target reward.

    Args:
        base_log_dir: Directory containing evaluation logs
        target_reward: The reward threshold to measure against
        output_path: Where to save the results (if None, results are only printed)

    Returns:
        mean_timesteps: Average timesteps to reach target_reward
        std_timesteps: Standard deviation of timesteps to reach target_reward
        num_reached: Number of runs that reached the target
        total_runs: Total number of runs analyzed
    """
    # Find all evaluation directories
    log_dirs = []
    for root, dirs, files in os.walk(base_log_dir):
        if "eval" in dirs:
            log_dirs.append(os.path.join(root, "eval"))

    if not log_dirs:
        raise ValueError(f"No evaluation directories found in {base_log_dir}")

    # Initialize storage for timesteps to reach target
    timesteps_to_target = []
    all_timesteps = None

    print(f"Found {len(log_dirs)} evaluation directories")

    # Process each evaluation log
    for log_dir in log_dirs:
        eval_file = os.path.join(log_dir, "evaluations.npz")
        if not os.path.exists(eval_file):
            print(f"Warning: {eval_file} not found, skipping")
            continue

        data = np.load(eval_file)

        # Store timesteps (should be same for all logs)
        if all_timesteps is None:
            all_timesteps = data["timesteps"]

        # Get rewards data - average across episodes if necessary
        if "results" in data:
            # Shape: (n_evals, n_episodes)
            rewards = data["results"].mean(axis=1)  # Average across episodes
        elif "mean_rewards" in data:
            # Already averaged
            rewards = data["mean_rewards"]
        else:
            print(f"Warning: No rewards data found in {eval_file}, skipping")
            continue

        # Only if max reward exceeds the target
        if max(rewards) >= target_reward:
            # Find timestep where reward first exceeds target
            for i in range(len(rewards) - 1):
                if rewards[i] < target_reward and rewards[i + 1] >= target_reward:
                    # Linear interpolation between these points
                    t1, t2 = all_timesteps[i], all_timesteps[i + 1]
                    r1, r2 = rewards[i], rewards[i + 1]

                    # Calculate interpolated timestep
                    timestep = t1 + (target_reward - r1) * (t2 - t1) / (r2 - r1)
                    timesteps_to_target.append(timestep)
                    break
            else:
                # If target was reached in first eval, use that timestep
                if rewards[0] >= target_reward:
                    timesteps_to_target.append(all_timesteps[0])

    # Calculate statistics
    if not timesteps_to_target:
        print(f"Warning: No runs reached the target reward of {target_reward}")
        mean_timesteps = float('inf')
        std_timesteps = float('nan')
    else:
        mean_timesteps = np.mean(timesteps_to_target)
        std_timesteps = np.std(timesteps_to_target)

        # Format results as text
        results_text = f"Target Reward: {target_reward}\n"
        results_text += f"Runs that reached target: {len(timesteps_to_target)}/{len(log_dirs)}\n"
        results_text += f"Average timesteps to reach target: {mean_timesteps:.2f}\n"
        results_text += f"Standard deviation: {std_timesteps:.2f}\n"

        # Print results
        print(results_text)

    # Save results if output path is provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        with open(output_path, 'w') as f:
            f.write(results_text)
            # Add individual timesteps as additional information
            if timesteps_to_target:
                f.write("\nIndividual timesteps to reach target:\n")
                for i, ts in enumerate(timesteps_to_target):
                    f.write(f"Run {i + 1}: {ts:.2f}\n")

        print(f"Results saved to: {output_path}")

    return mean_timesteps, std_timesteps, len(timesteps_to_target), len(log_dirs)


def analyze_multiple_targets(base_log_dir, target_rewards, output_dir=None):
    """
    Analyze multiple target rewards and compile results.

    Args:
        base_log_dir: Directory containing evaluation logs
        target_rewards: List of reward thresholds to measure against
        output_dir: Directory to save results
    """
    results = []

    for target in target_rewards:
        output_path = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"target_{target:.2f}.npz")

        print(f"\nAnalyzing target reward: {target}")
        mean, std, reached, total = calculate_timesteps_to_reward(
            base_log_dir, target, output_path=output_path
        )

        results.append({
            'target': target,
            'mean_timesteps': mean,
            'std_timesteps': std,
            'reached': reached,
            'total': total
        })

    # Compile summary
    if output_dir:
        summary_path = os.path.join(output_dir, "summary.csv")
        with open(summary_path, 'w') as f:
            f.write("target_reward,mean_timesteps,std_timesteps,runs_reached,total_runs\n")
            for r in results:
                f.write(f"{r['target']},{r['mean_timesteps']},{r['std_timesteps']},{r['reached']},{r['total']}\n")
        print(f"\nSummary saved to: {summary_path}")

    return results


# Example usage
if __name__ == "__main__":
    # Define base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # For a single target reward
    vae_ppo_log_dir = "../VAE_PPO_train/logs/batch_100k_VAE_Version_1.08_vae_exp_0.3noise_10ep_3"

    output_path = os.path.join(".", "sample_efficiency", "4_2", "v1.08")

    mean_timesteps, std_timesteps, runs_reached, total_runs = calculate_timesteps_to_reward(
        vae_ppo_log_dir,
        target_reward=400,
        output_path= output_path
    )
    '''
    # For multiple target rewards
    target_rewards = [100, 200, 300, 400, 500]
    results = analyze_multiple_targets(
        vae_ppo_log_dir,
        target_rewards=target_rewards,
        output_dir=os.path.join(base_dir, "results")
    )
    '''