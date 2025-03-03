import numpy as np
import os


def merge_npz_files(file_paths, output_path):
    """
    Merges multiple .npz data files into a single .npz file.

    Parameters:
    - file_paths (list of str): List of paths to the .npz files to be merged1.
    - output_path (str): Path where the merged1 file should be saved.
    """
    all_observations = []
    all_episode_starts = []
    all_episode_lengths = []

    total_length = 0  # To track correct episode starts

    for file_path in file_paths:
        # Load data
        data = np.load(file_path)

        # Extract observations and episode metadata
        observations = data['observations']
        episode_starts = data['episode_starts']
        episode_lengths = data['episode_lengths']

        # Adjust episode start indices to maintain consistency
        adjusted_episode_starts = episode_starts + total_length

        # Append to the main list
        all_observations.append(observations)
        all_episode_starts.append(adjusted_episode_starts)
        all_episode_lengths.append(episode_lengths)

        # Update total length for the next batch
        total_length += observations.shape[0]

    # Merge all data
    merged_observations = np.concatenate(all_observations, axis=0)
    merged_episode_starts = np.concatenate(all_episode_starts, axis=0)
    merged_episode_lengths = np.concatenate(all_episode_lengths, axis=0)

    # Save the merged1 data
    np.savez(output_path,
             observations=merged_observations,
             episode_starts=merged_episode_starts,
             episode_lengths=merged_episode_lengths)

    print(f"Merged {len(file_paths)} files into {output_path}")


if __name__ == "__main__" :
    # Example Usage
    directory = "./collected_data/train/explore_policy_rand_env/ppo_no_noise"
    file_paths = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(".npz")]
    output_file = "./collected_data/train/explore_policy_rand_env/ppo_no_noise/ppo_no_noise_merged.npz"

    merge_npz_files(file_paths, output_file)

