import numpy as np
import matplotlib.pyplot as plt
from Data_Collection.gym_data_collection import load_data
from typing import List, Union
import os

#from VAE_Latent_Space_Eval.Latent_Space_Plot import data_path


def visualize_observation_distribution(
    data_paths: Union[str, List[str]],
    observation_index: int = 0,
    save_path: str = None , filter_1_episode=True, show=False
):
    """
    Visualizes the distribution of data points for a specific observation.
    Parameters:
        - data_paths: Single .npz file path or list of .npz file paths
        - observation_index: Index of the observation to visualize (0 to 3 for CartPole)
        - save_path: Optional path to save the plot instead of showing it
    """
    # Ensure data_paths is a list
    if isinstance(data_paths, str):
        data_paths = [data_paths]

    # Collect observations from all files
    all_observations = []

    for data_path in data_paths:
        # Load data from each file
        reconstructed_episodes, _, _ = load_data(data_path)


        if filter_1_episode :
            # Filter out episodes of length 1
            filtered_episodes = [ep for ep in reconstructed_episodes if len(ep) > 1]

            if not filtered_episodes:
                print(f"Warning: All episodes were length 1 in {data_path}. No data to visualize.")
                continue
            file_observations = np.concatenate(filtered_episodes, axis=0)
        else :
            # Flatten all episodes into a single array of observations
            file_observations = np.concatenate(reconstructed_episodes, axis=0)

        # Extract the specific observation (column)
        all_observations.append(file_observations[:, observation_index])

    # Combine observations from all files
    combined_observations = np.concatenate(all_observations)

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    plt.hist(combined_observations, bins=50, alpha=0.7, label='Histogram')
    plt.title(f'Distribution of Observation {observation_index} (Multiple Files)')
    plt.xlabel(f'Observation {observation_index}')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.legend()

    # Either save or show the plot
    if save_path:
        plt.savefig(save_path)
        if show :
            plt.show()
        plt.close()
    else:
        if show :
            plt.show()



if __name__ == "__main__":
    # Example usage
    filter_1_episodes = True

    '''
    directory = "./collected_data/eval/rand_pol_rand_env/random_100_20250220_162141.npz"
    data_paths = []
    # Iterate through the directory
    for file_name in os.listdir(directory):
            # Construct the full path to the file
            full_path = os.path.join(directory, file_name)

            # Check if it's a file (and not a subdirectory)
            if os.path.isfile(full_path):
                data_paths.append(full_path)
    '''
    data_path = "collected_data/eval/merged1/merged1.npz"

    data_name = os.path.basename(data_path)
    name_without_extension, _ = os.path.splitext(data_name)

    if filter_1_episodes :
        save_dir = save_path = os.path.join("./Data_distribution", "merged",f"merged1") #f"{name_without_extension}_filtered"
    else :
        save_dir = save_path=os.path.join("./Data_distribution","merged", "merged1") #name_without_extension
    # Create the directory if it doesn’t exist
    os.makedirs(save_dir, exist_ok=True)

    for i in range(24) :
        visualize_observation_distribution(data_paths=data_path, observation_index=i, save_path=os.path.join(save_dir,f"data_explore_{i}"))
    #visualize_observation_distribution(data_paths="./collected_data/cartpole_expert_60.npz", observation_index=3,
    #                               save_path="./Data_distribution/expert_60/data_3")
