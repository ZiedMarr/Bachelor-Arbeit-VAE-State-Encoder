import numpy as np
import matplotlib.pyplot as plt
from Data_Collection.gym_data_collection import load_data
from typing import List, Union
import os

#from VAE_Latent_Space_Eval.Latent_Space_Plot import data_path


def visualize_observation_distribution(
    data_paths: Union[str, List[str]],
    observation_index: int = 0,
    save_path: str = None
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
        plt.show()
        plt.close()
    else:
        plt.show()



if __name__ == "__main__":
    # Example usage
    '''
    #data_paths = ['./collected data/cartpole_data_random_50.npz', "./collected data/cartpole_data_expert.npz" , "./collected data/cartpole_data_random_10.npz" ] # Replace with your actual file path
    directory = "collected_data"
    data_paths = []
    # Iterate through the directory
    for file_name in os.listdir(directory):
            # Construct the full path to the file
            full_path = os.path.join(directory, file_name)

            # Check if it's a file (and not a subdirectory)
            if os.path.isfile(full_path):
                data_paths.append(full_path)
    '''


   # visualize_observation_distribution(data_paths=data_paths, observation_index=3, save_path="./Data_distribution/explore/data_explore_3")
    visualize_observation_distribution(data_paths="./collected data/cartpole_expert_60.npz", observation_index=3,
                                   save_path="./Data_distribution/expert_60/data_3")
