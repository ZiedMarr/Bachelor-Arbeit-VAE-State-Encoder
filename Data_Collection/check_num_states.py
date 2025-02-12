import numpy as np


def check_num_states(data_path):
    """
    Loads the .npz file and prints the number of states stored in it.

    Parameters:
    - data_path (str): Path to the collected_data file.

    Returns:
    - int: Number of states in the dataset.
    """
    # Load the .npz file
    data = np.load(data_path)

    # Extract observations
    observations = data['observations']

    # Print and return the number of states
    num_states = observations.shape[0]
    print(f"Number of states in the dataset: {num_states}")

    return num_states

if __name__ == "__main__":
    # Example usage
    data_file = "collected_data/mixed_pol_rand_env/mixed_pol_rand_env.npz"  # Replace with the actual file path
    num_states = check_num_states(data_file)
    print(num_states)

