import os
import numpy as np
import torch
from Data_Collection.gym_data_collection import load_data  # Assuming this is the file with data functions
from VAE import VAE  # Assuming the VAE class is defined in VAE.py
import torch.optim as optim

def stack_data_per_episode(episodes, n, m):
    """
    Prepare stacked inputs and outputs for the VAE while respecting episode boundaries.
    :param episodes: List of reconstructed episodes.
    :param n: Number of input states to stack.
    :param m: Number of next states to predict.
    :return: Stacked inputs and outputs.
    """
    inputs = []
    outputs = []

    for episode in episodes:
        if len(episode) < n + m:
            continue  # Skip episodes that are too short

        for i in range(len(episode) - n - m + 1):
            input_states = episode[i:i + n].flatten()
            output_states = episode[i + n:i + n + m].flatten()
            inputs.append(input_states)
            outputs.append(output_states)

    return np.array(inputs), np.array(outputs)

def main():
    #get base_dir path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # File paths
    data_path = os.path.join(base_dir, "..", "Data_collection", "collected data", "cartpole_data_expert.npz")
    vae_model_path = os.path.join(base_dir, "..", "VAE_pretrain", "pretrained_vae", "5_in_2_out", "vae_offline_0")

    # Load the data
    reconstructed_episodes, _, _ = load_data(data_path)

    # Define VAE input/output stacking parameters
    n = 5  # Number of input states
    m = 2  # Number of output states

    # Prepare data while respecting episode boundaries
    inputs, outputs = stack_data_per_episode(reconstructed_episodes, n, m)

    # Initialize the VAE (adjust dimensions to your setup)
    input_dim = inputs.shape[1]  # Flattened input dimension
    output_dim = outputs.shape[1]  # Flattened output dimension
    latent_dim = 2  # Latent space dimension

    vae = VAE(input_dim=input_dim, latent_dim=latent_dim, output_dim=output_dim)

    # Load pretrained weights
    if os.path.exists(vae_model_path):
        vae.load_state_dict(torch.load(vae_model_path))
        vae.eval()
        print("Loaded pretrained VAE model.")
    else:
        raise FileNotFoundError(f"Pretrained VAE model not found at {vae_model_path}")

    # Select 5 random examples
    indices = np.random.choice(len(inputs), size=5, replace=False)
    selected_inputs = inputs[indices]
    corresponding_outputs = outputs[indices]

    # Convert inputs to tensors
    inputs_tensor = torch.tensor(selected_inputs, dtype=torch.float32)

    # Run the VAE on selected inputs
    with torch.no_grad():
        predicted_outputs, _, _, _ = vae(inputs_tensor)

    # Print input and output in a clear format
    for i, (inp, true_out, pred_out) in enumerate(zip(selected_inputs, corresponding_outputs, predicted_outputs.numpy())):
        print(f"Example {i+1}")
        print("Input:", inp.reshape(n, -1))  # Reshape to n rows of state dimensions
        print("True Output:", true_out.reshape(m, -1))  # Reshape to m rows of state dimensions
        print("Predicted Output:", pred_out.reshape(m, -1))  # Reshape to m rows of state dimensions
        print("-" * 50)

if __name__ == "__main__":
    main()
