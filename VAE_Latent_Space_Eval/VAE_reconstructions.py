import os
import numpy as np
import torch
from Data_Collection.gym_data_collection import load_data  # Assuming this is the file with data functions
from VAE import VAE  # Assuming the VAE class is defined in VAE.py
import gymnasium as gym
import torch.optim as optim
import imageio

# get base_dir path
base_dir = os.path.dirname(os.path.abspath(__file__))

def render_cartpole_from_observations(observations, gif_path="./GIF/cartpole_render.gif"):
    """
    Render a CartPole environment using a sequence of observations and save it as a GIF.

    :param observations: A sequence of observations (state vectors).
    :param gif_path: Path to save the generated GIF.
    """
    # Create the CartPole environment in "rgb_array" mode for rendering
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    frames = []

    # Loop through each observation and render the environment
    for obs in observations:
        env.reset()  # Reset the environment for each observation
        env.unwrapped.state = obs  # Manually set the environment's state

        # Render the environment as an RGB image
        frame = env.render()
        frames.append(frame)

    env.close()

    # Save the frames as a GIF
    imageio.mimsave(gif_path, frames, duration=0.1)
    print(f"CartPole rendering saved to {gif_path}")

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

def main(data_path, vae_model_path):

    # File paths
    #data_path = os.path.join(base_dir, "..", "Data_collection", "collected data", "cartpole_data_random_1.npz")
    #vae_model_path = os.path.join(base_dir, "..", "VAE_pretrain", "pretrained_vae", "5_in_2_out", "vae_random_10")

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
    vae_name = last_element = os.path.basename(vae_model_path)

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
    '''
    for i, (inp, true_out, pred_out) in enumerate(zip(selected_inputs, corresponding_outputs, predicted_outputs.numpy())):
        print(f"Example {i+1}")
        print("Input:", inp.reshape(n, -1))  # Reshape to n rows of state dimensions
        print("True Output:", true_out.reshape(m, -1))  # Reshape to m rows of state dimensions
        print("Predicted Output:", pred_out.reshape(m, -1))  # Reshape to m rows of state dimensions
        print("-" * 50)
    '''
    for i, (inp, true_out, pred_out) in enumerate(zip(selected_inputs, corresponding_outputs, predicted_outputs.numpy())):
        folder_path = os.path.join(base_dir,"gif" , f"{vae_name}", f"example_{i}")
        os.makedirs(folder_path, exist_ok=True)

        input_gif_path = os.path.join(folder_path, "input.gif")
        true_output_gif_path = os.path.join(folder_path, "true_output.gif")
        predicted_output_gif_path = os.path.join(folder_path, "predicted_output.gif")

        render_cartpole_from_observations(inp.reshape(n, -1), input_gif_path)
        render_cartpole_from_observations(true_out.reshape(m, -1), true_output_gif_path)
        render_cartpole_from_observations(pred_out.reshape(m, -1), predicted_output_gif_path)

if __name__ == "__main__":
    main(data_path=os.path.join(base_dir, "..", "Data_collection", "collected data", "cartpole_data_random_1.npz") , vae_model_path=os.path.join(base_dir, "..", "VAE_pretrain", "pretrained_vae", "5_in_2_out","explore", "vae_explore_17"))
