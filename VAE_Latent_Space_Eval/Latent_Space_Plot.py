import torch
import numpy as np
import matplotlib.pyplot as plt
from VAE import VAE  # Import the VAE class from VAE.py
from Data_Collection.gym_data_collection import load_data  # Import the load_data function
import os

#get base_dir path
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load the pretrained VAE model
model_path = os.path.join(base_dir, "..", "VAE_PPO_train", "trained_vae","batch2", "20000_vae_15000_vae_offline_expert_20250114_165817_20250123_141147")
num_input_states = 5  # Number of input states
state_dim = 4  # Each state has 4 elements
input_dim = state_dim * num_input_states  # Total dimension of the stacked input states
latent_dim = 2  # Define the correct latent dimension
output_dim = 8  # Define the correct output dimension

# Initialize the VAE model
vae = VAE(input_dim, latent_dim, output_dim)
vae.load_state_dict(torch.load(model_path))
vae.eval()  # Set the model to evaluation mode

# Load the collected_data
data_path = os.path.join(base_dir, "..", "Data_Collection", "collected_data", "cartpole_data_random_10.npz")

observations, episode_starts, episode_lengths = load_data(path=data_path)

# Function to split observations into chunks of the specified number of input states
def split_observations(observations, num_input_states):
    chunks = []
    for episode in observations:
        for i in range(0, len(episode) - num_input_states + 1):
            chunk = episode[i:i + num_input_states].flatten()  # Flatten the states
            chunks.append(chunk)
    return np.array(chunks)

# Split observations to match the input size of the VAE
split_obs = split_observations(observations, num_input_states)

# Extract latent representations from the VAE model
def get_latent_representations(vae, inputs):
    vae.eval()
    with torch.no_grad():
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32)  # Convert inputs to a PyTorch tensor
        latents = vae.encode(inputs_tensor).cpu().numpy()  # Encode the inputs and convert to numpy array
    return latents

# Get latent representations
latents = get_latent_representations(vae, split_obs)

# Plotting
fig, ax = plt.subplots()
sc = ax.scatter(latents[:, 0], latents[:, 1], alpha=0.5, picker=True)
plt.title('Latent Space Visualization')
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.colorbar(sc)

# Annotate a few points
annotations = [ax.annotate(str(split_obs[i]), (latents[i, 0], latents[i, 1]),
                textcoords="offset points", xytext=(0,10), ha='center') for i in range(len(latents))]
for annot in annotations:
    annot.set_visible(False)

# Function to update annotation
def update_annotation(ind):
    for annot in annotations:
        annot.set_visible(False)
    if len(ind['ind']) > 0:
        annotations[ind['ind'][0]].set_visible(True)

# Event handler for pick event
def on_pick(event):
    ind = event.ind
    if len(ind) > 0:
        update_annotation({'ind': ind})
        fig.canvas.draw_idle()

# Connect pick event to handler
fig.canvas.mpl_connect('pick_event', on_pick)

plt.show()