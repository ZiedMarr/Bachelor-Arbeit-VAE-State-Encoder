import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Union, List
from Data_Collection.gym_data_collection import load_data
from VAE import VAE
import os


def split_observations(observations, num_input_states):
    """
    Split observations into chunks of the specified number of input states.

    Parameters:
    - observations: List of episode observations
    - num_input_states: Number of states to stack

    Returns:
    - NumPy array of flattened observation chunks
    """
    chunks = []
    for episode in observations:
        for i in range(0, len(episode) - num_input_states + 1):
            chunk = episode[i:i + num_input_states].flatten()
            chunks.append(chunk)
    return np.array(chunks)


def plot_latent_space(
        data_paths: Union[str, List[str]],
        vae: VAE,
        num_input_states: int = 5,
        save_path: str = None
):
    """
    Visualizes the latent space representation of observations from multiple files.
    """
    # Ensure data_paths is a list
    if isinstance(data_paths, str):
        data_paths = [data_paths]

    # Collect latent representations
    all_latent_means = []
    all_split_obs = []

    for data_path in data_paths:
        # Load data from each file
        observations, _, _ = load_data(path=data_path)

        # Split observations to match VAE input size
        split_obs = split_observations(observations, num_input_states)

        # Get latent representations
        vae.eval()
        with torch.no_grad():
            inputs_tensor = torch.tensor(split_obs, dtype=torch.float32)
            latents = vae.encode(inputs_tensor).cpu().numpy()

        all_latent_means.append(latents)
        all_split_obs.append(split_obs)

    # Combine latent representations
    latent_space = np.concatenate(all_latent_means)
    split_obs = np.concatenate(all_split_obs)

    # Create a single figure with annotations
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(latent_space[:, 0], latent_space[:, 1], alpha=0.5, picker=True)
    plt.title('VAE Latent Space Visualization (Multiple Data Files)')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')

    # Annotate points
    annotations = [ax.annotate(str(split_obs[i]),
                               (float(latent_space[i, 0]), float(latent_space[i, 1])),
                               textcoords="offset points",
                               xytext=(0, 10),
                               ha='center') for i in range(len(latent_space))]
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

    # Save or show plot
    if save_path:
        plt.savefig(save_path)
        plt.show()
        plt.close()
    else:
        plt.show()

if __name__ == "__main__" :

    # get base_dir path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # model path
    model_path = os.path.join(base_dir, "..", "VAE_pretrain", "pretrained_vae", "5_in_2_out",
                              "vae_explore_17")
    #VAE_pretrain/pretrained_vae/5_in_2_out/vae_explore_17
    vae = VAE(input_dim=20, latent_dim=2, output_dim=8)
    vae.load_state_dict(torch.load(model_path))

    #Data
    #data1 = os.path.join(base_dir, "..", "Data_Collection", "collected data", "cartpole_data_random_10.npz")
    #data2 = os.path.join(base_dir, "..", "Data_Collection", "collected data", "cartpole_data_expert.npz")
    # Directory containing your data files
    data_dir = '../Data_Collection/collected_data'

    # List all files in the directory
    file_list = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.npz')]

    data_files = file_list[:12]
    plot_latent_space(
         data_paths=data_files,
         vae=vae,
         num_input_states=5,
         save_path='./Latent_Plots/latent_space_plot_test_data1.png'
     )