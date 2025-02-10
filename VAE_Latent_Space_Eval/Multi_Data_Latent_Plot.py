import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Union, List
from Data_Collection.gym_data_collection import load_data
from VAE import VAE
import os
from sklearn.decomposition import PCA

from configs import config

# get base_dir path
base_dir = os.path.dirname(os.path.abspath(__file__))

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
    print(f"Total chunks created: {len(chunks)}")  # Debugging

    return np.array(chunks)


def plot_latent_space(
        data_paths: Union[str, List[str]],
        vae: VAE,
        num_input_states: int = config.INPUT_STATE_SIZE,
        save_path: str = None, show=False, reduction=(config.LATENT_DIM > 2)
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

    if reduction and latent_space.shape[1] > 2:
        pca = PCA(n_components=2)
        latent_space = pca.fit_transform(latent_space)
        print("Applied PCA for dimensionality reduction.")

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
        '''
        # Save as interactive HTML
        html_path = save_path.replace('.png', '.html')
        mpld3.save_html(fig, html_path)
        '''

    if show:
        plt.show()
    plt.close()
def call_latent(vae_name,data_dir=os.path.join(base_dir, "..", "Data_Collection", "collected_data", "explore_rand_env"),show =False):

    # model path
    model_path = os.path.join(base_dir, "..", "VAE_pretrain", "pretrained_vae", config.VAE_Version, f"{config.INPUT_STATE_SIZE}_{config.OUTPUT_STATE_SIZE}", f"KL-D_{config.BETA_KL_DIV}", vae_name)
    #VAE_pretrain/pretrained_vae/5_in_2_out/vae_explore_17
    vae = VAE(input_dim=config.INPUT_DIMENSION, latent_dim=config.LATENT_DIM, output_dim=config.OUTPUT_DIMENSION)
    vae.load_state_dict(torch.load(model_path))
    vae_name = os.path.basename(model_path)  # Get the last element
    #define save path
    image_folder = os.path.join(base_dir,'Latent_Plots', config.VAE_Version, f'KL-D_{config.BETA_KL_DIV}', f'{config.INPUT_STATE_SIZE}_{config.OUTPUT_STATE_SIZE}')
    os.makedirs(image_folder, exist_ok=True)


    #Data
    #data1 = os.path.join(base_dir, "..", "Data_Collection", "collected_data", "cartpole_data_random_10.npz")
    #data2 = os.path.join(base_dir, "..", "Data_Collection", "collected_data", "cartpole_data_expert.npz")
    #data_rand = os.path.join(base_dir, "..", "Data_Collection", "collected_data", "1000_rand_Eval" , "random_1000_20250130_122312.npz")
    # Directory containing your data files


    # List all files in the directory
    file_list = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.npz')]


    plot_latent_space(
         data_paths=file_list,
         vae=vae,
         num_input_states=config.INPUT_STATE_SIZE,
         save_path=os.path.join(image_folder,f'{vae_name}.png')
     )
    # Save the array to a text file
    np.savetxt(os.path.join(image_folder,f'{vae_name}.txt'),  file_list, fmt="%s", delimiter=",")  # Save as CSV format



if __name__ == "__main__" :


    # model path
    model_path = os.path.join(base_dir, "..", "VAE_pretrain", "pretrained_vae", config.VAE_Version, f"{config.INPUT_STATE_SIZE}_{config.OUTPUT_STATE_SIZE}", f"KL-D_{config.BETA_KL_DIV}", "vae_rand_100k")
    #VAE_pretrain/pretrained_vae/5_in_2_out/vae_explore_17
    vae = VAE(input_dim=config.INPUT_DIMENSION, latent_dim=config.LATENT_DIM, output_dim=config.OUTPUT_DIMENSION)
    vae.load_state_dict(torch.load(model_path))
    vae_name = os.path.basename(model_path)  # Get the last element
    #define save path
    image_folder = f'Latent_Plots/{config.VAE_Version}/KL-D_{config.BETA_KL_DIV}/{config.INPUT_STATE_SIZE}_{config.OUTPUT_STATE_SIZE}'
    os.makedirs(image_folder, exist_ok=True)


    #Data
    #data1 = os.path.join(base_dir, "..", "Data_Collection", "collected_data", "cartpole_data_random_10.npz")
    #data2 = os.path.join(base_dir, "..", "Data_Collection", "collected_data", "cartpole_data_expert.npz")
    #data_rand = os.path.join(base_dir, "..", "Data_Collection", "collected_data", "1000_rand_Eval" , "random_1000_20250130_122312.npz")
    # Directory containing your data files
    data_dir = '../Data_Collection/collected_data/explore_rand_env'

    # List all files in the directory
    file_list = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.npz')]


    plot_latent_space(
         data_paths=file_list,
         vae=vae,
         num_input_states=config.INPUT_STATE_SIZE,
         save_path=os.path.join(image_folder,f'{vae_name}.png')
        ,
        show=True
     )
    # Save the array to a text file
    np.savetxt(os.path.join(image_folder,f'{vae_name}.txt'),  file_list, fmt="%s", delimiter=",")  # Save as CSV format