import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Union, List
from Data_Collection.gym_data_collection import load_data
from VAE import VAE
import os
from sklearn.decomposition import PCA
import mpld3
import config



# get base_dir path
base_dir = os.path.dirname(os.path.abspath(__file__))


def split_observations(observations, num_input_states):
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
        save_path: str = None, show=True, reduction=False
):
    if isinstance(data_paths, str):
        data_paths = [data_paths]

    all_latent_means = []
    all_split_obs = []

    for data_path in data_paths:
        observations, _, _ = load_data(path=data_path)
        split_obs = split_observations(observations, num_input_states)

        vae.eval()
        with torch.no_grad():
            inputs_tensor = torch.tensor(split_obs, dtype=torch.float32)
            latents = vae.encode(inputs_tensor).cpu().numpy()

        all_latent_means.append(latents)
        all_split_obs.append(split_obs)

    latent_space = np.concatenate(all_latent_means)
    split_obs = np.concatenate(all_split_obs)

    if reduction and latent_space.shape[1] > 2:
        pca = PCA(n_components=2)
        latent_space = pca.fit_transform(latent_space)
        print("Applied PCA for dimensionality reduction.")

    num_observations = split_obs.shape[1] // num_input_states

    # Plot for each observation
    for obs_idx in range(num_observations):
        avg_obs_values = np.mean(split_obs[:, obs_idx::num_observations], axis=1)

        fig, ax = plt.subplots(figsize=(10, 8))
        sc = ax.scatter(latent_space[:, 0], latent_space[:, 1], c=avg_obs_values, cmap='viridis', alpha=0.5,
                        picker=True)
        plt.colorbar(sc, ax=ax, label=f'Average Value of Observation {obs_idx + 1}')
        plt.title(f'Latent Space Visualization - Observation {obs_idx + 1}')
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')

        if save_path:
            png_path = save_path.replace('.png', f'_obs_{obs_idx + 1}.png')
            html_path = png_path.replace('.png', '.html')
            plt.savefig(png_path)
            mpld3.save_html(fig, html_path)
            print(f"Saved plot and interactive HTML for Observation {obs_idx + 1}")

        if show:
            plt.show()
        plt.close()


def call_latent_colored(vae_name, data_dir=os.path.join(base_dir, "..", "Data_Collection", "collected_data", "explore_rand_env"), show=False, reduction=True):
    model_path = os.path.join(base_dir, "..", "VAE_pretrain", "pretrained_vae", config.VAE_Version,
                              f"{config.INPUT_STATE_SIZE}_{config.OUTPUT_STATE_SIZE}", f"KL-D_{config.BETA_KL_DIV}", vae_name)
    vae = VAE(input_dim=config.INPUT_DIMENSION, latent_dim=config.LATENT_DIM, output_dim=config.OUTPUT_DIMENSION)
    vae.load_state_dict(torch.load(model_path))
    vae_name = os.path.basename(model_path)
    image_folder = f'Latent_Plots/{config.VAE_Version}/KL-D_{config.BETA_KL_DIV}/{config.INPUT_STATE_SIZE}_{config.OUTPUT_STATE_SIZE}'
    os.makedirs(image_folder, exist_ok=True)

    file_list = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.npz')]

    plot_latent_space(
        data_paths=file_list,
        vae=vae,
        num_input_states=config.INPUT_STATE_SIZE,
        save_path=os.path.join(image_folder, f'{vae_name}.png'),
        reduction=reduction
    )
    np.savetxt(os.path.join(image_folder, f'{vae_name}.txt'), file_list, fmt="%s", delimiter=",")


if __name__ == "__main__":
    call_latent_colored("vae_rand_100k", reduction=True)
