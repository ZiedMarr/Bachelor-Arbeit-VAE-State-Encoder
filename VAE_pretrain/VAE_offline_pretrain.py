from VAE import VAE
from Data_Collection.gym_data_collection import load_data
import torch
import numpy as np
import os

#from VAE_PPO_train.model_batch_train import vae_model_path

#from VAE_pretrain.VAE_PPO_pretrain import vae_save_path
#from VAE_pretrain.VAE_random_pretrain import vae_save_path

# Define base paths
base_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the current script

def offline_pretrain(vae_save_path, data_path, vae_model_path) :
    #vae_save_path = os.path.join(base_dir, "pretrained_vae", "5_in_2_out", "vae_offline_expert")
    #data_path = os.path.join(base_dir, "..", "Data_Collection", "collected data", "cartpole_data_expert.npz")

    #Hyperparameters
    # define training frequency :
    train_frequency = 5
    n = 5  # Number of input states
    m = 2  # Number of next states



    # Define the VAE
    vae = VAE(input_dim=20, latent_dim=2, output_dim=8)  # Example dimensions
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    if not(vae_model_path is None) :
        vae.load_state_dict(torch.load(vae_model_path))

    #Load Data :
    episodes , _ , _ = load_data(path=data_path)

    # Train the VAE on every episode
    # initialize training frequency :

    for episode in episodes:
        # initialize the first element of sliding window
        inp_obs_1_index = 0

        # train on the current episode :
        while inp_obs_1_index + n + m <= len(episode):
            # Extract the n current states
            stacked_obs = np.concatenate(list(episode)[inp_obs_1_index:inp_obs_1_index + n], axis=-1)

            # Extract the m next states starting after the n-th state
            stacked_next_obs = np.concatenate(list(episode)[inp_obs_1_index + n:inp_obs_1_index + n + m],
                                              axis=-1)

            # Convert to tensors
            input_tensor = torch.tensor(stacked_obs, dtype=torch.float32).unsqueeze(0)
            target_tensor = torch.tensor(stacked_next_obs, dtype=torch.float32).unsqueeze(0)

            # Forward pass through the VAE
            predicted_next_states, mu, log_var, _ = vae(input_tensor)

            # Compute the VAE loss
            reconstruction_loss = torch.nn.MSELoss()(predicted_next_states, target_tensor)
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = reconstruction_loss + kl_loss

            # Backpropagation and optimization
            vae_optimizer.zero_grad()
            loss.backward()
            vae_optimizer.step()

            inp_obs_1_index += train_frequency

            # self.train_count += 1
            # if self.verbose:
            #    print(f"VAE Training Loss (Iteration {self.train_count}): {loss.item():.4f}")

    #####################################


    # Save the trained VAE model
    torch.save(vae.state_dict(), vae_save_path)
    print(f"Trained VAE model saved to {vae_save_path}")


if __name__ == "__main__":

    offline_pretrain(vae_model_path= None ,vae_save_path="./pretrained_vae/5_in_2_out/explore/vae_explore_0", data_path="../Data_Collection/collected_data/cartpole_ppo_data_0.npz")

    #vae_path
    vae_model_path = "./pretrained_vae/5_in_2_out/explore/vae_explore_0"
    # Directory containing your data files
    data_dir = '../Data_Collection/collected_data'

    # List all files in the directory
    file_list = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.npz')]
    file_list = file_list[1:]
    i=1
    for file in file_list :
        offline_pretrain(vae_model_path= f"./pretrained_vae/5_in_2_out/explore/vae_explore_{i-1}" ,vae_save_path=f"./pretrained_vae/5_in_2_out/explore/vae_explore_{i}",
                         data_path= file)
        i = i+ 1