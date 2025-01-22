from VAE import VAE
from Data_Collection.gym_data_collection import load_data
import torch
import numpy as np


#Define Paths :
vae_save_path = "pretrained_vae/5_in_2_out/vae_offline_expert"  # Define saving path for pretrained vae :
data_path = "../Data_Collection/collected data/cartpole_data_expert.npz" # Define Data Path #

#Hyperparameters
# define training frequency :
train_frequency = 5
n = 5  # Number of input states
m = 2  # Number of next states



# Define the VAE
vae = VAE(input_dim=20, latent_dim=2, output_dim=8)  # Example dimensions
vae_optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

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
