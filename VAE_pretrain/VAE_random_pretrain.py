import gymnasium as gym
import torch.optim as optim
from VAE import VAE
from Wrapped_environment import VAEWrapperWithHistory
from torch.utils.tensorboard import SummaryWriter
import torch


# Define saving path for pretrained vae :
vae_save_path = "pretrained_vae/5_in_2_out/vae_random_10"



# Define the VAE
vae = VAE(input_dim=20, latent_dim=2, output_dim=8)  # Example dimensions
vae_optimizer = optim.Adam(vae.parameters(), lr=1e-3)

# Create the wrapped environment
n = 5  # Number of input states
m = 2  # Number of next states
env = gym.make("CartPole-v1")
wrapped_env = VAEWrapperWithHistory(env, vae, n=n, m=m, vae_optimizer=vae_optimizer)
print(wrapped_env.observation_space.shape[0])
# Training parameters
num_episodes = 10
vae_train_frequency = 3  # Train VAE every X steps

#Tensorboard initialization
writer = SummaryWriter()

#Random Training loop
for episode in range(num_episodes):
    obs, info = wrapped_env.reset()
    done = False
    total_steps = 0
    episode_loss = 0.0

    while not done:
        # Take a random action (replace with agent's action if available)
        action = wrapped_env.action_space.sample()
        obs, reward, done, truncated, info = wrapped_env.step(action)
        total_steps += 1

        # Train the VAE periodically
        if total_steps % vae_train_frequency == 0:
            loss = wrapped_env.train_vae()  # Train the VAE
            episode_loss += loss

    #log episode loss
    writer.add_scalar('try/test', episode_loss, episode+1)

    #print loss
    print(f"Episode {episode + 1}, Total Steps: {total_steps}, VAE Loss: {episode_loss:.4f}")

# Save the trained VAE model
torch.save(vae.state_dict(), vae_save_path)
print(f"Trained VAE model saved to {vae_save_path}")


writer.close()