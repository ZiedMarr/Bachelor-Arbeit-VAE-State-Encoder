import gymnasium as gym
import torch.optim as optim
from VAE import VAE
from Wrapped_environment import VAEWrapperWithHistory
from stable_baselines3 import PPO  # Import stable-baselines3 for pre-trained policy
from torch.utils.tensorboard import SummaryWriter
import torch


# Define saving path for pretrained vae :
vae_save_path = "/pretrained_vae/vae_1"

# Define the VAE
vae = VAE(input_dim=20, latent_dim=2, output_dim=8)  # Example dimensions
vae_optimizer = optim.Adam(vae.parameters(), lr=1e-3)

# Create the wrapped environment
n = 5  # Number of input states
m = 2  # Number of next states
env = gym.make("CartPole-v1")
wrapped_env = VAEWrapperWithHistory(env, vae, n=n, m=m, vae_optimizer=vae_optimizer)

# Load a pre-trained policy
policy_path = "../PPO_cartpole/PPO_cartpole_trained/ppo_cartpole_0.zip"  # Replace with the actual path to your policy
pretrained_policy = PPO.load(policy_path)

# Training parameters
num_episodes = 30
vae_train_frequency = 3  # Train VAE every X steps

# Tensorboard initialization
writer = SummaryWriter()

# Policy-Driven Training Loop
for episode in range(num_episodes):
    obs, info = wrapped_env.reset()
    done = False
    total_steps = 0
    episode_loss = 0.0

    while not done:
        # Use the pre-trained policy to select an action
        action, _ = pretrained_policy.predict(obs, deterministic=True)

        # Step in the environment
        obs, reward, done, truncated, info = wrapped_env.step(action)
        total_steps += 1

        # Train the VAE periodically
        if total_steps % vae_train_frequency == 0:
            loss = wrapped_env.train_vae()  # Train the VAE
            episode_loss += loss

    # Log episode loss
    writer.add_scalar('try/test', episode_loss, episode + 1)

    # Print loss
    print(f"Episode {episode + 1}, Total Steps: {total_steps}, VAE Loss: {episode_loss:.4f}")


# Save the trained VAE model
torch.save(vae.state_dict(), vae_save_path)
print(f"Trained VAE model saved to {vae_save_path}")


writer.close()
