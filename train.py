import gymnasium as gym
import torch
from stable_baselines3 import PPO
import torch.optim as optim
from Wrapped_environment import VAEWrapperWithHistory
from torch.utils.tensorboard import SummaryWriter
from  VAE_callback_PPO import VAETrainingCallback
from VAE import VAE


# Define the VAE   : TODO : replace with load VAE
vae = VAE(input_dim=20, latent_dim=2, output_dim=8)  # Example dimensions
vae_optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)


# Create the wrapped environment
n = 5  # Number of input states
m = 2  # Number of next states

# Define environment :
# set a random seed
seed = 42
env = gym.make("CartPole-v1")
env.reset(seed=seed)
env.observation_space.seed(seed)
wrapped_env = VAEWrapperWithHistory(env, vae, n=n, m=m, vae_optimizer=vae_optimizer)

# Define PPO model
ppo_model = PPO("MlpPolicy", wrapped_env, verbose=1)

# Initialize the callback
vae_callback = VAETrainingCallback(
    vae=vae, optimizer=vae_optimizer, train_frequency=2, n=n, m=m, verbose=1, original_obs_shape=4
)

# Train PPO with VAE training in the callback
ppo_model.learn(total_timesteps=1000, callback=vae_callback)

# Save the trained model
ppo_model.save("trained_models/ppo_model")
print("Model saved successfully!")

