import gymnasium as gym
import torch
from stable_baselines3 import PPO
from Wrapped_environment import VAEWrapperWithHistory
from Callbacks.VAE_callback_PPO import VAETrainingCallback
from Callbacks.SaveModelCallback import SaveModelCallback
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from VAE import VAE
import os
from datetime import datetime


#get timestamp :
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#define number of total training time steps :
total_timesteps = 5000
#define vae model path
vae_model_path = "../VAE_pretrain/pretrained_vae/vae_offline_expert"

# Extract the VAE model name from its path
vae_model_name = os.path.basename(vae_model_path)  # Get the last element of the path

# Generate a log directory name
log_dir = f"logs_{total_timesteps}_{vae_model_name}_{timestamp}"


# Define the VAE   : TODO : replace with load VAE
vae = VAE(input_dim=20, latent_dim=2, output_dim=8)  # Example dimensions
vae_optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
#load vae wights
vae.load_state_dict(torch.load(vae_model_path))


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
wrapped_env = Monitor(wrapped_env)

##############test#######################
#print("Wrapped environment observation space:", wrapped_env.observation_space)
##############test#######################

# Define PPO model
ppo_model = PPO("MlpPolicy", wrapped_env, verbose=1)
# Set up TensorBoard logger
ppo_model.set_logger(configure(f"{log_dir}/tensorboard_logs", ["tensorboard"]))

# Initialize Callbacks
vae_callback = VAETrainingCallback(
    vae=vae, optimizer=vae_optimizer, train_frequency=2, n=n, m=m, verbose=1, original_obs_shape=4
)
#initialize eval_callback
eval_callback = EvalCallback(
    wrapped_env,
    best_model_save_path=f"{log_dir}/best_model/",
    log_path=f"{log_dir}/eval/",
    eval_freq=2048,  # Evaluate every 2048 timesteps
    deterministic=True,
    render=False
)



# Initialize checkpoint callback for saving models every 500 timesteps
#checkpoint_callback = CheckpointCallback(save_freq=1, save_path="../trained_models/")
#event_callback = EveryNTimesteps(n_steps=2048, callback=checkpoint_callback)


# Train PPO with VAE training in the callback
ppo_model.learn(total_timesteps=total_timesteps, callback=[vae_callback, eval_callback])


print("Training and periodic saving completed!")

# Save the trained model
#ppo_model.save("trained_models/ppo_model")
#print("Model saved successfully!")

