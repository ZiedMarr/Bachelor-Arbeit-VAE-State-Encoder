import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Create the environment
env = gym.make("CartPole-v1")

# Initialize the PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
timesteps = 10000  # Number of training steps
model.learn(total_timesteps=timesteps)

# Save the model
model.save("PPO_cartpole_trained/ppo_cartpole_0")

env.close()