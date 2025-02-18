import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import os

# Define the base directory (directory of the current script)
base_dir = os.path.dirname(os.path.abspath(__file__))


# Load the environment and model
env = gym.make("CartPole-v1", render_mode="human")
model_path = os.path.join(base_dir, "PPO_trained", "rl_model_2048_steps")
model = PPO.load(model_path)

# Evaluate the model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Test the model
obs, _ = env.reset()
for _ in range(500):  # Max steps in an episode
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    env.render()
    if done:
        obs, _ = env.reset()

env.close()
