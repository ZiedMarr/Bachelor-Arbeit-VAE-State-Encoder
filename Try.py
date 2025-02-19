import gymnasium as gym
import numpy as np
from Wrappers.RandomStartLunarLander import RandomStartLunarLander



# Create environment with the custom wrapper
env = RandomStartLunarLander(gym.make("LunarLander-v3", render_mode="human"))
seeds = [2 ,45 ,654 ,77 ,23 , 22 , 323, 33 ,43  , 32 ,334 ,53]
# Run a few episodes to check the new initialization
for i in range(10):
    obs, _ = env.reset(seed=seeds[i])
    print("Randomized Initial Observation:", obs)

    for _ in range(10):
        action = env.action_space.sample()  # Take a random action
        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()

        if terminated or truncated:
            break  # Restart if episode ends early

env.close()
