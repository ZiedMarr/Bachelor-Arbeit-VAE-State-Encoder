import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit

class RandomStartLunarLander(gym.Wrapper):
    def __init__(self, env, max_steps=100):
        super().__init__(env)
        #self.env = TimeLimit(env, max_episode_steps=max_steps)  # Limit episode length

    def reset(self, **kwargs):
        """Resets the environment and applies a more controlled random initialization."""
        obs, info = self.env.reset(**kwargs)

        # Define reasonable initialization ranges to ensure a more uniform start
        obs_ranges = {
            0: (-2.5 , 2.5 ) #starting position
        }

        # Apply randomization within the defined ranges
        for idx, (low, high) in obs_ranges.items():
            obs[idx] = np.random.uniform(low, high)

        return obs, info
