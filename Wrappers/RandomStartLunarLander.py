import gymnasium as gym
import numpy as np

class RandomStartLunarLander(gym.Wrapper):
    def __init__(self, env, max_steps=100):
        super().__init__(env)
        # self.env = TimeLimit(env, max_episode_steps=max_steps)  # Limit episode length

    def reset(self, **kwargs):
        """Resets the environment and modifies the actual lander position."""
        obs, info = self.env.reset(**kwargs)

        # Define reasonable initialization ranges to ensure a more uniform start
        start_x = np.random.uniform(-5, 5)  # Random x position

        # Modify the internal lander's position
        if hasattr(self.env.unwrapped, 'lander'):  # Ensure lander exists
            self.env.unwrapped.lander.position[0] = start_x  # Modify x position
            obs[0] = start_x  # Ensure observation reflects this change
        else :
            print("no attribute")

        return obs, info
