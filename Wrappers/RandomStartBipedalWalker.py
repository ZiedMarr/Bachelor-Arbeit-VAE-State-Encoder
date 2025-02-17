import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit

class RandomStartBipedalWalker(gym.Wrapper):
    def __init__(self, env, max_steps=100):
        super().__init__(env)
        #self.env = TimeLimit(env, max_episode_steps=max_steps)  # Limit episode length

    def reset(self, **kwargs):
        """Resets the environment and applies a more controlled random initialization."""
        obs, info = self.env.reset(**kwargs)

        # Define reasonable initialization ranges to ensure a more uniform start
        obs_ranges = {
            0: (-2, 2),  # Hull angle (radians), slight tilt to avoid extreme rotations
            1: (-3.0, 3.0),  # Hull angular velocity, reasonable rotation speed
            2: (-3.0, 3.0),  # Horizontal speed, avoids excessive initial movement
            3: (-2.0, 3.0),  # Vertical speed, avoids free falls or jumps
            4: (-2.0, 2.0),  # Hip joint (left), allows reasonable stride variety
            5: (-3.0, 3.0),  # Knee joint (left), starts in a more natural bent position
            6: (-2.0, 2.0),  # Hip joint (right), symmetrical with left
            7: (-3.2, 3.0),  # Knee joint (right), similar to left
            #8: (-2.0, 2.0),  # Hip joint angular velocity (left), moderate joint movement
            9: (-2.0, 2.0),  # Knee joint angular velocity (left), slightly higher for faster adaptation
            10: (-3.0, 3.0),  # Hip joint angular velocity (right)
            11: (-2.0, 2.0),  # Knee joint angular velocity (right)
        }

        # Apply randomization within the defined ranges
        for idx, (low, high) in obs_ranges.items():
            obs[idx] = np.random.uniform(low, high)

        return obs, info
