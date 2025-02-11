import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit


class RandomStartCartPole(gym.Wrapper):
    def __init__(self, env, max_steps=50):
        super().__init__(env)
        self.env = TimeLimit(env, max_episode_steps=max_steps)  # Limit episode length

    def reset(self, **kwargs):
        """Resets the environment and sets a random initial position."""
        obs, info = self.env.reset(**kwargs)


        cart_position = np.random.uniform(-2.3, 2.3)  # Full cart track range
        cart_velocity = np.random.uniform(-2.0, 2.0)  # Some random velocity
        pole_angle = np.random.uniform(-.2094, .2094)  # Random pole angle
        pole_angular_velocity = np.random.uniform(-2.0, 2.0)  # Random spin

        # Set new randomized initial state
        self.env.unwrapped.state = [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

        return np.array(self.env.unwrapped.state, dtype=np.float32), info