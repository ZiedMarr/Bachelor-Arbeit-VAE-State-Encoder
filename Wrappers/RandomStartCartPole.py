import gymnasium as gym
import numpy as np

class RandomStartCartPole(gym.Wrapper):
    def reset(self, **kwargs):
        """Resets the environment and sets a random initial position."""
        obs, info = self.env.reset(**kwargs)

        # Randomize initial state
        cart_position = np.random.uniform(-2.4, 2.4)  # Full cart track range
        cart_velocity = np.random.uniform(-1.0, 1.0)  # Some random velocity
        pole_angle = np.random.uniform(-0.2, 0.2)  # Random pole angle
        pole_angular_velocity = np.random.uniform(-1.0, 1.0)  # Random spin

        # Set new randomized initial state
        self.env.unwrapped.state = [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

        return np.array(self.env.unwrapped.state, dtype=np.float32), info