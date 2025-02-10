import gymnasium as gym
import numpy as np
import configs.eval_config as eval_config

class RandomStartCartPoleEval(gym.Wrapper):
    def reset(self, **kwargs):
        """Resets the environment and sets a random initial position."""
        obs, info = self.env.reset(**kwargs)

        # Randomize initial state
        cart_position = np.random.uniform(eval_config.CART_POS[0], eval_config.CART_POS[1])  # Full cart track range
        cart_velocity = np.random.uniform(eval_config.CART_VELO[0], eval_config.CART_VELO[1])  # Some random velocity
        pole_angle = np.random.uniform(eval_config.POLE_ANG[0], eval_config.POLE_ANG[1])  # Random pole angle
        pole_angular_velocity = np.random.uniform(eval_config.POLE_ANG_VEL[0], eval_config.POLE_ANG_VEL[1])  # Random spin

        # Set new randomized initial state
        self.env.unwrapped.state = [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

        return np.array(self.env.unwrapped.state, dtype=np.float32), info