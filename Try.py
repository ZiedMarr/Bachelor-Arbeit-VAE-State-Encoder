import gymnasium as gym
import numpy as np
from Wrappers.RandomStartBipedalWalker import RandomStartBipedalWalker

class RandomStartCartPole(gym.Wrapper):
    def reset(self, **kwargs):
        """Resets the environment and sets a random initial position."""
        obs, info = self.env.reset(**kwargs)

        # Randomize initial state
        cart_position = -2.2  # Full cart track range
        cart_velocity = np.random.uniform(-1.0, 1.0)  # Some random velocity
        pole_angle = np.random.uniform(-0.2, 0.2)  # Random pole angle
        pole_angular_velocity = np.random.uniform(-1.0, 1.0)  # Random spin

        # Set new randomized initial state
        self.env.unwrapped.state = [.5 , .5 , .5 , .5 , .5,.5 , .5 , .5 , .5 , .5,.5 , .5 , .5 , .5 , .5,.5 , .5 , .5 , .5 , .5, .5]

        return np.array(self.env.unwrapped.state, dtype=np.float32), info

# Create environment with the custom wrapper
env = RandomStartBipedalWalker(gym.make("BipedalWalker-v3", render_mode="human"))

# Run a few episodes to check the new initialization
for _ in range(5):
    obs, _ = env.reset()
    print("Randomized Initial Observation:", obs)

    for _ in range(100):
        action = env.action_space.sample()  # Take a random action
        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()

        if terminated or truncated:
            break  # Restart if episode ends early

env.close()
