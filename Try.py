import gymnasium as gym
import numpy as np
from Wrappers.RandomStartLunarLander import RandomStartLunarLander
from configs import config as config_module

'''
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
'''
def load_config_from_file(file_path):
    """Load configuration from a text file into a dictionary."""
    config_dict = {}

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and "=" in line:  # Ignore empty lines
                key, value = map(str.strip, line.split("=", 1))

                # Convert numerical values if possible
                try:
                    value = eval(value)  # Convert numbers & tuples (be careful with eval)
                except:
                    pass  # Keep as string if conversion fails

                config_dict[key] = value

    return config_dict

class Config:
    """Dynamically loads attributes from a dictionary."""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)  # Dynamically add attributes

# Load config dictionary from file
config_dict = load_config_from_file("VAE_pretrain/pretrained_vae/VAE_Version_2.1/2_2/KL-D_0.001/VAE_config_config_B.txt")

# Create a Config object with loaded values
config = Config(config_dict)

# Update global config module values for compatibility
for key, value in vars(config).items():
    setattr(config_module, key, value)

# Print the attributes to verify
print(vars(config_module))
