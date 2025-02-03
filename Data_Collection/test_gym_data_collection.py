import gym_data_collection
import numpy as np

path = './collected_data test/cartpole_data.npz'

all_observations_collected , episode_starts_collected , episode_lengths_collected =gym_data_collection.random_collect(path)
all_observations_loaded , episode_starts_loaded , episode_lengths_loaded =gym_data_collection.load_data(path)

def test_loading_observations() :
    obs_col = np.array(all_observations_collected, dtype=object)
    obs_load = np.array(all_observations_loaded, dtype=object)

    same_content = all(np.array_equal(o1,o2) for o1,o2 in zip(obs_col,obs_load))
    return same_content

