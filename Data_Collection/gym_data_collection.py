from random import random

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO  # Import stable-baselines3 for pre-trained policy
from datetime import datetime
import os
from Wrappers.RandomStartLunarLander import RandomStartLunarLander


# Define the base directory (directory of the current script)
base_dir = os.path.dirname(os.path.abspath(__file__))

def random_collect( output_path= "" , num_episodes =10 , env_wrapper= None) :

    # Generate dynamic path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # create save_dir :
    save_dir = os.path.join(base_dir, "collected_data",output_path)
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir , f"random_{num_episodes}_{timestamp}.npz")

    #create env
    env = gym.make('BipedalWalker-v3')
    if env_wrapper is not None :
        env = env_wrapper(env)

    #data storage
    all_observations = []
    episode_starts = [0]
    episode_lengths = []



    #collection loop
    for episode in range(num_episodes) :
        observation, _ = env.reset()
        episode_observations = []
        done= False
        step_count = 0

        #running episode
        while not done :
            episode_observations.append(observation)
            action = env.action_space.sample()       #random action

            observation, _ , done, truncated,_ = env.step(action)
            step_count += 1

            if done or truncated :
                break

        #save episode data
        all_observations.append(episode_observations)
        episode_starts.append(episode_starts[episode] + step_count )
        episode_lengths.append(step_count)

    #remove last element
    episode_starts = episode_starts[:-1]
    #print(f"length of episode starts is : {len(episode_starts)}")
    #print(f"length of episode 1 : {len(all_observations[1])}")

    #close environment
    env.close()



    # Convert `all_observations` to a single numpy array
    # Flatten episodes while keeping their boundaries
    flat_observations = np.concatenate(all_observations, axis=0)

    # Save data to .npz
    np.savez(path,
             observations=flat_observations,
             episode_starts=np.array(episode_starts),
             episode_lengths=np.array(episode_lengths))
    return all_observations, episode_starts , episode_lengths
################################################
################################################
def expert_collect(  output_path, policy_path=os.path.join(base_dir, "path_to_expert_policy.zip"), num_episodes =20, env_wrapper=None, noise=False ,noise_scale=0.3) :
    # Generate dynamic path
    policy_name = policy_path.split('/')[-1].split('.')[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_path is None :
        output_path = os.path.join(base_dir, "collected_data", "train", "explore_pol_standard_env", f"{policy_name}_{num_episodes}_{timestamp}.npz")

    # Load a pre-trained policy
    pretrained_policy = PPO.load(policy_path, device='cpu')

    # create env
    env = gym.make('BipedalWalker-v3')

    if env_wrapper is not None :
        env = env_wrapper(env)

    # data storage
    all_observations = []
    episode_starts = [0]
    episode_lengths = []

    #collection loop
    for episode in range(num_episodes) :
        observation, _ = env.reset()
        episode_observations = []
        done= False
        step_count = 0

        #running episode
        while not done :
            episode_observations.append(observation)
            # Use the pre-trained policy to select an action


            if noise :
                noise_chance = np.random.uniform(0,1)
                if noise_chance > noise_scale :
                    action, _ = pretrained_policy.predict(observation, deterministic=True)
                else :
                    action =   env.action_space.sample()       #random action
            else :
                action, _ = pretrained_policy.predict(observation, deterministic=True)

            observation, _ , done, truncated,_ = env.step(action)
            step_count += 1

            if done or truncated :
                break

        #save episode data
        all_observations.append(episode_observations)
        episode_starts.append(episode_starts[episode] + step_count )
        episode_lengths.append(step_count)

    #remove last element
    episode_starts = episode_starts[:-1]
    #print(f"length of episode starts is : {len(episode_starts)}")
    #print(f"length of episode 1 : {len(all_observations[1])}")

    #close environment
    env.close()



    # Convert `all_observations` to a single numpy array
    # Flatten episodes while keeping their boundaries
    flat_observations = np.concatenate(all_observations, axis=0)

    # Save data to .npz
    np.savez(output_path,
             observations=flat_observations,
             episode_starts=np.array(episode_starts),
             episode_lengths=np.array(episode_lengths))
    return all_observations, episode_starts , episode_lengths



################################################
################################################


def mixed_random_expert_collect(  output_path, policy_path=os.path.join(base_dir, "path_to_expert_policy.zip"), num_episodes =20, env_wrapper=None) :
    # Generate dynamic path
    policy_name = policy_path.split('/')[-1].split('.')[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_path is None :
        output_path = os.path.join(base_dir, "collected_data", "mixed_pol_rand_env",f"{policy_name}_{num_episodes}_{timestamp}.npz")

    # Load a pre-trained policy
    pretrained_policy = PPO.load(policy_path)

    # create env
    env = gym.make('BipedalWalker-v3')

    if env_wrapper is not None :
        env = env_wrapper(env)

    # data storage
    all_observations = []
    episode_starts = [0]
    episode_lengths = []

    #collection loop
    for episode in range(num_episodes) :
        observation, _ = env.reset()
        episode_observations = []
        done= False
        step_count = 0

        #running episode
        while not done :
            episode_observations.append(observation)
            # determine action based on observations
            if observation[0]< -2.2 or observation[0]> 2.2 or observation[1]< -2.2 or observation[1]> 2.2 or observation[2]< -0.2 or observation[2]> 0.2 or observation[3]< -2 or observation[3]> 2 :
                action, _ = pretrained_policy.predict(observation, deterministic=True)
            else :
                action = env.action_space.sample()  # random action

            observation, _ , done, truncated,_ = env.step(action)
            step_count += 1

            if done or truncated :
                break

        #save episode data
        all_observations.append(episode_observations)
        episode_starts.append(episode_starts[episode] + step_count )
        episode_lengths.append(step_count)

    #remove last element
    episode_starts = episode_starts[:-1]
    #print(f"length of episode starts is : {len(episode_starts)}")
    #print(f"length of episode 1 : {len(all_observations[1])}")

    #close environment
    env.close()



    # Convert `all_observations` to a single numpy array
    # Flatten episodes while keeping their boundaries
    flat_observations = np.concatenate(all_observations, axis=0)

    # Save data to .npz
    np.savez(output_path,
             observations=flat_observations,
             episode_starts=np.array(episode_starts),
             episode_lengths=np.array(episode_lengths))
    return all_observations, episode_starts , episode_lengths



################################################
################################################



def load_data(path = os.path.join(base_dir, "collected_data", "cartpole_data.npz")) :
    data = np.load(path)

    # Extract the data
    observations_loaded = data['observations']
    episode_starts_loaded = data['episode_starts']
    episode_lengths_loaded = data['episode_lengths']

    # If needed, reconstruct episodes from `observations` and `episode_starts`
    reconstructed_episodes = [observations_loaded[episode_starts_loaded[i]:episode_starts_loaded[i]+episode_lengths_loaded[i]]
                              for i in range(len(episode_starts_loaded) )]

    return reconstructed_episodes, episode_starts_loaded,episode_lengths_loaded

################# Collect data from batch of policies #####################
def collect_data_from_model(model_path, index, num_episodes=50, output_path = os.path.join(base_dir,"collected_data", "train", "explore_pol_standard_env") , env_wrapper = None, noise=False, noise_scale=0.3):
    """
    Collect data using the expert_collect function and save it with an indexed filename.

    Parameters:
    - model_path (str): Path to the expert model.
    - index (int): Index for the output filename.
    - num_episodes (int): Number of episodes to collect data over.
    """
    # Define the output path with the given index
    output_path = os.path.join(output_path, f"BipedalWalker_ppo_data_{index}.npz")

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Collect data using the expert_collect function
    expert_collect(output_path=output_path, policy_path=model_path, num_episodes=num_episodes , env_wrapper=env_wrapper, noise=noise, noise_scale=noise_scale)


def collect_from_batch(root_dir= os.path.join(base_dir,"..", "PPO","logs", "explore"), output_path  = os.path.join(base_dir, "explore_expert") , env_wrapper = None, noise=False, noise_scale=0.3, num_episodes=50):

    # Initialize index
    index = 0

    # Traverse the directory structure
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file == 'best_model.zip':
                # Construct the full path to the model
                model_path = os.path.join(subdir, file)

                print(f" collecting data of {model_path}")
                # Collect data from the model
                collect_data_from_model(model_path, index, output_path=output_path, env_wrapper=env_wrapper, noise=noise, noise_scale=noise_scale, num_episodes=num_episodes)

                # Increment the index for the next file
                index += 1
                print(index)



if __name__ == "__main__":
    # Example calls for testing
    #expert_collect(output_path = os.path.join(base_dir, "collected_data", "cartpole_expert_60"),policy_path = os.path.join(base_dir, "..", "PPO", "logs","batch2","logs_20000_20250123_151149","best_model", "best_model.zip"), num_episodes=60)

    random_collect(output_path=os.path.join("eval","rand_pol_rand_env"), num_episodes=400, env_wrapper=None)

    #collect_from_batch(root_dir= os.path.join(base_dir,"..", "PPO","logs", "explore", "batch_10_500k"),output_path=os.path.join(base_dir,"collected_data", "eval", "explore_pol_standard_env","ppo_500k_noise_0.5"), noise=True,noise_scale=0.5, num_episodes=10,env_wrapper=None)
    #collect_from_batch(root_dir= os.path.join(base_dir,"..", "PPO","logs", "explore", "batch_10_200k"),output_path=os.path.join(base_dir,"collected_data", "eval", "explore_pol_standard_env","ppo_200k_noise_0.5"), noise=True,noise_scale=0.5, num_episodes=10,env_wrapper=None)



    #mixed_random_expert_collect(output_path="mixed_pol_rand_env", num_episodes=10000, env_wrapper=RandomStartCartPole, policy_path="../PPO/logs/explore_rand_env/batch_20000_timesteps_rand_env_100k_steps/logs_100000_20250210_110044/best_model/best_model.zip")
    #collect_from_batch(root_dir='../PPO/logs/explore/', output_path= os.path.join(base_dir, "collected_data", "explore_rand_env"), env_wrapper=RandomStartCartPole)
