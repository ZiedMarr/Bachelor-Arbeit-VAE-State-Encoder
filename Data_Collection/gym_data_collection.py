import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO  # Import stable-baselines3 for pre-trained policy



def random_collect(path='./collected data/cartpole_data_random.npz', num_episodes =20) :
    #create env
    env = gym.make('CartPole-v1')

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
def expert_collect(path='./collected data/cartpole_data_expert.npz', policy_path="path_to_expert_policy.zip", num_episodes =20) :

    # Load a pre-trained policy
    pretrained_policy = PPO.load(policy_path)

    # create env
    env = gym.make('CartPole-v1')

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
    np.savez(path,
             observations=flat_observations,
             episode_starts=np.array(episode_starts),
             episode_lengths=np.array(episode_lengths))
    return all_observations, episode_starts , episode_lengths



################################################
################################################

def load_data(path='./collected data/cartpole_data.npz') :
    data = np.load(path)

    # Extract the data
    observations_loaded = data['observations']
    episode_starts_loaded = data['episode_starts']
    episode_lengths_loaded = data['episode_lengths']

    # If needed, reconstruct episodes from `observations` and `episode_starts`
    reconstructed_episodes = [observations_loaded[episode_starts_loaded[i]:episode_starts_loaded[i]+episode_lengths_loaded[i]]
                              for i in range(len(episode_starts_loaded) )]

    return reconstructed_episodes, episode_starts_loaded,episode_lengths_loaded


#expert_collect()
#episodes , _ , _ = load_data()
#print(f"len of episode 1 is {len(episodes[1])}. \n episode 0 is :  {episodes[0]}" )
