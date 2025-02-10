from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import torch
from configs import config


class VAETrainingCallback(BaseCallback):
    def __init__(self, vae, optimizer, train_frequency, n, m, verbose=0, original_obs_shape=4, batch_size=32):
        super(VAETrainingCallback, self).__init__(verbose)
        self.vae = vae
        self.optimizer = optimizer
        self.train_frequency = train_frequency
        self.n = n
        self.m = m
        self.train_count = 0
        self.original_obs_shape = original_obs_shape
        self.batch_size = batch_size

    def _on_rollout_end(self) -> None:
        """
        Called after a rollout ends. Train the VAE on the rollout buffer.
        """
        self.vae.train()

        rollout_buffer = self.model.rollout_buffer  # Access PPO's buffer
        episode_starts = rollout_buffer.episode_starts
        #dones = ~episode_starts  # Invert episode_starts to get "dones"
        ###############################
        # Split the rollout buffer into distinct episodes (only observations)
        split_episodes = []
        current_episode = []

        for i in range(len(episode_starts)):
            if episode_starts[i]:
                if current_episode:
                    # Save the current episode
                    split_episodes.append(np.array(current_episode))
                # Start a new episode
                current_episode = []

            # Append the current observation to the episode

            #wrapped_obs = rollout_buffer.observations[i]
            #orihinal_obs_array = [rollout_buffer.observations[i][0][:self.original_obs_shape]]



            current_episode.append([rollout_buffer.observations[i][0][:self.original_obs_shape]])
            #current_episode.append(rollout_buffer.observations[i][:self.original_obs_shape])

        # Add the last episode if it exists
        if current_episode:
            split_episodes.append(np.array(current_episode))


        #####################################new batch method##############################
            # Collect all sliding window samples
            all_inputs, all_targets = [], []
            for episode in split_episodes:
                for inp_obs_1_index in range(0, len(episode) - config.INPUT_STATE_SIZE - config.OUTPUT_STATE_SIZE,
                                             self.train_frequency):
                    stacked_obs = np.concatenate(
                        list(episode)[inp_obs_1_index:inp_obs_1_index + config.INPUT_STATE_SIZE],
                        axis=-1)
                    stacked_next_obs = np.concatenate(list(episode)[
                                                      inp_obs_1_index + config.INPUT_STATE_SIZE:inp_obs_1_index + config.INPUT_STATE_SIZE + config.OUTPUT_STATE_SIZE],
                                                      axis=-1)

                    all_inputs.append(stacked_obs)
                    all_targets.append(stacked_next_obs)

            # Convert to torch tensors
            inputs_tensor = torch.tensor(all_inputs, dtype=torch.float32)
            targets_tensor = torch.tensor(all_targets, dtype=torch.float32)

            # Create DataLoader for batched training
            dataset = torch.utils.data.TensorDataset(inputs_tensor, targets_tensor)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            # Batch training loop

            for batch_inputs, batch_targets in dataloader:
                predicted_next_states, mu, log_var, _ = self.vae(batch_inputs)

                loss = self.vae.MSE_Loss(mu=mu, log_var=log_var,
                                        predicted_next_states=predicted_next_states,
                                        target_tensor=batch_targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        ##################################### new batch method ############################








        '''
        #################Old Method######################
        # Train the VAE on every episode
        #initialize training frequencey :
        train_frequency = self.train_frequency
        for episode in split_episodes :
            #initialize the first element of sliding window
            inp_obs_1_index = 0

            #train on the current episode :
            while inp_obs_1_index +self.n +self.m <= len(episode) :
                # Extract the n current states
                stacked_obs = np.concatenate(list(episode)[inp_obs_1_index:inp_obs_1_index+self.n], axis=-1)

                # Extract the m next states starting after the n-th state
                stacked_next_obs = np.concatenate(list(episode)[inp_obs_1_index+self.n:inp_obs_1_index+self.n + self.m], axis=-1)

                # Convert to tensors
                input_tensor = torch.tensor(stacked_obs, dtype=torch.float32).unsqueeze(0)
                target_tensor = torch.tensor(stacked_next_obs, dtype=torch.float32).unsqueeze(0)

                # Forward pass through the VAE
                predicted_next_states, mu, log_var, _ = self.vae(input_tensor)

                # Compute the VAE loss
                reconstruction_loss = torch.nn.MSELoss()(predicted_next_states, target_tensor)
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = reconstruction_loss + BETA_KL_DIV*kl_loss

                # Backpropagation and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                inp_obs_1_index += train_frequency
                ##############old method###########################
                '''
                #self.train_count += 1
                #if self.verbose:
                #    print(f"VAE Training Loss (Iteration {self.train_count}): {loss.item():.4f}")
                ##############
        #####################################





    def _on_step(self) -> bool:
        return True