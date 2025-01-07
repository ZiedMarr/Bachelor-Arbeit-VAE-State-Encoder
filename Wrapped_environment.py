from collections import deque
import numpy as np
import torch
import gymnasium as gym


class VAEWrapperWithHistory(gym.ObservationWrapper):
    def __init__(self, env, vae_model, n, m, vae_optimizer):
        """
        Wrapper to integrate VAE for state prediction and PPO training.
        :param env: Gymnasium environment.
        :param vae_model: VAE model for state encoding and prediction.
        :param n: Number of current states for VAE input.
        :param m: Number of next states for VAE target.
        :param vae_optimizer: Optimizer for VAE training.
        """
        super(VAEWrapperWithHistory, self).__init__(env)
        self.vae = vae_model
        self.vae_optimizer = vae_optimizer
        self.n = n  # Number of states for VAE input
        self.m = m  # Number of next states for VAE target
        self.buffer_obs = deque(maxlen=n + m)  # Buffer for observations        #TODO : check if data structure is optimal

        # Original observation and latent dimensions
        self.original_obs_dim = env.observation_space.shape[0]
        self.latent_dim = vae_model.encoder[-1].out_features // 2

        # Update the observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.original_obs_dim + self.latent_dim,),
            dtype=np.float32
        )

    def reset(self, **kwargs):
        """
        Reset the environment and initialize the buffer.
        """

        obs, info = self.env.reset(**kwargs)
        #print("First observation at reset:", obs)
        for _ in range(self.n + self.m):  # Fill buffer with initial observation
            self.buffer_obs.append(obs)
        return self._get_combined_observation(), info

    def observation(self, obs):
        """
        Update the buffer with the new observation.
        """
        self.buffer_obs.append(obs)
        return self._get_combined_observation()

    def _get_combined_observation(self):
        """
        Generate the combined observation with the original state and latent variables.
        The latent variables are computed based on the most recent n states.
        """
        # Extract the last n observations from the buffer
        stacked_obs = np.concatenate(list(self.buffer_obs)[-self.n:], axis=-1)

        # Convert the stacked observations to a torch tensor
        stacked_obs_tensor = torch.tensor(stacked_obs, dtype=torch.float32).unsqueeze(0)

        # Get latent representation from the VAE
        with torch.no_grad():
            latent = self.vae.encode(stacked_obs_tensor).squeeze(0).numpy()

        # Concatenate the most recent observation with the latent variables
        combined_obs = np.concatenate([self.buffer_obs[-1], latent], axis=-1)
        return combined_obs

    def train_vae(self):
        """
        Train the VAE using n current states as input and m next states as target.
        """
        # Ensure sufficient data in the buffer
        assert len(self.buffer_obs) >= self.n + self.m, "Buffer does not contain enough states for training."

        # Extract the n current states
        stacked_obs = np.concatenate(list(self.buffer_obs)[:self.n], axis=-1)

        # Extract the m next states starting after the n-th state
        stacked_next_obs = np.concatenate(list(self.buffer_obs)[self.n:self.n + self.m], axis=-1)

        # Convert to tensors
        input_tensor = torch.tensor(stacked_obs, dtype=torch.float32).unsqueeze(0)
        target_tensor = torch.tensor(stacked_next_obs, dtype=torch.float32).unsqueeze(0)

        # Forward pass through the VAE
        predicted_next_states, mu, log_var, _ = self.vae(input_tensor)

        # Compute VAE loss
        loss = self._vae_loss(predicted_next_states, target_tensor, mu, log_var)

        # Backpropagation and optimization
        self.vae_optimizer.zero_grad()
        loss.backward()
        self.vae_optimizer.step()

        return loss.item()

    @staticmethod
    def _vae_loss(predicted_next_states, target_next_states, mu, log_var):

        # TODO :  Include coefficients for the 2 losses to balance them in the final loss
        """
        Compute the VAE loss: Prediction loss + KL divergence.
        """
        # Prediction loss (MSE)
        prediction_loss = torch.nn.MSELoss()(predicted_next_states, target_next_states)

        # KL Divergence loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # Total loss
        return prediction_loss + kl_loss
