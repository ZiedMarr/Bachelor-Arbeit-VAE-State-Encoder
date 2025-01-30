import torch
import torch.nn as nn

from config import ENCODER_HIDDEN, DECODER_HIDDEN


#TODO : Think about if it's better to separate the network that produces mu and sigma
#TODO : Think if we should use the log of sigma or not

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        """
        Variational Autoencoder (VAE) for predicting m next states from n input states.
        :param input_dim: Total dimension of the n stacked input states.
        :param latent_dim: Dimension of the latent space.
        :param output_dim: Total dimension of the m stacked output states (next states).
        """
        super(VAE, self).__init__()

        # Encoder: Maps input to latent space (outputs mean and log-variance)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, ENCODER_HIDDEN),
            nn.ReLU(),
            nn.Linear(ENCODER_HIDDEN, latent_dim * 2)  # Outputs: mean and log-variance
        )

        # Decoder: Maps latent space to predicted next states
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, DECODER_HIDDEN),
            nn.ReLU(),
            nn.Linear(DECODER_HIDDEN, output_dim)  # Outputs: predicted m next states
        )

    def forward(self, x):
        """
        Forward pass through the VAE.
        :param x: Input tensor of shape (batch_size, input_dim).
        :return: predicted_next_states, mean, log_variance, latent_vector
        """
        # Encode input into latent space
        encoded = self.encoder(x)
        mu, log_var = encoded.chunk(2, dim=-1)  # Split into mean and log-variance
        std = torch.exp(0.5 * log_var)
        z = mu + std * torch.randn_like(std)  # Reparameterization trick

        # Decode latent vector into predicted next states
        predicted_next_states = self.decoder(z)


        return predicted_next_states, mu, log_var, z

    def encode(self, x):
        """
        Encodes the input into the latent space.
        :param x: Input tensor of shape (batch_size, input_dim).
        :return: Latent mean (mu) of shape (batch_size, latent_dim).
        """
        encoded = self.encoder(x)
        mu, _ = encoded.chunk(2, dim=-1)
        return mu

    def decode(self, z):
        pass
    #TODO : Add decode function to check smoothness of latent space