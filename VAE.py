import torch.nn as nn
import torch
import config


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        super(VAE, self).__init__()

        # Encoder with Batch Normalization
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, config.ENCODER_HIDDEN),
            nn.BatchNorm1d(config.ENCODER_HIDDEN),
            nn.LeakyReLU(0.1),

            nn.Linear(config.ENCODER_HIDDEN, config.ENCODER_HIDDEN2),
            nn.BatchNorm1d(config.ENCODER_HIDDEN2),
            nn.LeakyReLU(0.1),

            nn.Linear(config.ENCODER_HIDDEN2, config.ENCODER_HIDDEN3),
            nn.BatchNorm1d(config.ENCODER_HIDDEN3),
            nn.LeakyReLU(0.1),
        )

        # Latent space layers
        self.fc_mu = nn.Linear(config.ENCODER_HIDDEN3, latent_dim)
        self.fc_var = nn.Linear(config.ENCODER_HIDDEN3, latent_dim)

        # Decoder with Batch Normalization
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, config.DECODER_HIDDEN),
            nn.BatchNorm1d(config.DECODER_HIDDEN),
            nn.LeakyReLU(0.1),

            nn.Linear(config.DECODER_HIDDEN, config.DECODER_HIDDEN2),
            nn.BatchNorm1d(config.DECODER_HIDDEN2),
            nn.LeakyReLU(0.1),

            nn.Linear(config.DECODER_HIDDEN2, output_dim)
        )

    def encode(self, x):
        # Return only mu, not the full tuple
        mu, log_var = self.encode_full(x)
        return mu

    def encode_full(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode_full(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var, z

    def MSE_Loss(self, mu , log_var, predicted_next_states, target_tensor):
        reconstruction_loss = torch.nn.MSELoss()(predicted_next_states, target_tensor)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = reconstruction_loss + config.BETA_KL_DIV * kl_loss

        return loss

    def MSE_loss_feature_Standardization(self, mu , log_var, predicted_next_states, target_tensor):
        # Calculate mean and std for normalization (per feature/dimension)
        mean = target_tensor.mean(dim=0, keepdim=True)
        std = target_tensor.std(dim=0, keepdim=True)

        # Add small epsilon to avoid division by zero, use max to ensure non-zero std
        std = torch.max(std, torch.tensor(1e-8, device=std.device))

        # Normalize both predicted and target tensors
        predicted_normalized = (predicted_next_states - mean) / std
        target_normalized = (target_tensor - mean) / std

        # Reconstruction loss (normalized)
        reconstruction_loss = torch.nn.MSELoss()(predicted_normalized, target_normalized)

        # KL Divergence loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # Final loss with KL divergence regularization
        loss = reconstruction_loss + config.BETA_KL_DIV * kl_loss

        return loss