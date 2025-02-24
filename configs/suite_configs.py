SUITE_CONFIGS = {

    # ----- CONFIG A -----
    'config_A_0': {
        'INPUT_STATE_SIZE': 2,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 2,
        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48,
        'BETA_KL_DIV': 0.0008,
        'TRAIN_FREQUENCY': 5,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 2,
        'ACT_FUNC': "LeakyReLU(0.1)",
        'NORM_FUNC': "LayerNorm",
        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],
        'VAE_Version': "VAE_Version_2.1"
    },

    'config_A_1': {
        'INPUT_STATE_SIZE': 2,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 2,
        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48,
        'BETA_KL_DIV': 0.0009,  # Slight variation in KL divergence
        'TRAIN_FREQUENCY': 5,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 2,
        'ACT_FUNC': "LeakyReLU(0.1)",
        'NORM_FUNC': "LayerNorm",
        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],
        'VAE_Version': "VAE_Version_2.1"
    },

    # ----- CONFIG B -----
    'config_B_0': {
        'INPUT_STATE_SIZE': 3,
        'OUTPUT_STATE_SIZE': 3,
        'LATENT_DIM': 3,
        'ENCODER_HIDDEN': 64,
        'ENCODER_HIDDEN2': 48,
        'ENCODER_HIDDEN3': 32,
        'ENCODER_HIDDEN4': 16,
        'DECODER_HIDDEN': 16,
        'DECODER_HIDDEN2': 32,
        'DECODER_HIDDEN3': 48,
        'DECODER_HIDDEN4': 64,
        'BETA_KL_DIV': 0.0009,
        'VAE_Version': "VAE_Version_2.2"
    },

    'config_B_1': {
        'INPUT_STATE_SIZE': 3,
        'OUTPUT_STATE_SIZE': 3,
        'LATENT_DIM': 4,  # Increased latent dim
        'ENCODER_HIDDEN': 64,
        'ENCODER_HIDDEN2': 48,
        'ENCODER_HIDDEN3': 32,
        'ENCODER_HIDDEN4': 16,
        'DECODER_HIDDEN': 16,
        'DECODER_HIDDEN2': 32,
        'DECODER_HIDDEN3': 48,
        'DECODER_HIDDEN4': 64,
        'BETA_KL_DIV': 0.0009,
        'VAE_Version': "VAE_Version_2.2"
    },

    # ----- CONFIG C -----
    'config_C_0': {
        'INPUT_STATE_SIZE': 4,
        'OUTPUT_STATE_SIZE': 4,
        'LATENT_DIM': 4,
        'ENCODER_HIDDEN': 96,
        'ENCODER_HIDDEN2': 64,
        'ENCODER_HIDDEN3': 48,
        'ENCODER_HIDDEN4': 32,
        'DECODER_HIDDEN': 32,
        'DECODER_HIDDEN2': 48,
        'DECODER_HIDDEN3': 64,
        'DECODER_HIDDEN4': 96,
        'BETA_KL_DIV': 0.001,
        'VAE_Version': "VAE_Version_2.3"
    },

    'config_C_1': {
        'INPUT_STATE_SIZE': 4,
        'OUTPUT_STATE_SIZE': 4,
        'LATENT_DIM': 4,
        'ENCODER_HIDDEN': 96,
        'ENCODER_HIDDEN2': 72,  # Slight change in layer size
        'ENCODER_HIDDEN3': 48,
        'ENCODER_HIDDEN4': 32,
        'DECODER_HIDDEN': 32,
        'DECODER_HIDDEN2': 48,
        'DECODER_HIDDEN3': 64,
        'DECODER_HIDDEN4': 96,
        'BETA_KL_DIV': 0.001,
        'VAE_Version': "VAE_Version_2.3"
    },

    # Continue adding up to config_J_0 and config_J_1

    'config_J_0': {
        'INPUT_STATE_SIZE': 7,
        'OUTPUT_STATE_SIZE': 7,
        'LATENT_DIM': 5,
        'ENCODER_HIDDEN': 160,
        'ENCODER_HIDDEN2': 128,
        'ENCODER_HIDDEN3': 96,
        'ENCODER_HIDDEN4': 64,
        'DECODER_HIDDEN': 64,
        'DECODER_HIDDEN2': 96,
        'DECODER_HIDDEN3': 128,
        'DECODER_HIDDEN4': 160,
        'BETA_KL_DIV': 0.0009,
        'VAE_Version': "VAE_Version_2.10"
    },

    'config_J_1': {
        'INPUT_STATE_SIZE': 7,
        'OUTPUT_STATE_SIZE': 7,
        'LATENT_DIM': 5,
        'ENCODER_HIDDEN': 160,
        'ENCODER_HIDDEN2': 128,
        'ENCODER_HIDDEN3': 96,
        'ENCODER_HIDDEN4': 64,
        'DECODER_HIDDEN': 64,
        'DECODER_HIDDEN2': 96,
        'DECODER_HIDDEN3': 128,
        'DECODER_HIDDEN4': 160,
        'BETA_KL_DIV': 0.0008,  # Small variation in KL divergence
        'VAE_Version': "VAE_Version_2.10"
    },
}
