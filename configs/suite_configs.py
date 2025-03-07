SUITE_CONFIGS = {

    'config_A': {
        'INPUT_STATE_SIZE': 4,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 2,

        'ENCODER_HIDDEN': 16,
        'ENCODER_HIDDEN2': 16,
        'ENCODER_HIDDEN3': 12,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 6,
        'DECODER_HIDDEN2': 8,
        'DECODER_HIDDEN3': 8,
        'DECODER_HIDDEN4': 10,
        'DECODER_HIDDEN5': 12,
        'DECODER_HIDDEN6': 12,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.001,
        'TRAIN_FREQUENCY' : 4,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 2,
        'ACT_FUNC' : "LeakyReLU(0.1)",
        'NORM_FUNC' : "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"




    },

'config_B': {
        'INPUT_STATE_SIZE': 6,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 2,

        'ENCODER_HIDDEN': 16,
        'ENCODER_HIDDEN2': 16,
        'ENCODER_HIDDEN3': 12,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 6,
        'DECODER_HIDDEN2': 8,
        'DECODER_HIDDEN3': 8,
        'DECODER_HIDDEN4': 10,
        'DECODER_HIDDEN5': 12,
        'DECODER_HIDDEN6': 12,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.001,
        'TRAIN_FREQUENCY' : 4,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 2,
        'ACT_FUNC' : "LeakyReLU(0.1)",
        'NORM_FUNC' : "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"




    },
    # Add additional configurations as needed

}