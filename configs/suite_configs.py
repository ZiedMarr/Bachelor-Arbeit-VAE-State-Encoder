SUITE_CONFIGS = {



'config_v1_penta_2': {
    'INPUT_STATE_SIZE': 6,
    'OUTPUT_STATE_SIZE': 2,
    'LATENT_DIM': 4,
    'ENCODER_HIDDEN': 36,
    'ENCODER_HIDDEN2': 26,
    'ENCODER_HIDDEN3': 16,
    'ENCODER_HIDDEN4': 6,
    'ENCODER_HIDDEN5': 0,
    'ENCODER_HIDDEN6': 0,
    'ENCODER_HIDDEN7': 0,
    'ENCODER_HIDDEN8': 0,
    'ENCODER_HIDDEN9': 0,
    'DECODER_HIDDEN': 6,
    'DECODER_HIDDEN2': 16,
    'DECODER_HIDDEN3': 26,
    'DECODER_HIDDEN4': 36,
    'DECODER_HIDDEN5': 0,
    'DECODER_HIDDEN6': 0,
    'DECODER_HIDDEN7': 0,
    'DECODER_HIDDEN8': 0,
    'DECODER_HIDDEN9': 0,
    'BETA_KL_DIV': 0.0001,
    'TRAIN_FREQUENCY': 5,
    'LOSS_FUNC': "MSE_loss_feature_Standardization",
    'EPOCHS': 5,
    'ACT_FUNC': "LeakyReLU(0.1)",
    'NORM_FUNC': "LayerNorm",
    'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],
    'VAE_Version': "VAE_Version_1.08",
    'gradual_beta': "True"
},

}