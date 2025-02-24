SUITE_CONFIGS = {

    'config_A': {
        'INPUT_STATE_SIZE': 2,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 2,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48 ,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.001,
        'TRAIN_FREQUENCY' : 5,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 2,
        'ACT_FUNC' : "LeakyReLU(0.1)",
        'NORM_FUNC' : "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
'config_A_2': {
        'INPUT_STATE_SIZE': 2,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 2,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48 ,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.001,
        'TRAIN_FREQUENCY' : 5,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 2,
        'ACT_FUNC' : "LeakyReLU(0.1)",
        'NORM_FUNC' : "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
'config_A_3': {
        'INPUT_STATE_SIZE': 2,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 2,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48 ,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.001,
        'TRAIN_FREQUENCY' : 5,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 3,
        'ACT_FUNC' : "LeakyReLU(0.1)",
        'NORM_FUNC' : "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
'config_A_4': {
        'INPUT_STATE_SIZE': 2,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 2,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48 ,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.001,
        'TRAIN_FREQUENCY' : 5,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 3,
        'ACT_FUNC' : "LeakyReLU(0.1)",
        'NORM_FUNC' : "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },

'config_B': {
        'INPUT_STATE_SIZE': 2,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 2,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48 ,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.0008,
        'TRAIN_FREQUENCY' : 5,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 2,
        'ACT_FUNC' : "LeakyReLU(0.1)",
        'NORM_FUNC' : "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },

'config_B_2': {
        'INPUT_STATE_SIZE': 2,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 2,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48 ,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.0008,
        'TRAIN_FREQUENCY' : 5,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 2,
        'ACT_FUNC' : "LeakyReLU(0.1)",
        'NORM_FUNC' : "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
'config_B_3': {
        'INPUT_STATE_SIZE': 2,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 2,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48 ,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.0008,
        'TRAIN_FREQUENCY' : 5,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 3,
        'ACT_FUNC' : "LeakyReLU(0.1)",
        'NORM_FUNC' : "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
'config_B_4': {
        'INPUT_STATE_SIZE': 2,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 2,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48 ,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.0008,
        'TRAIN_FREQUENCY' : 5,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 3,
        'ACT_FUNC' : "LeakyReLU(0.1)",
        'NORM_FUNC' : "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },

'config_C': {
        'INPUT_STATE_SIZE': 2,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 2,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48 ,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.0009,
        'TRAIN_FREQUENCY' : 5,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 2,
        'ACT_FUNC' : "LeakyReLU(0.1)",
        'NORM_FUNC' : "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
'config_C_2': {
        'INPUT_STATE_SIZE': 2,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 2,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48 ,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.0009,
        'TRAIN_FREQUENCY' : 5,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 2,
        'ACT_FUNC' : "LeakyReLU(0.1)",
        'NORM_FUNC' : "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
'config_C_3': {
        'INPUT_STATE_SIZE': 2,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 2,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48 ,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.0009,
        'TRAIN_FREQUENCY' : 5,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 3,
        'ACT_FUNC' : "LeakyReLU(0.1)",
        'NORM_FUNC' : "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
'config_C_4': {
        'INPUT_STATE_SIZE': 2,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 2,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48 ,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.0009,
        'TRAIN_FREQUENCY' : 5,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 3,
        'ACT_FUNC' : "LeakyReLU(0.1)",
        'NORM_FUNC' : "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },

'config_D': {
        'INPUT_STATE_SIZE': 4,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 2,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48 ,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.0008,
        'TRAIN_FREQUENCY' : 5,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 2,
        'ACT_FUNC' : "LeakyReLU(0.1)",
        'NORM_FUNC' : "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
'config_D_2': {
        'INPUT_STATE_SIZE': 4,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 2,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48 ,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.0008,
        'TRAIN_FREQUENCY' : 5,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 2,
        'ACT_FUNC' : "LeakyReLU(0.1)",
        'NORM_FUNC' : "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
'config_D_3': {
        'INPUT_STATE_SIZE': 4,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 2,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48 ,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.0008,
        'TRAIN_FREQUENCY' : 5,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 3,
        'ACT_FUNC' : "LeakyReLU(0.1)",
        'NORM_FUNC' : "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
'config_D_4': {
        'INPUT_STATE_SIZE': 4,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 2,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48 ,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.0008,
        'TRAIN_FREQUENCY' : 5,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 3,
        'ACT_FUNC' : "LeakyReLU(0.1)",
        'NORM_FUNC' : "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
'config_E': {
        'INPUT_STATE_SIZE': 4,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 2,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48 ,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.001,
        'TRAIN_FREQUENCY' : 5,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 2,
        'ACT_FUNC' : "LeakyReLU(0.1)",
        'NORM_FUNC' : "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
'config_E_2': {
        'INPUT_STATE_SIZE': 4,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 2,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48 ,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.001,
        'TRAIN_FREQUENCY' : 5,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 2,
        'ACT_FUNC' : "LeakyReLU(0.1)",
        'NORM_FUNC' : "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
'config_E_3': {
        'INPUT_STATE_SIZE': 4,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 2,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48 ,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.001,
        'TRAIN_FREQUENCY' : 5,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 3,
        'ACT_FUNC' : "LeakyReLU(0.1)",
        'NORM_FUNC' : "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
'config_E_4': {
        'INPUT_STATE_SIZE': 4,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 2,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48 ,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.001,
        'TRAIN_FREQUENCY' : 5,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 3,
        'ACT_FUNC' : "LeakyReLU(0.1)",
        'NORM_FUNC' : "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
'config_F': {
        'INPUT_STATE_SIZE': 4,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 3,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48 ,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.0008,
        'TRAIN_FREQUENCY' : 5,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 2,
        'ACT_FUNC' : "LeakyReLU(0.1)",
        'NORM_FUNC' : "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
'config_F_2': {
        'INPUT_STATE_SIZE': 4,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 3,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48 ,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.0008,
        'TRAIN_FREQUENCY' : 5,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 2,
        'ACT_FUNC' : "LeakyReLU(0.1)",
        'NORM_FUNC' : "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
'config_F_3': {
        'INPUT_STATE_SIZE': 4,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 3,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48 ,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.0008,
        'TRAIN_FREQUENCY' : 5,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 3,
        'ACT_FUNC' : "LeakyReLU(0.1)",
        'NORM_FUNC' : "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
'config_F_4': {
        'INPUT_STATE_SIZE': 4,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 3,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48 ,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.0008,
        'TRAIN_FREQUENCY' : 5,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 3,
        'ACT_FUNC' : "LeakyReLU(0.1)",
        'NORM_FUNC' : "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
'config_G': {
        'INPUT_STATE_SIZE': 4,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 3,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48 ,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.001,
        'TRAIN_FREQUENCY' : 5,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 2,
        'ACT_FUNC' : "LeakyReLU(0.1)",
        'NORM_FUNC' : "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
'config_G_2': {
        'INPUT_STATE_SIZE': 4,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 3,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48 ,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.001,
        'TRAIN_FREQUENCY' : 5,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 2,
        'ACT_FUNC' : "LeakyReLU(0.1)",
        'NORM_FUNC' : "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
'config_G_3': {
        'INPUT_STATE_SIZE': 4,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 3,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48 ,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.001,
        'TRAIN_FREQUENCY' : 5,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 3,
        'ACT_FUNC' : "LeakyReLU(0.1)",
        'NORM_FUNC' : "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
'config_G_4': {
        'INPUT_STATE_SIZE': 4,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 3,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48 ,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.001,
        'TRAIN_FREQUENCY' : 5,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 3,
        'ACT_FUNC' : "LeakyReLU(0.1)",
        'NORM_FUNC' : "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },

    'config_A_t2': {
        'INPUT_STATE_SIZE': 2,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 2,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.001,
        'TRAIN_FREQUENCY': 2,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 2,
        'ACT_FUNC': "LeakyReLU(0.1)",
        'NORM_FUNC': "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
    'config_A_2_t2': {
        'INPUT_STATE_SIZE': 2,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 2,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.001,
        'TRAIN_FREQUENCY': 2,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 2,
        'ACT_FUNC': "LeakyReLU(0.1)",
        'NORM_FUNC': "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
    'config_A_3_t2': {
        'INPUT_STATE_SIZE': 2,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 2,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.001,
        'TRAIN_FREQUENCY': 2,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 3,
        'ACT_FUNC': "LeakyReLU(0.1)",
        'NORM_FUNC': "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
    'config_A_4_t2': {
        'INPUT_STATE_SIZE': 2,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 2,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.001,
        'TRAIN_FREQUENCY': 2,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 3,
        'ACT_FUNC': "LeakyReLU(0.1)",
        'NORM_FUNC': "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },

    'config_B_t2': {
        'INPUT_STATE_SIZE': 2,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 2,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.0008,
        'TRAIN_FREQUENCY': 2,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 2,
        'ACT_FUNC': "LeakyReLU(0.1)",
        'NORM_FUNC': "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },

    'config_B_2_t2': {
        'INPUT_STATE_SIZE': 2,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 2,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.0008,
        'TRAIN_FREQUENCY': 2,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 2,
        'ACT_FUNC': "LeakyReLU(0.1)",
        'NORM_FUNC': "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
    'config_B_3_t2': {
        'INPUT_STATE_SIZE': 2,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 2,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.0008,
        'TRAIN_FREQUENCY': 2,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 3,
        'ACT_FUNC': "LeakyReLU(0.1)",
        'NORM_FUNC': "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
    'config_B_4_t2': {
        'INPUT_STATE_SIZE': 2,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 2,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.0008,
        'TRAIN_FREQUENCY': 2,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 3,
        'ACT_FUNC': "LeakyReLU(0.1)",
        'NORM_FUNC': "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },

    'config_C_t2': {
        'INPUT_STATE_SIZE': 2,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 2,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.0009,
        'TRAIN_FREQUENCY': 2,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 2,
        'ACT_FUNC': "LeakyReLU(0.1)",
        'NORM_FUNC': "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
    'config_C_2_t2': {
        'INPUT_STATE_SIZE': 2,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 2,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.0009,
        'TRAIN_FREQUENCY': 2,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 2,
        'ACT_FUNC': "LeakyReLU(0.1)",
        'NORM_FUNC': "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
    'config_C_3_t2': {
        'INPUT_STATE_SIZE': 2,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 2,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.0009,
        'TRAIN_FREQUENCY': 2,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 3,
        'ACT_FUNC': "LeakyReLU(0.1)",
        'NORM_FUNC': "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
    'config_C_4_t2': {
        'INPUT_STATE_SIZE': 2,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 2,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.0009,
        'TRAIN_FREQUENCY': 2,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 3,
        'ACT_FUNC': "LeakyReLU(0.1)",
        'NORM_FUNC': "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },

    'config_D_t2': {
        'INPUT_STATE_SIZE': 4,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 2,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.0008,
        'TRAIN_FREQUENCY': 2,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 2,
        'ACT_FUNC': "LeakyReLU(0.1)",
        'NORM_FUNC': "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
    'config_D_2_t2': {
        'INPUT_STATE_SIZE': 4,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 2,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.0008,
        'TRAIN_FREQUENCY': 2,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 2,
        'ACT_FUNC': "LeakyReLU(0.1)",
        'NORM_FUNC': "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
    'config_D_3_t2': {
        'INPUT_STATE_SIZE': 4,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 2,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.0008,
        'TRAIN_FREQUENCY': 2,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 3,
        'ACT_FUNC': "LeakyReLU(0.1)",
        'NORM_FUNC': "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
    'config_D_4_t2': {
        'INPUT_STATE_SIZE': 4,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 2,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.0008,
        'TRAIN_FREQUENCY': 2,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 3,
        'ACT_FUNC': "LeakyReLU(0.1)",
        'NORM_FUNC': "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
    'config_E_t2': {
        'INPUT_STATE_SIZE': 4,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 2,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.001,
        'TRAIN_FREQUENCY': 2,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 2,
        'ACT_FUNC': "LeakyReLU(0.1)",
        'NORM_FUNC': "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
    'config_E_2_t2': {
        'INPUT_STATE_SIZE': 4,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 2,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.001,
        'TRAIN_FREQUENCY': 2,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 2,
        'ACT_FUNC': "LeakyReLU(0.1)",
        'NORM_FUNC': "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
    'config_E_3_t2': {
        'INPUT_STATE_SIZE': 4,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 2,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.001,
        'TRAIN_FREQUENCY': 2,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 3,
        'ACT_FUNC': "LeakyReLU(0.1)",
        'NORM_FUNC': "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
    'config_E_4_t2': {
        'INPUT_STATE_SIZE': 4,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 2,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.001,
        'TRAIN_FREQUENCY': 2,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 3,
        'ACT_FUNC': "LeakyReLU(0.1)",
        'NORM_FUNC': "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
    'config_F_t2': {
        'INPUT_STATE_SIZE': 4,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 3,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.0008,
        'TRAIN_FREQUENCY': 2,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 2,
        'ACT_FUNC': "LeakyReLU(0.1)",
        'NORM_FUNC': "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
    'config_F_2_t2': {
        'INPUT_STATE_SIZE': 4,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 3,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.0008,
        'TRAIN_FREQUENCY': 2,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 2,
        'ACT_FUNC': "LeakyReLU(0.1)",
        'NORM_FUNC': "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
    'config_F_3_t2': {
        'INPUT_STATE_SIZE': 4,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 3,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.0008,
        'TRAIN_FREQUENCY': 2,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 3,
        'ACT_FUNC': "LeakyReLU(0.1)",
        'NORM_FUNC': "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
    'config_F_4_t2': {
        'INPUT_STATE_SIZE': 4,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 3,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.0008,
        'TRAIN_FREQUENCY': 2,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 3,
        'ACT_FUNC': "LeakyReLU(0.1)",
        'NORM_FUNC': "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
    'config_G_t2': {
        'INPUT_STATE_SIZE': 4,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 3,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.001,
        'TRAIN_FREQUENCY': 2,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 2,
        'ACT_FUNC': "LeakyReLU(0.1)",
        'NORM_FUNC': "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
    'config_G_2_t2': {
        'INPUT_STATE_SIZE': 4,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 3,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.001,
        'TRAIN_FREQUENCY': 2,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 2,
        'ACT_FUNC': "LeakyReLU(0.1)",
        'NORM_FUNC': "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
    'config_G_3_t2': {
        'INPUT_STATE_SIZE': 4,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 3,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.001,
        'TRAIN_FREQUENCY': 2,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 3,
        'ACT_FUNC': "LeakyReLU(0.1)",
        'NORM_FUNC': "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
    'config_G_4_t2': {
        'INPUT_STATE_SIZE': 4,
        'OUTPUT_STATE_SIZE': 2,
        'LATENT_DIM': 3,

        'ENCODER_HIDDEN': 48,
        'ENCODER_HIDDEN2': 32,
        'ENCODER_HIDDEN3': 24,
        'ENCODER_HIDDEN4': 12,
        'ENCODER_HIDDEN5': 10,
        'ENCODER_HIDDEN6': 8,
        'ENCODER_HIDDEN7': 8,
        'ENCODER_HIDDEN8': 6,
        'DECODER_HIDDEN': 12,
        'DECODER_HIDDEN2': 24,
        'DECODER_HIDDEN3': 32,
        'DECODER_HIDDEN4': 48,
        'DECODER_HIDDEN5': 24,
        'DECODER_HIDDEN6': 3,
        'DECODER_HIDDEN7': 16,
        'DECODER_HIDDEN8': 16,

        'BETA_KL_DIV': 0.001,
        'TRAIN_FREQUENCY': 2,
        'LOSS_FUNC': "MSE_loss_feature_Standardization",
        'EPOCHS': 3,
        'ACT_FUNC': "LeakyReLU(0.1)",
        'NORM_FUNC': "LayerNorm",

        'EVAL_SEED': [1, 33, 545, 65, 6, 66, 78, 48, 24, 98],

        'VAE_Version': "VAE_Version_2.2"

    },
    # Add additional configurations as needed

}