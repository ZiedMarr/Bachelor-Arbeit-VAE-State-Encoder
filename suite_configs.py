SUITE_CONFIGS = {
    'config_A': {
    'INPUT_STATE_SIZE' : 4,
    'OUTPUT_STATE_SIZE' : 4,
    'LATENT_DIM' : 5,



    'ENCODER_HIDDEN' : 12,
    'ENCODER_HIDDEN2': 10,
    'ENCODER_HIDDEN3' : 8,
    'ENCODER_HIDDEN4' : 6,
    'DECODER_HIDDEN' : 6,
    'DECODER_HIDDEN2' : 8,
    'DECODER_HIDDEN3': 10,
    'DECODER_HIDDEN4' : 12,

    'BETA_KL_DIV': 0.005,
    'LOSS_FUNC' : "MSE_loss_feature_Standardization",

    'EVAL_SEED' : [1, 33, 545, 65 ,6 , 66, 78, 48 , 24 , 98],

    'VAE_Version' : "VAE_Version_3.8"

    },
'config_B': {
    'INPUT_STATE_SIZE' : 4,
    'OUTPUT_STATE_SIZE' : 4,
    'LATENT_DIM' : 5,



    'ENCODER_HIDDEN' : 12,
    'ENCODER_HIDDEN2': 10,
    'ENCODER_HIDDEN3' : 8,
    'ENCODER_HIDDEN4' : 6,
    'DECODER_HIDDEN' : 6,
    'DECODER_HIDDEN2' : 8,
    'DECODER_HIDDEN3': 10,
    'DECODER_HIDDEN4' : 12,

    'BETA_KL_DIV': 0.003,
    'LOSS_FUNC' : "MSE_loss_feature_Standardization",

    'EVAL_SEED' : [1, 33, 545, 65 ,6 , 66, 78, 48 , 24 , 98],

    'VAE_Version' : "VAE_Version_3.8"

    },

'config_C': {
    'INPUT_STATE_SIZE' : 4,
    'OUTPUT_STATE_SIZE' : 4,
    'LATENT_DIM' : 5,



    'ENCODER_HIDDEN' : 12,
    'ENCODER_HIDDEN2': 10,
    'ENCODER_HIDDEN3' : 8,
    'ENCODER_HIDDEN4' : 6,
    'DECODER_HIDDEN' : 6,
    'DECODER_HIDDEN2' : 8,
    'DECODER_HIDDEN3': 10,
    'DECODER_HIDDEN4' : 12,

    'BETA_KL_DIV': 0.007,
    'LOSS_FUNC' : "MSE_loss_feature_Standardization",

    'EVAL_SEED' : [1, 33, 545, 65 ,6 , 66, 78, 48 , 24 , 98],

    'VAE_Version' : "VAE_Version_3.8"

    },
    # Add additional configurations as needed

}