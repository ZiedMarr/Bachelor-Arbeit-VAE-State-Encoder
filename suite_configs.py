SUITE_CONFIGS = {
    'config_A': {
    'INPUT_STATE_SIZE' : 3,
    'OUTPUT_STATE_SIZE' : 3,
    'LATENT_DIM' : 2,



    'ENCODER_HIDDEN' : 12,
    'ENCODER_HIDDEN2': 10,
    'ENCODER_HIDDEN3' : 8,
    'ENCODER_HIDDEN4' : 6,
    'DECODER_HIDDEN' : 6,
    'DECODER_HIDDEN2' : 8,
    'DECODER_HIDDEN3': 10,
    'DECODER_HIDDEN4' : 12,

    'BETA_KL_DIV': 0.01,
    'LOSS_FUNC' : "MSE_loss_feature_Standardization",

    'EVAL_SEED' : [1, 33, 545, 65 ,6 , 66, 78, 48 , 24 , 98],

    'VAE_Version' : "VAE_Version_3.5"

    },
    'config_B': {
    'INPUT_STATE_SIZE' : 3,
    'OUTPUT_STATE_SIZE' : 3,
    'LATENT_DIM' : 2,



    'ENCODER_HIDDEN' : 12,
    'ENCODER_HIDDEN2': 10,
    'ENCODER_HIDDEN3' : 8,
    'ENCODER_HIDDEN4' : 6,
    'DECODER_HIDDEN' : 6,
    'DECODER_HIDDEN2' : 8,
    'DECODER_HIDDEN3': 10,
    'DECODER_HIDDEN4' : 12,

    'BETA_KL_DIV': 0.015,
    'LOSS_FUNC' : "MSE_loss_feature_Standardization",

    'EVAL_SEED' : [1, 33, 545, 65 ,6 , 66, 78, 48 , 24 , 98],

    'VAE_Version' : "VAE_Version_3.5"
    },
'config_C': {
    'INPUT_STATE_SIZE' : 3,
    'OUTPUT_STATE_SIZE' : 3,
    'LATENT_DIM' : 2,



    'ENCODER_HIDDEN' : 12,
    'ENCODER_HIDDEN2': 10,
    'ENCODER_HIDDEN3' : 8,
    'ENCODER_HIDDEN4' : 6,
    'DECODER_HIDDEN' : 6,
    'DECODER_HIDDEN2' : 8,
    'DECODER_HIDDEN3': 10,
    'DECODER_HIDDEN4' : 12,

    'BETA_KL_DIV': 0.02,
    'LOSS_FUNC' : "MSE_loss_feature_Standardization",

    'EVAL_SEED' : [1, 33, 545, 65 ,6 , 66, 78, 48 , 24 , 98],

    'VAE_Version' : "VAE_Version_3.5"
    }
    # Add additional configurations as needed

}