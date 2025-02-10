SUITE_CONFIGS = {
    'config_A': {
    'INPUT_STATE_SIZE' : 2,
    'OUTPUT_STATE_SIZE' : 2,
    'LATENT_DIM' : 2,



    'ENCODER_HIDDEN' : 8,
    'ENCODER_HIDDEN2': 8,
    'ENCODER_HIDDEN3' : 6,
    'ENCODER_HIDDEN4' : 6,
    'DECODER_HIDDEN' : 6,
    'DECODER_HIDDEN2' : 6,
    'DECODER_HIDDEN3': 8,
    'DECODER_HIDDEN4' : 8,

    'BETA_KL_DIV': 0.0023,
    'LOSS_FUNC' : "MSE_loss_feature_Standardization",

    'EVAL_SEED' : [1, 33, 545, 65 ,6 , 66, 78, 48 , 24 , 98],

    'VAE_Version' : "VAE_Version_3.13"

    },
'config_B': {
    'INPUT_STATE_SIZE' : 2,
    'OUTPUT_STATE_SIZE' : 2,
    'LATENT_DIM' : 2,



    'ENCODER_HIDDEN' : 8,
    'ENCODER_HIDDEN2': 8,
    'ENCODER_HIDDEN3' : 6,
    'ENCODER_HIDDEN4' : 6,
    'DECODER_HIDDEN' : 6,
    'DECODER_HIDDEN2' : 6,
    'DECODER_HIDDEN3': 8,
    'DECODER_HIDDEN4' : 8,

    'BETA_KL_DIV': 0.0025,
    'LOSS_FUNC' : "MSE_loss_feature_Standardization",

    'EVAL_SEED' : [1, 33, 545, 65 ,6 , 66, 78, 48 , 24 , 98],

    'VAE_Version' : "VAE_Version_3.13"

    },
    # Add additional configurations as needed

}