SUITE_CONFIGS = {
    'config_A': {
    'INPUT_STATE_SIZE' : 3,
    'OUTPUT_STATE_SIZE' : 3,
    'LATENT_DIM' : 4,

    'INPUT_DIMENSION' : INPUT_STATE_SIZE * 4,
    'OUTPUT_DIMENSION' :  OUTPUT_STATE_SIZE * 4,

    'ENCODER_HIDDEN' : 12,
    'ENCODER_HIDDEN2': 10,
    'ENCODER_HIDDEN3' : 8,
    'DECODER_HIDDEN' : 8,
    'DECODER_HIDDEN2': 10,
    'DECODER_HIDDEN3' : 12,

    'BETA_KL_DIV': 0.2,

    'EVAL_SEED' : [1, 33, 545, 65 ,6 , 66, 78, 48 , 24 , 98],

    'VAE_Version' : "VAE_Version_9"

    },
    'config_B': {
        'LEARNING_RATE': 0.0005,
        'BATCH_SIZE': 64,
        'NUM_EPOCHS': 20,
    },
    # Add additional configurations as needed
}