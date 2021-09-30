# Small-TransGAN hyperparameters
hparams = {'batch_size': 64,
           'noise_dim': 64, # Default: 256
           'g_dim': 64, # Default: 1024
           'g_depth': [1, 1, 1], # Default: [5, 4, 2]
           'g_heads': [4, 4, 4], # Default: [4, 4, 4]
           'g_mlp': [512, 512, 512], # Default: [4096, 1024, 256]
           'g_initializer': 'orthogonal', # Default: 'glorot_uniform'
           'g_learning_rate': 0.0001,
           'g_beta_1': 0.0,
           'g_beta_2': 0.99,
           'd_dim': [192, 64], # Default: [192, 192]
           'd_depth': [2, 2], # Default: [3, 3]
           'd_heads': [4, 4, 4], # Default: [4, 4, 4]
           'd_mlp': [768, 1024, 1024], # Default: [768, 1536, 1536]
           'd_initializer': 'orthogonal', # Default: 'glorot_uniform'
           'd_patch_size': 4, # Default: 2
           'd_learning_rate': 0.0001,
           'd_beta_1': 0.0,
           'd_beta_2': 0.99,
           'd_steps': 1, # Default: 1
           'loss': 'wgan', # Loss types: ('bce', 'hinge', 'wgan') Default: 'wgan'
           'gp_weight': 10.0, # Default: 10.0
           'policy': 'color,translation,cutout'}
