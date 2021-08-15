hparams = {'batch_size': 64,
           'noise_dim': 64,
           'g_dim': 1024, # 1024
           'g_depth': [5, 4, 2], # [5, 4, 2]
           'g_learning_rate': 0.0001,
           'g_beta_1': 0.0,
           'g_beta_2': 0.99,
           'd_dim': [192, 192], # [192, 192]
           'd_depth': [3, 3], # [3, 3]
           'd_patch_size': 2,
           'd_learning_rate': 0.0001,
           'd_beta_1': 0.0,
           'd_beta_2': 0.99,
           'd_steps': 2,
           'loss': 'bce'}
