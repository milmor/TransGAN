'''
Author: Emilio Morales (mil.mor.mor@gmail.com)
        Jun 2021
'''
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable tensorflow debugging logs
import tensorflow as tf
import time
from model import Generator, Discriminator
from utils import *
from hparams import hparams


def run_training(args): 
    print('\n##############')
    print('TransGAN Train')
    print('##############\n')
    dataset_path = args.dataset_path
    model_name = args.model_name
    main_dir = args.main_dir
    ckpt_interval = args.ckpt_interval
    max_ckpt_to_keep = args.max_ckpt_to_keep
    epochs = args.epochs
    train_seed = args.train_seed
    test_seed = args.test_seed
    
    # Create dirs
    os.makedirs(main_dir, exist_ok=True)

    model_dir = os.path.join(main_dir, model_name)
    log_dir = os.path.join(model_dir, 'log-dir')

    writer = tf.summary.create_file_writer(log_dir)

    gen_test_dir = os.path.join(model_dir, 'test-gen')
    os.makedirs(gen_test_dir, exist_ok=True)

    # Define model
    generator = Generator(model_dim=hparams['g_dim'], 
                          noise_dim=hparams['noise_dim'],
                          depth=hparams['g_depth'],
                          heads=hparams['g_heads'],
                          mlp_dim=hparams['g_mlp'],
                          initializer=hparams['g_initializer'])
    discriminator = Discriminator(model_dim=hparams['d_dim'], 
                                  depth=hparams['d_depth'],
                                  heads=hparams['d_heads'],
                                  mlp_dim=hparams['d_mlp'],
                                  initializer=hparams['d_initializer'],
                                  patch_size=hparams['d_patch_size'],
                                  policy=hparams['policy'])
    
    generator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=hparams['g_learning_rate'], 
        beta_1=hparams['g_beta_1'], 
        beta_2=hparams['g_beta_2'])
    discriminator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=hparams['d_learning_rate'], 
        beta_1=hparams['d_beta_1'],
        beta_2=hparams['d_beta_2'])

    # Create/Load checkpoint
    checkpoint_dir = os.path.join(model_dir, 'training-checkpoints')
    ckpt = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                               discriminator_optimizer=discriminator_optimizer,
                               generator=generator,
                               discriminator=discriminator,
                               epoch=tf.Variable(0))

    ckpt_manager = tf.train.CheckpointManager(ckpt, directory=checkpoint_dir, 
                                              max_to_keep=max_ckpt_to_keep)
    ckpt.restore(ckpt_manager.latest_checkpoint)
    
    if ckpt_manager.latest_checkpoint:
        print('Restored {} from: {}\n'.format(model_name, ckpt_manager.latest_checkpoint))
    else:
        print('Initializing {} from scratch\n'.format(model_name))
        save_hparams(hparams, model_dir, model_name)
    for key, value in hparams.items():
        print(key, ': ', value)
    print('\n')
    
    # Dataset stup
    if dataset_path == 'CIFAR-10':
        (train_images, _), (_, _) = tf.keras.datasets.cifar10.load_data()
        train_images = train_images.astype('float32')
        train_images = (train_images - 127.5) / 127.5 
        train_dataset = create_cifar_ds(train_images, hparams['batch_size'], seed=train_seed)
    else:
        train_dataset = create_train_ds(dataset_path, hparams['batch_size'], seed=train_seed)
              
    if hparams['loss'] == 'bce':
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        def discriminator_loss(real_img, fake_img):
            real_loss = cross_entropy(tf.ones_like(real_img), real_img)
            fake_loss = cross_entropy(tf.zeros_like(fake_img), fake_img)
            return real_loss + fake_loss

        def generator_loss(fake_img):
            return cross_entropy(tf.ones_like(fake_img), fake_img)

    elif hparams['loss'] == 'hinge':
        def d_real_loss(logits):
            return tf.reduce_mean(tf.nn.relu(1.0 - logits))

        def d_fake_loss(logits):
            return tf.reduce_mean(tf.nn.relu(1.0 + logits))

        def discriminator_loss(real_img, fake_img):
            real_loss = d_real_loss(real_img)
            fake_loss = d_fake_loss(fake_img)
            return fake_loss + real_loss

        def generator_loss(fake_img):
            return -tf.reduce_mean(fake_img)
            
    elif hparams['loss'] == 'wgan':
        def discriminator_loss(real_img, fake_img):
            real_loss = tf.reduce_mean(real_img)
            fake_loss = tf.reduce_mean(fake_img)
            return fake_loss - real_loss + (tf.reduce_mean(real_img) ** 2) * 1e-3

        def generator_loss(fake_img):
            return -tf.reduce_mean(fake_img)

    gen_loss_avg = tf.keras.metrics.Mean()
    disc_loss_avg = tf.keras.metrics.Mean()
    gp_avg = tf.keras.metrics.Mean()

    @tf.function
    def train_step(real_images):
        noise = tf.random.normal([hparams['batch_size'], hparams['noise_dim']])

        # Train the discriminator
        for _ in range(hparams['d_steps']):
            with tf.GradientTape() as disc_tape:
                generator_output = generator(noise, training=True)
                real_disc_output = discriminator(real_images, training=True)
                fake_disc_output = discriminator(generator_output[0], training=True)
                d_cost = discriminator_loss(real_disc_output[0], fake_disc_output[0])  
                if hparams['loss'] == 'wgan':
                    gp = gradient_penalty(
                        discriminator, real_images, 
                        generator_output[0]) * hparams['gp_weight']
                else:
                    gp = 0.0
                disc_loss = d_cost + gp 

            disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            disc_gradients, _ = tf.clip_by_global_norm(disc_gradients, 5.0)
            discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables)) 
            disc_loss_avg(d_cost)
            gp_avg(gp)

        noise = tf.random.normal([hparams['batch_size'], hparams['noise_dim']])

        # Train the generator
        with tf.GradientTape() as gen_tape:
            generator_output = generator(noise, training=True)
            fake_disc_output = discriminator(generator_output[0], training=True)
            gen_loss = generator_loss(fake_disc_output[0])

        gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gen_gradients, _ = tf.clip_by_global_norm(gen_gradients, 5.0)
        generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
        gen_loss_avg(gen_loss)

    # n examples to plot with generate_and_save_images()
    num_examples_to_generate = args.n_plot_images 
    # noise_seed to plot with generate_and_save_images()
    noise_seed = tf.random.normal([num_examples_to_generate, 
                                  hparams['noise_dim']], seed=test_seed) 
    writer = tf.summary.create_file_writer(log_dir)

    for _ in range(int(ckpt.epoch), epochs):
        start = time.time()
        step_int = int(ckpt.epoch)
        # Clear metrics
        gen_loss_avg.reset_states()
        disc_loss_avg.reset_states()
        gp_avg.reset_states()

        # Run epoch
        for image_batch in train_dataset:
            train_step(image_batch)

        # Print and save Tensorboard
        print('\nTime for epoch {} is {} sec'.format(step_int, time.time()-start))
        print('Generator loss: {:.4f}'.format(gen_loss_avg.result()))
        print('Discriminator loss: {:.4f}'.format(disc_loss_avg.result()))  
        print('GP: {:.4f}'.format(gp_avg.result()))  
        with writer.as_default():
            tf.summary.scalar('generator_loss', gen_loss_avg.result(), step=step_int)
            tf.summary.scalar('discriminator_loss', disc_loss_avg.result(), step=step_int)
            tf.summary.scalar('gp', gp_avg.result(), step=step_int)

        # Generate and save test images plot
        generate_and_save_images(generator, step_int, noise_seed, gen_test_dir)

        # Save checkpoint
        if (step_int) % ckpt_interval == 0:
            ckpt_manager.save(step_int)
            print('Checkpoint saved at epoch {}'.format(step_int)) 
        ckpt.epoch.assign_add(1)
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='CIFAR-10')
    parser.add_argument('--model_name', default='model')
    parser.add_argument('--main_dir', default='logs-TransGAN')
    parser.add_argument('--ckpt_interval', type=int, default=5)
    parser.add_argument('--max_ckpt_to_keep', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--train_seed', type=int, default=15)    
    parser.add_argument('--test_seed', type=int, default=15)    
    parser.add_argument('--n_plot_images', type=int, default=64) 
    args = parser.parse_args()

    run_training(args)


if __name__ == '__main__':
    main()
