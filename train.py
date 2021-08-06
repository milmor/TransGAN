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
from diffaug import DiffAugment
from utils import *
from hparams import hparams


def run_training(args): 
    model_name = args.model_name
    main_dir = args.main_dir
    ckpt_interval = args.ckpt_interval
    max_ckpt_to_keep = args.max_ckpt_to_keep
    epochs = args.epochs
    
    generator = Generator(hparams['g_dim'], 
                          noise_dim=hparams['noise_dim'],
                          depth=hparams['g_depth'])
    discriminator = Discriminator(hparams['d_dim'], 
                                  noise_dim=hparams['noise_dim'],
                                  depth=hparams['d_depth'])
    
    generator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=hparams['g_learning_rate'], 
        beta_1=hparams['g_beta_1'], 
        beta_2=hparams['g_beta_2'])
    discriminator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=hparams['d_learning_rate'], 
       beta_1=hparams['d_beta_1'],
       beta_2=hparams['d_beta_2'])

    os.makedirs(main_dir, exist_ok=True)

    model_dir = os.path.join(main_dir, model_name)
    log_dir = os.path.join(model_dir, 'log-dir')

    writer = tf.summary.create_file_writer(log_dir)

    gen_train_dir = os.path.join(model_dir, 'train-gen')
    gen_test_dir = os.path.join(model_dir, 'test-gen')
    os.makedirs(gen_train_dir, exist_ok=True)
    os.makedirs(gen_test_dir, exist_ok=True)

    checkpoint_dir = os.path.join(model_dir, 'training-checkpoints')
    ckpt = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                               discriminator_optimizer=discriminator_optimizer,
                               generator=generator,
                               discriminator=discriminator,
                               epoch=tf.Variable(0))

    ckpt_manager = tf.train.CheckpointManager(ckpt, directory=checkpoint_dir, 
                                              max_to_keep=max_ckpt_to_keep)
    
    print('\n##############')
    print('TransGAN Train')
    print('##############\n')
    if ckpt_manager.latest_checkpoint:
        print('Restored {} from: {}'.format(model_name, ckpt_manager.latest_checkpoint))
    else:
        print('Initializing {} from scratch'.format(model_name))
    save_hparams(model_dir, model_name)
    
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')
    train_images = (train_images - 127.5) / 127.5 
    test_images = (test_images - 127.5) / 127.5 
    train_dataset = create_ds(train_images, train_labels, hparams['batch_size'])
    test_dataset = create_ds(test_images, test_labels, hparams['batch_size'])

    train_batch = next(iter(train_dataset))
    test_batch = next(iter(test_dataset))
    
    gen_loss_avg = tf.keras.metrics.Mean()
    disc_loss_avg = tf.keras.metrics.Mean()
    
    @tf.function
    def train_step(real_images, labels):
        noise = tf.random.normal([hparams['batch_size'], hparams['noise_dim']])

        # Train the discriminator
        for i in range(hparams['d_steps']):
            with tf.GradientTape() as disc_tape:
                generator_output = generator(labels, noise, training=True)
                aug_real_img = DiffAugment(real_images, policy)
                aug_gen_img = DiffAugment(generator_output[0], policy)
                real_disc_output = discriminator(aug_real_img,  labels, training=True)
                fake_disc_output = discriminator(aug_gen_img, labels, training=True)

                d_cost = discriminator_loss(real_disc_output[0], fake_disc_output[0])  
                disc_loss = d_cost

            disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables)) 

        noise = tf.random.normal([hparams['batch_size'], hparams['noise_dim']])

        # Train the generator
        with tf.GradientTape() as gen_tape:
            generator_output = generator(labels, noise, training=True)
            generator_output[0] = DiffAugment(generator_output[0], policy)
            aug_gen_img = DiffAugment(generator_output[0])
            fake_disc_output = discriminator(aug_gen_img, labels, training=True)

            gen_loss = generator_loss(fake_disc_output[0])

        gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))

        # Update metrics
        gen_loss_avg(gen_loss)
        disc_loss_avg(d_cost)
    
    if hparams['loss'] == 'hinge':
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
        
    num_examples_to_generate = 16
    seed = tf.random.normal([num_examples_to_generate, hparams['noise_dim']])
    policy = 'color,translation,cutout' 
    print('Total batches: {}'.format(tf.data.experimental.cardinality(train_dataset).numpy()))
    
    for _ in range(epochs):
        start = time.time()
        step_int = int(ckpt.epoch)
        # Clear metrics
        gen_loss_avg.reset_states()
        disc_loss_avg.reset_states()

        for image_batch, label_batch in train_dataset:
            loss = train_step(image_batch, label_batch)

        # Print and save Tensorboard
        print('\nTime for epoch {} is {} sec'.format(step_int, time.time()-start))
        print('Generator loss: {:.4f}'.format(gen_loss_avg.result()))
        print('Discriminator loss: {:.4f}'.format(disc_loss_avg.result()))  
        with writer.as_default():
            tf.summary.scalar('generator_loss', gen_loss_avg.result(), step=step_int)
            tf.summary.scalar('discriminator_loss', disc_loss_avg.result(), step=step_int)

        # Generate and save train/test images  
        generate_and_save_images(generator, step_int,
                                 train_batch[1][:num_examples_to_generate],
                                 seed, gen_train_dir)
        generate_and_save_images(generator, step_int,
                                 test_batch[1][:num_examples_to_generate],
                                 seed, gen_test_dir)

        # Save checkpoint
        if (step_int) % ckpt_interval == 0:
            ckpt_manager.save(step_int)
        ckpt.epoch.assign_add(1)
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='model')
    parser.add_argument('--main_dir', default='TransGAN')
    parser.add_argument('--ckpt_interval', type=int, default=3)
    parser.add_argument('--max_ckpt_to_keep', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=3000)
    
    args = parser.parse_args()

    run_training(args)


if __name__ == '__main__':
    main()
