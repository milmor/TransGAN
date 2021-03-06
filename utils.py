import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import json

AUTOTUNE = tf.data.experimental.AUTOTUNE


def train_convert(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [32, 32])
    img = tf.image.random_flip_left_right(img)
    img = (img - 127.5) / 127.5 
    return img

def create_train_ds(train_dir, batch_size, seed=15):
    img_paths = tf.data.Dataset.list_files(str(train_dir))
    BUFFER_SIZE = tf.data.experimental.cardinality(img_paths)
     
    img_paths = img_paths.cache().shuffle(BUFFER_SIZE, seed=seed)
    ds = img_paths.map(train_convert, num_parallel_calls=AUTOTUNE).batch(
        batch_size, drop_remainder=True, num_parallel_calls=AUTOTUNE).prefetch(
        AUTOTUNE)
    print('Train dataset size: {}'.format(BUFFER_SIZE))  
    print('Batches: {}'.format(tf.data.experimental.cardinality(ds))) 
    return ds

def create_cifar_ds(images, batch_size, seed=15):
    BUFFER_SIZE = images.shape[0]
    images = tf.data.Dataset.from_tensor_slices(images)

    ds = images.cache().shuffle(
            BUFFER_SIZE, seed=seed).batch(batch_size, 
                drop_remainder=True, 
                num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    print('Train dataset size: {}'.format(BUFFER_SIZE))  
    print('Batches: {}'.format(tf.data.experimental.cardinality(ds))) 
    return ds

def save_hparams(hparams, model_dir, model_name):
    json_hparams = json.dumps(hparams)
    f = open(os.path.join(model_dir, '{}_hparams.json'.format(model_name)), 'w')
    f.write(json_hparams)
    f.close()
    
def generate_and_save_images(model, epoch, noise, direct, img_size=32, f_size=2.88):
    predictions = model(noise, training=False)
 
    gen_img = tf.clip_by_value(predictions[0] * 127.5 + 127.5, 0.0, 255.0)
    gen_img = tf.cast(gen_img, tf.uint8)

    fig = plt.figure(figsize=(f_size, f_size))

    for i in range(gen_img.shape[0]):
        plt.subplot(8, 8, i+1)
        plt.imshow(gen_img[i, :, :, :])
        plt.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    path = os.path.join(direct, '{:04d}.png'.format(epoch))
    plt.savefig(path)

    # Clear the current axes.
    plt.cla() 
    # Clear the current figure.
    plt.clf() 
    # Closes all the figure windows.
    plt.close('all')
    
def gradient_penalty(critic, real_samples, fake_samples):
    alpha = tf.random.uniform([real_samples.shape[0], 1, 1, 1], minval=0., maxval=1.)
    diff = fake_samples - real_samples
    interpolation = real_samples + alpha * diff

    with tf.GradientTape() as gradient_tape:
        gradient_tape.watch(interpolation)
        pred = critic(interpolation, training=True)

    gradients = gradient_tape.gradient(pred[0], [interpolation])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
    gradient_penalty = tf.reduce_mean((norm - 1.0) ** 2)
    return gradient_penalty
