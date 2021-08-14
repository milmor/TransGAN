import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from hparams import hparams

AUTOTUNE = tf.data.experimental.AUTOTUNE


def create_ds(images, labels, batch_size, seed=15):
    BUFFER_SIZE = images.shape[0]
    img_ds = tf.data.Dataset.from_tensor_slices(images)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    print('Total images: {}'.format(images.shape[0]))
    ds = tf.data.Dataset.zip(
    (img_ds, label_ds)).cache().shuffle(
        BUFFER_SIZE, seed=seed).batch(
        batch_size, drop_remainder=True, num_parallel_calls=AUTOTUNE).prefetch(
        AUTOTUNE
    )
    return ds

def save_hparams(model_dir, name):
    json_hparams = json.dumps(hparams)
    f = open(os.path.join(model_dir, '{}_hparams.json'.format(name)), 'w')
    f.write(json_hparams)
    f.close()
    
def generate_and_save_images(model, epoch, test_label, noise, direct):
    predictions = model(test_label, noise, training=False)

    fig = plt.figure(figsize=(5, 5))
    predictions = tf.cast(predictions[0] * 127.5 + 127.5, tf.uint8)
    for i in range(predictions.shape[0]):
        plt.subplot(5, 5, i+1)
        plt.imshow(predictions[i, :, :, :])
        plt.axis('off')
    path = os.path.join(direct, '{:04d}.png'.format(epoch))
    plt.savefig(path)
    # Clear the current axes.
    plt.cla() 
    # Clear the current figure.
    plt.clf() 
    # Closes all the figure windows.
    plt.close('all')
