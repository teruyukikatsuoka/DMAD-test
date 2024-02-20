import os

import tensorflow as tf

from forward_process import *
from dataset import *
from sample import *
from loss import *


def trainer(model, category, config):
    
    train_dataset = DatasetMaker(
        root= config.data.data_dir,
        category=category,
        config = config,
        is_train=True,
    )
    
    images = []
    labels = []
    for image, label in iter(train_dataset):
        images.append(image)
        labels.append(label)

    train_dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    train_dataset = train_dataset.shuffle(buffer_size=10 * config.data.batch_size)
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.batch(config.data.batch_size, drop_remainder=True)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    train_dataset = train_dataset.map(lambda x, y: x)
    
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    if not os.path.exists(config.model.checkpoint_dir):
        os.mkdir(config.model.checkpoint_dir)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.model.learning_rate,
            name='Adam'
        )
    )
    
    model_save_dir = os.path.join(os.getcwd(), config.model.checkpoint_dir, category+f'_{config.data.image_size}')
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    model.fit(
        train_dataset,
        epochs=config.model.epochs,
    )

    # Save the model
    if config.model.save_model:
        model_save_dir = os.path.join(os.getcwd(), config.model.checkpoint_dir, category+f'_{config.data.image_size}')
        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)
        model.save_weights(os.path.join(model_save_dir, str(config.model.epochs)+'.h5'))