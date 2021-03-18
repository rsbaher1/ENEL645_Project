import datetime

import tensorflow as tf
import config

model_name = config.MODEL_NAME


# Different CNN are defined below
# After determining which architecture is better (model 1 or model 2)
# TODO apply different dropouts to the best one

def model_1(num_class, k=128):

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), input_shape=(512, 512, 3), data_format="channels_last", padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(k, activation='relu'),
        tf.keras.layers.Dense(num_class, activation='softmax')
    ])

    return model


# CNN From Assignment 2
def model_2(k=101):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(48, (3, 3), input_shape=(512, 512, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(48, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Conv2D(96, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(96, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(k, activation='softmax')
    ])

    return model


def compile_model(model,  lr=1e-4, optim=0):
    if optim == 1:
        optimizer_cnn = tf.keras.optimizers.SGD(learning_rate=lr)
    else:
        optimizer_cnn = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer_cnn, loss='categorical_crossentropy',
                  metrics=["accuracy"])
    return model

