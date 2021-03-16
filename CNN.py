import datetime

import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import sklearn as skl
import utils

model_name = "cnn_model_1"


# model_name = "cnn_model_2"


data_preprocessing =  tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Resizing(512, 512),
        tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)])


data_scaling_only = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Resizing(512, 512),
        tf.keras.layers.experimental.preprocessing.Rescaling(1./255)])


# Define CNN model here. Include accuracy in the metrics list when you compile it
# Experiment with different network architectures, learnig rates, dropout, etc.
def model_1(num_class, k=128, batch=32):

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), input_shape=(None, 512, 512, 3), data_format="channels_last", padding='same', activation='relu'),
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


def model_2(k=101, batch=32):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(48, (3, 3), input_shape=(batch, 512, 512, 3), padding='same', activation='relu'),
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


early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

monitor_func = tf.keras.callbacks.ModelCheckpoint(model_name, monitor='val_loss',
                                                  verbose=0, save_best_only=True,
                                                  save_weights_only=True, mode='min')

# Learning rate schedule
def scheduler(epoch, lr):
    if epoch % 10 == 0:
        lr = lr / 2
    return lr


lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)
