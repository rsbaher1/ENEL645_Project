# This file contains common functions for data processing independent of the dataset and model used.
# Not all of these functions are required to be used in the project
import tensorflow as tf
import config

# Loading Tensorboard Logging dir and file
log_dir = config.LOG_DIR
log_file = config.LOG_FILE
model_name = config.MODEL_NAME

# ------- THE FUNCTIONS BELOW ARE USED IN THIS PROJECT ------- #
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

data_augmentation_flip_rotate = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)])

data_scaling_resizing = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Resizing(512, 512),
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)])

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


# ------- THE FUNCTIONS BELOW WILL NoT be USED IN THIS PROJECT ------- #
# ------- They Did not work with our dataset


# Shuffle indexes of given dataset and labels
def shuffle_indexes(X, Y):
    import numpy as np
    indexes = np.arange(X.shape[0], dtype=int)
    np.random.shuffle(indexes)
    X_new = X[indexes]
    Y_new = Y[indexes]
    return X_new, Y_new


# Splitting the given dataset (dataset_X, dataset_Y) into two portions
# dataset_X is the data values and dataset_Y are the corresponding labels
# (X_LG, Y_LG) will have the first {percent*100}% of the dataset and
# (X_SM, Y_SM) will have the last {1 - percent}*100% of the dataset
def split_dataset(dataset_X, dataset_Y, percent):
    # Calculate splitting index
    nsplit = int(percent * dataset_X.shape[0])

    # split dataset into
    X_LG = dataset_X[:nsplit]
    Y_LG = dataset_Y[:nsplit]
    X_SM = dataset_X[nsplit:]
    Y_SM = dataset_Y[nsplit:]
    return X_LG, Y_LG, X_SM, Y_SM


# Returns One Hot Encoding for given train, validate and test dataset labels or None
def one_hot_encoding(train=None, validate=None, test=None):
    train_oh = None
    validate_oh = None
    test_oh = None
    if train is not None:
        train_oh = tf.keras.utils.to_categorical(train)
    if validate is not None:
        validate_oh = tf.keras.utils.to_categorical(validate)
    if test is not None:
        test_oh = tf.keras.utils.to_categorical(test)
    return train_oh, validate_oh, test_oh


# returns normalized dataset values
# norm_type = 0 -> min-max; norm_type = 1 -> standardization
def normalise_data(train, val, test, norm_type=0):
    if norm_type == 0:
        X_train = train / 255
        X_val = val / 255
        X_test = test / 255
    else:
        train_mean, train_std = train.mean(), train.std()
        X_train = (train - train_mean) / train_std
        X_val = (val - train_mean) / train_std
        X_test = (test - train_mean) / train_std
    return X_train, X_val, X_test


# Resize Images from TFDS Dataset
def resize_dataset(img, label):
    img = tf.image.resize(img, (512, 512))
    return img, label


# Extract Images and Labels in Dataset
def feature_extraction(ds):
    img = []
    lbl = []
    for i in ds:
        img.append(i[0])
        lbl.append(i[1])
    return img, lbl


# Normalizes Images in Dataset using min-max method
def dataset_normalization_min_max(img, lbl):
    img = img / 255
    return img, lbl


# Normalizes Images in Dataset using Standardization method
def dataset_normalization_std(img, lbl):
    mean, std = img.mean(), img.std()
    img = (img - mean) / std
    return img, lbl


# Augment Images in Dataset using random flip method
def dataset_augmentation_flip(img, lbl):
    return tf.image.random_flip_left_right(img), lbl


# Augment Images in Dataset using random contrast method
def dataset_augmentation_contrast(img, lbl):
    return tf.image.random_contrast(img, lower=0.0, upper=1.0), lbl
