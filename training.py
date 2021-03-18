
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import utils
import CNN
import config

# Load Variables from Configuration File
VISUALIZE_IMG = config.VISUALIZATION
HISTOGRAM = config.HISTOGRAM
norm_type = "min-max"
data_augment = config.DATA_AUGMENTATION
MODEL = config.MODEL_ID
EPOCHS = config.EPOCHS
LOG_DIR = config.LOG_DIR
LOG_FILE = config.LOG_FILE
MODEL_NAME = config.MODEL_NAME
batch = config.BATCH
num_classes = 101
# Get GPU Working with CuDNN
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# ---------- LOADING Dataset ---------- #
# Loading Food 101 train validation and test datasets with shuffled indexes  'validation',
(food101_ds_train, food101_ds_val, food101_ds_test), metadata = tfds.load('food101',
                                                                          split=['train[:86%]', 'validation', 'train[-14%:]'],
                                                                          shuffle_files=True, as_supervised=True, with_info=True)
print("Split Keys: ", list(metadata.splits.keys()))
print("info.features: ", metadata.features)
print("train type before: ", type(food101_ds_train))
print("Num of Classes: ", metadata.features["label"].num_classes)
print("Lengths: ", len(food101_ds_train), len(food101_ds_val), len(food101_ds_test))
exit(0)
assert isinstance(food101_ds_train, tf.data.Dataset), "Training dataset is not a TF Dataset"
assert isinstance(food101_ds_val, tf.data.Dataset), "Validation dataset is not a TF Dataset"
assert isinstance(food101_ds_test, tf.data.Dataset), "Test dataset is not a TF Dataset"

get_label_name = metadata.features['label'].int2str
num_training = tf.data.experimental.cardinality(food101_ds_train).numpy()
print("Num training images: ", str(num_training))
print("Num Val images: ", str(tf.data.experimental.cardinality(food101_ds_val).numpy()))
print("Num test images: ", str(tf.data.experimental.cardinality(food101_ds_test).numpy()))

# ---------- Pre-Processing Dataset ---------- #
# Resizing training and validation datasets to be (512, 512, 3)
# Applying One-Hot encoding to Labels
# DO NOT APPLY THIS TO THE TESTING SET
print("train type before: ", type(food101_ds_train))
food101_ds_train = food101_ds_train.map(lambda im_t, l_t: (utils.data_scaling_resizing(im_t, training=True), l_t))
food101_ds_train = food101_ds_train.map(lambda im_t, l_t: (im_t, tf.one_hot(l_t,depth=101)))
food101_ds_val = food101_ds_val.map(lambda im_v, l_v: (utils.data_scaling_resizing(im_v, training=True), l_v))
food101_ds_val = food101_ds_val.map(lambda im_v, l_v: (im_v, tf.one_hot(l_v,depth=101)))
food101_ds_train = food101_ds_train.batch(batch)
food101_ds_val = food101_ds_val.batch(batch)
print("train type after: ", type(food101_ds_train))

i = 0
for img, lbl in food101_ds_train:
    i += 1
    print("Train take one: ", img.shape, " lbl shape: ", lbl.shape, " lbl class: ", type(lbl) )
    if i > 4:
        break

# ---------- Transfer Learning ---------- #
# TODO Transfer Learning


# ---------- Training Models ---------- #
print("Training Model ", MODEL, "  Norm Type: ", norm_type, "  Data Aug: ", data_augment)
k = 128
optimizer = 0
learning_rate = 1e-4
if MODEL == 1:
    model1 = CNN.model_1(num_classes, k)  
    model1 = CNN.compile_model(model1, learning_rate, optimizer)
    history1 = model1.fit(food101_ds_train, validation_data=food101_ds_val,
                          epochs=EPOCHS, verbose=1, batch_size=batch,
                          callbacks=[utils.early_stop, utils.monitor_func,
                                     utils.lr_schedule, utils.tensorboard_callback])
elif MODEL == 2:
    food101_ds_train = food101_ds_train.map(lambda im_t, l_t: (utils.data_augmentation_flip_rotate(im_t, training=True), l_t))
    food101_ds_val = food101_ds_val.map(lambda im_v, l_v: (utils.data_augmentation_flip_rotate(im_v, training=True), l_v))

    model2 = CNN.model_1(num_classes, k)
    model2 = CNN.compile_model(model2, learning_rate, optimizer)
    history2 = model2.fit(food101_ds_train, validation_data=food101_ds_val,
                          epochs=EPOCHS, verbose=1,
                          callbacks=[utils.early_stop, utils.monitor_func,
                                     utils.lr_schedule, utils.tensorboard_callback])
elif MODEL == 3:
    model3 = CNN.model_2(num_classes)
    model3 = CNN.compile_model(model3, learning_rate, optimizer)
    history3 = model3.fit(food101_ds_train, validation_data=food101_ds_val,
                          epochs=EPOCHS, verbose=1,
                          callbacks=[utils.early_stop, utils.monitor_func,
                                     utils.lr_schedule, utils.tensorboard_callback])
elif MODEL == 4:
    food101_ds_train = food101_ds_train.map(lambda im_t, l_t: (utils.data_augmentation_flip_rotate(im_t, training=True), l_t))
    food101_ds_val = food101_ds_val.map(lambda im_v, l_v: (utils.data_augmentation_flip_rotate(im_v, training=True), l_v))

    model4 = CNN.model_2(num_classes)
    model4 = CNN.compile_model(model4, learning_rate, optimizer)
    history4 = model4.fit(food101_ds_train, validation_data=food101_ds_val,
                          epochs=EPOCHS, verbose=1,
                          callbacks=[utils.early_stop, utils.monitor_func,
                                     utils.lr_schedule, utils.tensorboard_callback])

# TensorBoard Logging ------- Optional
writer = tf.summary.create_file_writer(LOG_DIR)


# ---------- Visualizing Dataset ---------- #
if VISUALIZE_IMG is True:
    # Method 1 of splitting Dataset to Images and Labels
    # train_X, train_Y = tuple(zip(*food101_ds_train))
    # val_X, val_Y = tuple(zip(*food101_ds_val))
    # test_X, test_Y = tuple(zip(*food101_ds_test))

    # Method 2 of Splitting ataset to Images and Labels
    train_X, train_Y = utils.feature_extraction(food101_ds_train)
    val_X, val_Y = utils.feature_extraction(food101_ds_val)
    test_X, test_Y = utils.feature_extraction(food101_ds_test)

    # Printing some information about the Dataset
    print("info.features: ", metadata.features)
    print("Split Keys: ", list(metadata.splits.keys()))
    # print("Num of Training examples 1%: ", info.splits['validation[1%:]'].num_examples)
    print("Num of Classes: ", metadata.features["label"].num_classes)
    # print("Classes: ", info.features["label"].names)

    # Displaying Figure of samples from Dataset
    fig2 = tfds.show_examples(food101_ds_train, metadata, rows=5)

    # Displaying Figure of samples from Dataset using PyPlot
    i = 0
    plt.figure(1)
    for img, lbl in food101_ds_train:
        i = i + 1
        print("Train take one: ", img.shape, int(lbl))
        ax = plt.subplot(2, 2, i)
        plt.imshow(img.numpy().astype("uint8"))
        ax.set_title(get_label_name(lbl))
        plt.axis("off")
        if i >= 4:
            break

    # Displaying Figure of samples from Dataset using PyPlot
    # and using the Images and Labels dataset instead of the combined one (for comparison)
    i = 0
    plt.figure(2)
    for item in train_X:
        i = i + 1
        print("Train img: ", item.shape)
        ax = plt.subplot(2, 2, i)
        plt.imshow(item.numpy().astype("uint8"))
        plt.axis("off")
        if i >= 4:
            break

    i = 0
    for item in train_Y:
        i = i + 1
        print("Train lbl: ", int(item), "  name: ", get_label_name(item))
        if i >= 4:
            break
    plt.show()

    # Displaying the Histogram
    if HISTOGRAM is True:
        sample_indexes = np.random.choice(np.arange(len(train_X), dtype=int), size=10, replace=False)
        plt.figure()
        for (ii, jj) in enumerate(sample_indexes):
            plt.subplot(5, 6, ii + 1)
            plt.imshow(train_X[jj], cmap="gray")
            plt.title("Label: %d" % train_Y[jj])
        plt.show()




