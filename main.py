import datetime
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import utils
import CNN

# Configurable  Vars
VISUALIZE_IMG = False
SPLIT = True
HISTO = False
norm_type = "min-max"
data_augment = "None"  #
MODEL = 1  # 1, 2, 3, 4
EPOCHS = 5
LOG_DIR = utils.log_dir
MODEL_NAME = CNN.model_name
batch = 32

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

if SPLIT is True:
    # Loading Food 101 train validation and test datasets with shuffled indexes
    (food101_ds_train, food101_ds_val, food101_ds_test), \
        metadata = tfds.load('food101', split=['train[:86%]', 'validation', 'train[-14%:]'],
                             shuffle_files=True, as_supervised=True, with_info=True)
    assert isinstance(food101_ds_train, tf.data.Dataset), "Training dataset is not a TF Dataset"
    assert isinstance(food101_ds_val, tf.data.Dataset), "Validation dataset is not a TF Dataset"
    assert isinstance(food101_ds_test, tf.data.Dataset), "Test dataset is not a TF Dataset"

    get_label_name = metadata.features['label'].int2str
    num_training = tf.data.experimental.cardinality(food101_ds_train).numpy()
    print("Num training images: ", str(num_training))
    print("Num Val images: ", str(tf.data.experimental.cardinality(food101_ds_val).numpy()))
    print("Num test images: ", str(tf.data.experimental.cardinality(food101_ds_test).numpy()))

    # resizing
    food101_ds_train = food101_ds_train.map(utils.resize_dataset)
    food101_ds_val = food101_ds_val.map(utils.resize_dataset)
    for img, lbl in food101_ds_test:
        print("train size: ", img.shape, "  label: ", get_label_name(lbl))
        break

if VISUALIZE_IMG is True:
    # train_X, train_Y = tuple(zip(*food101_ds_train))
    # val_X, val_Y = tuple(zip(*food101_ds_val))
    # test_X, test_Y = tuple(zip(*food101_ds_test))

    train_X, train_Y = utils.feature_extraction(food101_ds_train)
    val_X, val_Y = utils.feature_extraction(food101_ds_val)
    test_X, test_Y = utils.feature_extraction(food101_ds_test)

    print("info.features: ", metadata.features)
    print("Split Keys: ", list(metadata.splits.keys()))
    # print("Num of Training examples 1%: ", info.splits['validation[1%:]'].num_examples)
    print("Num of Classes: ", metadata.features["label"].num_classes)
    # print("Classes: ", info.features["label"].names)

    fig2 = tfds.show_examples(food101_ds_train, metadata, rows=5)

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

    if HISTO is True:
        sample_indexes = np.random.choice(np.arange(len(train_X), dtype=int), size=10, replace=False)
        plt.figure()
        for (ii, jj) in enumerate(sample_indexes):
            plt.subplot(5, 6, ii + 1)
            plt.imshow(train_X[jj], cmap="gray")
            plt.title("Label: %d" % train_Y[jj])
        plt.show()

# todo transfer learning

# Training Models
print("Training Model ", MODEL, "  Norm Type: ", norm_type, "  Data Aug: ", data_augment)
if MODEL == 1:
    print("train type before: ", type(food101_ds_train))
    food101_ds_train = food101_ds_train.map(lambda im_t, l_t: (CNN.data_scaling_only(im_t, training=True), l_t))
    food101_ds_val = food101_ds_val.map(lambda im_v, l_v: (CNN.data_scaling_only(im_v, training=True), l_v))
    print("train type after: ", type(food101_ds_train))
    i = 0
    for img, lbl in food101_ds_train:
        i += 1
        print("Train take one: ", img.shape, int(lbl), ": ", get_label_name(lbl))
        if i > 4:
            break

    model1 = CNN.model_1(101, 128, num_training)  # num_training
    model1 = CNN.compile_model(model1, 1e-4, 0)
    history1 = model1.fit(food101_ds_train, validation_data=food101_ds_val,
                          epochs=EPOCHS, verbose=1, batch_size=batch,
                          callbacks=[CNN.early_stop, CNN.monitor_func,
                                     CNN.lr_schedule, utils.tensorboard_callback])
elif MODEL == 2:
    model2 = CNN.model_2(101, batch)
    model2 = CNN.compile_model(model2, 1e-4, 0)
    history2 = model2.fit(food101_ds_train, validation_data=food101_ds_val,
                          epochs=EPOCHS, verbose=1,
                          callbacks=[CNN.early_stop, CNN.monitor_func,
                                     CNN.lr_schedule, utils.tensorboard_callback])
elif MODEL == 3:
    model3 = CNN.model_1_data_augmented(101, 128, batch)
    model3 = CNN.compile_model(model3, 1e-4, 0)
    history3 = model3.fit(food101_ds_train, validation_data=food101_ds_val,
                          epochs=EPOCHS, verbose=1,
                          callbacks=[CNN.early_stop, CNN.monitor_func,
                                     CNN.lr_schedule, utils.tensorboard_callback])
elif MODEL == 4:
    model4 = CNN.model_2_data_augmented(101, batch)
    model4 = CNN.compile_model(model4, 1e-4, 0)
    history4 = model4.fit(food101_ds_train, validation_data=food101_ds_val,
                          epochs=EPOCHS, verbose=1,
                          callbacks=[CNN.early_stop, CNN.monitor_func,
                                     CNN.lr_schedule, utils.tensorboard_callback])
