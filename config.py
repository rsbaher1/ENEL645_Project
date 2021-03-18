# This is a config file for the project, it will contain all configurable variables in one place
# Once development of the code is complete, this should be the only file that needs to be edited.
import datetime
# Set to True if you want to print some information about the model and
# view sample images from the dataset
VISUALIZATION = False

# Set to True if you want to see the histogram for the dataset
HISTOGRAM = False

# Set to True to apply data Augmentation to the dataset before training
# Options:
DATA_AUGMENTATION = True

# Select the Model you want to train
# Options:
MODEL = 1

# Specify the number of epochs for training
EPOCHS = 5

# Specify the batch size for pre-processing and training
# Ex: 32, 64, 128
BATCH = 32

# Below are the Variables dependant on the Model selected
if MODEL == 1 and DATA_AUGMENTATION is False:
    # Using Model 1 without data augmentation
    MODEL_ID = 1
    LOG_DIR = "./tmp/logs/model1"
    MODEL_NAME = "cnn_model_1"
elif MODEL == 1 and DATA_AUGMENTATION is not False:
    # Using Model 1 with data augmentation
    MODEL_ID = 2
    LOG_DIR = "./tmp/logs/model1_data_aug"
    MODEL_NAME = "cnn_model_1_data_aug"
elif MODEL == 2 and DATA_AUGMENTATION is False:
    # Using Model 2 without data augmentation
    MODEL_ID = 3
    LOG_DIR = "./tmp/logs/model2"
    MODEL_NAME = "cnn_model_2"
elif MODEL == 2 and DATA_AUGMENTATION is not False:
    # Using Model 2 with data augmentation
    MODEL_ID = 4
    LOG_DIR = "./tmp/logs/model2_data_aug"
    MODEL_NAME = "cnn_model_2_data_aug"
else:
    # Error Did not Select a Model
    print("ERROR: Value entered for MODEL is incorrect. Received: ", MODEL, "\n EXPECTING: 1 or 2")
    exit(500)

LOG_FILE = LOG_DIR + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
