import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pickle

'''2.1: Data Processing'''

# We need to define the image size which is suppose to be 500 x 500
img_height = 500
img_width = 500
batch_size = 32 # determines how many images are processed in one batch when training

# Setting the directory paths for train and validation folder
# Notes: validation and test are different
# Validation: is used during the training process to evaluate the model's performance
#             after each training iteration (epoch)
# Test: is used to evaluate the final model's performance after training and tuning

train_directory = 'Project2Data/Data/train'
valid_directory = 'Project2Data/Data/valid'
test_directory = 'Project2Data/Data/test'

# Creating an ImageDataGenerator for Training Data

# ImageDataGenerator is used to geenrate batches of tensor image data with real-time
# data augmentation. It helps enhance the diversity of the training dataset without
# collecting more data.

train_datagen = ImageDataGenerator(
    rescale = 1./255, # normalizes the pixel values, scaling them from [0, 255] to [0, 1]
    shear_range = 0.2, # the rest are different data augementation methods
    zoom_range = 0.2,
    horizontal_flip = True,
    rotation_range = 30,
    brightness_range = [0.1, 1.0]
    )

# many of the arguments for ImageDataGenerator are used to make the training data
# more diverse and robust by manipulating the image slightly so it learns for 
# different situations

# For example, zoom_range zooms in by a certain percentage and it simulates the effect
# of an image being taken at different distances which helps the model learns to 
# recognize features at different scales

# Creating an ImageDataGenerator for Validation Data (only apply rescaling)

# the validation data is only normalized with no augmentation so we can ensure 
# that the validation results reflect the model's performance without any distortions

valid_datagen = ImageDataGenerator(rescale = 1./255)

test_datagen = ImageDataGenerator(rescale = 1./255)

# Now, we can grab the images from the directories and create image data to use

train_generator = train_datagen.flow_from_directory(
    train_directory,
    target_size = (img_height, img_width), # Resizes images to 96 x 96
    batch_size = batch_size,
    color_mode = 'grayscale', # keeps it grayscale instead of RGB
    class_mode = 'categorical' # Used for multi-class classification (3 labels)
    )

valid_generator = valid_datagen.flow_from_directory(
    valid_directory,
    target_size = (img_height, img_width),
    batch_size = batch_size,
    color_mode = 'grayscale',
    class_mode = 'categorical'
    )

test_generator = test_datagen.flow_from_directory(
    test_directory,
    target_size = (img_height, img_width),
    batch_size = batch_size,
    color_mode = 'grayscale',
    class_mode = 'categorical'
)

# Checking if the labels to see if it split correctly

print("Class indices: ", train_generator.class_indices)