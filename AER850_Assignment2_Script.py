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

# We need to define the image size which is suppose to be 500 x 500, but we have to compress it so when training and testing occurs, it doesn't run for lik 12 hours
img_height = 96
img_width = 96
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

'''2.2a: Neural Network Architecture Design For FIRST MODEL'''

# Creating a sequential model where you can add each layer individually 

model1 = Sequential()

# adding layers to the model

# The convolution layer is responsible for extracting features from the input data 
# by applying various operations. This layer helps the model identify different features
# such as simple shapes within the image. Each of the filters (or kernels) detects a specific
# type of feature like an edge or a horizontal line within the input data. For this project,
# the convolution layers will use the ReLU activation function.

# The max pooling layer reduces the spatial dimensions of the input feature maps while
# retaining the important parts. Basically just makes the model more computationally efficient 
# and robust to small translations or distortions

# Batch Normalization allows the model to stabilize and also speeds up he training

model1.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (96, 96, 1), kernel_regularizer = l2(0.001)))
model1.add(BatchNormalization())
model1.add(MaxPooling2D(pool_size = (2, 2)))

# increase in filters in order to capture and learn more complex features as we go
model1.add(Conv2D(64, (3, 3), activation = 'relu', kernel_regularizer = l2(0.001)))
model1.add(BatchNormalization())
model1.add(MaxPooling2D(pool_size = (2, 2)))

model1.add(Conv2D(128, (3, 3), activation = 'relu', kernel_regularizer = l2(0.001)))
model1.add(BatchNormalization())
model1.add(MaxPooling2D(pool_size = (2, 2)))

# flatten layer converts the 2D feature maps into a 1D vector to be passed into
# the fully connected layers
model1.add(BatchNormalization())
model1.add(Flatten())

# dense layer adds a fully connected layer with 128 neurons, this where it will
# begin to connect every input from previous neurons to the ones in this layer
# and begin to learn patterns and relationships
model1.add(Dense(128, activation = 'relu'))

# dropout layer will randomly set a certain % of input units to 0 during training
# in order to prevent overfitting 
model1.add(Dropout(0.3))

# final output layer that has 3 neurons (one for each class)
# softmax activation function used to convert output into a probability distribution
# across the classes
model1.add(Dense(3, activation = 'softmax'))

# compiling the model
model1.compile(optimizer = Adam(learning_rate = 0.0005), loss = 'categorical_crossentropy', metrics = ['accuracy'])

# summary of the model which shows the number of layers, output shape of each layer, 
# and total parameters in the model
model1.summary()

early_stopping = EarlyStopping(
    monitor = "val_accuracy",  # Metric to monitor
    patience = 20,  # Number of epochs to wait for improvement
    restore_best_weights = True,  # Restore the weights of the best model
)

reduce_lr = ReduceLROnPlateau(
    monitor = "val_loss",
    factor = 0.5,
    patience = 5,
    min_lr = 1e-6
)

'''2.3a: Model Training For FIRST MODEL'''

# training the first model

# steps_per_epoch defines how many batchs to process before considering an epoch to be complete

# epochs is the number of times the model goes through the entire training set

# validation_data uses the validation generator to evaluate the model after each epoch

# validation_steps defines the number of batches to process from the validation
# generator at each epoch

trained1 = model1.fit(
    train_generator,
    epochs = 100,
    validation_data = valid_generator,
    callbacks = [early_stopping, reduce_lr]
    )

# Saving Model 1
model1.save('AER850_Project2/Model1/trained_model1_{0}.h5')

with open('AER850_Project2/Model1_history.pkl', 'wb') as file:
    pickle.dump(trained1.history, file)
    
# Opening Model 1 (if needed)
with open('AER850_Project2/Model1_history.pkl', 'rb') as f:
    history = pickle.load(f)


# plotting model 1
plt.figure(figsize = (12 ,5))

# accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label = 'Training Accuracy')
plt.plot(history['val_accuracy'], label = 'Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# loss plot
plt.subplot(1, 2, 2)
plt.plot(history['loss'], label = 'Training Loss')
plt.plot(history['val_loss'], label = 'Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

'''2.4a: Model Evaluation For FIRST MODEL'''

# model 1
test_loss, test_accuracy = model1.evaluate(test_generator, verbose = 2)

# Print the results
print("Test Loss 1:", test_loss)
print("Test Accuracy 1:", test_accuracy)