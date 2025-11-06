'''AER850: Project 2 — Model 1 and Model 2 (500x500 RGB configration)
Steps 2.1 to 2.4 (Data → Architecture → Training → Evaluation)'''

'''Importing Necessary Libraries'''
import tensorflow as tf
import random
import numpy as np
import os

'''Reproducibility Setup'''

# Set all random seeds for reproducibility
def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# Set environment variables for TensorFlow reproducibility
def set_tf_options():
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    # For GPU - if you're using one
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Apply all reproducibility settings
set_seeds(42)
set_tf_options()

# Continue with existing imports
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pickle
import os

'''2.1: Data Processing'''

# We need to define the image size which was originally 500 x 500. 
# The project requires (500, 500, 3) input shape for the model.
img_height = 500
img_width = 500
batch_size = 32  # Determines how many images are processed in one batch
epochs = 20      # Max number of times the model will go through the entire dataset

# Setting the directory paths for train, validation, and test datasets.
# Validation data is used during training to check model performance after each epoch.
# Test data is used at the very end to evaluate the final trained model.
train_directory = 'Project2Data/Data/train'
valid_directory = 'Project2Data/Data/valid'
test_directory = 'Project2Data/Data/test'

# Creating ImageDataGenerator for training data.
# The goal is to apply real-time data augmentation which improves model robustness
# without needing more raw data. Each transformation creates slightly different versions
# of existing images, helping the model generalize better.

train_datagen = ImageDataGenerator(
    rescale=1./255,             # Normalizes pixel values to [0, 1]
    shear_range=0.2,            # Applies shearing transformations
    zoom_range=0.2,             # Random zoom-in and zoom-out
    rotation_range=30,          # Randomly rotates images up to ±30°
    brightness_range=[0.1, 1.0],# Adjusts brightness randomly
    horizontal_flip=True,       # Randomly flips images horizontally
    fill_mode='nearest'         # Fill missing pixels with nearest values
)

# Validation and test data should not be augmented — they must reflect real-world conditions.
# Therefore, we only rescale them to normalize pixel values.
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Now we use flow_from_directory to load images directly from folders.
# This function automatically assigns labels based on folder names.
# Added seed=42 for reproducible shuffling
train_generator = train_datagen.flow_from_directory(
    train_directory,
    target_size=(img_height, img_width),   # Resize to 500x500
    batch_size=batch_size,
    color_mode='rgb',                # RGB input as required
    class_mode='categorical',        # Multi-class classification (3 labels)
    seed=42,                         # For reproducible shuffling
    shuffle=True                     # Shuffle training data
)

valid_generator = valid_datagen.flow_from_directory(
    valid_directory,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='categorical',
    seed=42,                         # For reproducible shuffling
    shuffle=False                    # No need to shuffle validation data
)

test_generator = test_datagen.flow_from_directory(
    test_directory,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='categorical',
    seed=42,                         # For reproducible shuffling
    shuffle=False                    # No need to shuffle test data
)

# Print out the detected classes to verify that directory structure was read correctly.
print("Class indices:", train_generator.class_indices)

# Create output folders if not already existing
os.makedirs('AER850_Project2/Model1', exist_ok=True)
os.makedirs('AER850_Project2/Model2', exist_ok=True)


'''2.2a/2.3a: Neural Network Architecture Design/Hyperparamters – MODEL 1'''

# Model 1 acts as the baseline CNN. It is moderately deep and contains three
# convolutional blocks to capture essential spatial features. Batch normalization
# stabilizes training, while dropout helps reduce overfitting.

model1 = Sequential()

# --- Convolutional Layers ---
# Each Conv2D layer extracts features using filters (kernels).
# These filters learn to detect edges, corners, and textures.
model1.add(Conv2D(32, (3, 3), activation='relu', input_shape=(500, 500, 3), kernel_regularizer=l2(0.001)))
model1.add(BatchNormalization())
model1.add(MaxPooling2D(pool_size=(2, 2)))

# Increasing the number of filters allows the model to detect more complex patterns.
model1.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)))
model1.add(BatchNormalization())
model1.add(MaxPooling2D(pool_size=(2, 2)))

model1.add(Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)))
model1.add(BatchNormalization())
model1.add(MaxPooling2D(pool_size=(2, 2)))

# --- Fully Connected Layers ---
# Flatten transforms 2D feature maps into 1D vectors for dense layers.
model1.add(BatchNormalization())
model1.add(Flatten())

# Dense layer learns high-level relationships between extracted features.
model1.add(Dense(128, activation='relu'))

# Dropout randomly disables neurons during training to reduce overfitting.
model1.add(Dropout(0.3))

# Final layer — 3 neurons for the three output classes (crack, paint off, missing screw).
# Softmax converts outputs to probabilities.
model1.add(Dense(3, activation='softmax'))

# Compile the model — defines optimizer, loss function, and evaluation metrics.
model1.compile(optimizer=Adam(learning_rate=0.0005),
               loss='categorical_crossentropy',
               metrics=['accuracy'])

# Display model structure.
model1.summary()

# Define callbacks for early stopping and learning rate reduction.
early_stopping1 = EarlyStopping(
    monitor="val_accuracy",
    patience=20,
    restore_best_weights=True
)

reduce_lr1 = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6
)


'''2.2b/2.3b: Neural Network Architecture Design/Hyperparamters – MODEL 2'''

# Model 2 is deeper and more complex, allowing it to learn more detailed hierarchical features.
# It contains four convolutional layers and two dense layers, with stronger regularization
# to balance its increased capacity.

model2 = Sequential()

# --- Convolutional Layers ---
model2.add(Conv2D(32, (4, 4), activation='relu', input_shape=(500, 500, 3)))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2)))

model2.add(Conv2D(64, (4, 4), activation='relu'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2)))

model2.add(Conv2D(128, (4, 4), activation='relu'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2)))

model2.add(Conv2D(256, (4, 4), activation='relu'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2)))

# --- Fully Connected Layers ---
model2.add(BatchNormalization())
model2.add(Flatten())

# Two dense layers improve the model's ability to form deeper abstract representations.
model2.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
model2.add(Dropout(0.2))

model2.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
model2.add(Dropout(0.2))

# Final softmax layer outputs class probabilities.
model2.add(Dense(3, activation='softmax', kernel_regularizer=l2(0.001)))

# Compile Model 2 using a smaller learning rate to ensure stable training for deeper networks.
model2.compile(optimizer=Adam(learning_rate=0.0001),
               loss='categorical_crossentropy',
               metrics=['accuracy'])

# Show model summary to confirm architecture.
model2.summary()

# Define callbacks (different patience for deeper model).
early_stopping2 = EarlyStopping(monitor="val_accuracy", patience=25, restore_best_weights=True)
reduce_lr2 = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6)


'''2.4: Model Training and Evaluation'''

#Training MODEL 1
trained1 = model1.fit(
    train_generator,
    epochs=epochs,
    validation_data=valid_generator,
    callbacks=[early_stopping1, reduce_lr1]
)

# Save Model 1 and its history for later use.
model1.save('AER850_Project2/Model1/model1.keras')
with open('AER850_Project2/Model1/model1_history.pkl', 'wb') as file:
    pickle.dump(trained1.history, file)

# Load history and plot accuracy/loss curves.
with open('AER850_Project2/Model1/model1_history.pkl', 'rb') as f:
    history1 = pickle.load(f)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history1['accuracy'], label='Training Accuracy')
plt.plot(history1['val_accuracy'], label='Validation Accuracy')
plt.title('Model 1 — Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history1['loss'], label='Training Loss')
plt.plot(history1['val_loss'], label='Validation Loss')
plt.title('Model 1 — Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()

# Save the plots for inclusion in report.
plt.savefig('AER850_Project2/Model1/model1_training_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Evaluate Model 1 performance on unseen test data.
test_loss1, test_accuracy1 = model1.evaluate(test_generator, verbose=2)
print("Test Loss (Model 1):", test_loss1)
print("Test Accuracy (Model 1):", test_accuracy1)


# Training MODEL 2
trained2 = model2.fit(
    train_generator,
    epochs=epochs,
    validation_data=valid_generator,
    callbacks=[early_stopping2, reduce_lr2]
)

# Save Model 2 and its history.
model2.save('AER850_Project2/Model2/model2.keras')
with open('AER850_Project2/Model2/model2_history.pkl', 'wb') as file:
    pickle.dump(trained2.history, file)

# Load history and plot accuracy/loss for Model 2.
with open('AER850_Project2/Model2/model2_history.pkl', 'rb') as f:
    history2 = pickle.load(f)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history2['accuracy'], label='Training Accuracy')
plt.plot(history2['val_accuracy'], label='Validation Accuracy')
plt.title('Model 2 — Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history2['loss'], label='Training Loss')
plt.plot(history2['val_loss'], label='Validation Loss')
plt.title('Model 2 — Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()

plt.savefig('AER850_Project2/Model2/model2_training_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Evaluate Model 2 on test data.
test_loss2, test_accuracy2 = model2.evaluate(test_generator, verbose=2)
print("Test Loss (Model 2):", test_loss2)
print("Test Accuracy (Model 2):", test_accuracy2)