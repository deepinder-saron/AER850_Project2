'''2.5: Model Testing and Visualization (Evaluating Trained CNNs on Unseen Test Images)'''

# In this section, we test both trained models (Model 1 and Model 2) using 
# a few unseen test images. The goal is to evaluate how well each model 
# generalizes to new data and to visualize their predictions.

# Importing required libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

''' STep 1: Loading the Trained Models'''

# The models were saved previously in `.keras` format after training.
# Using `tf.keras.models.load_model()` allows us to load them exactly 
# as they were (architecture + weights + optimizer state).

model1 = tf.keras.models.load_model('AER850_Project2/Model1/model1.keras')
model2 = tf.keras.models.load_model('AER850_Project2/Model2/model2.keras')

# These labels correspond to the three fault types in the dataset:
# 1. crack         → visible surface crack
# 2. missing-head  → screw head or bolt is missing
# 3. paint-off     → region of paint removed or damaged
class_labels = ['crack', 'missing-head', 'paint-off']

'''Step 2: Define Test Images'''

# Each test image represents one defect class. These are new examples 
# not seen by the model during training or validation.
image_paths = [
    'Project2Data/Data/test/crack/test_crack.jpg',
    'Project2Data/Data/test/missing-head/test_missinghead.jpg',
    'Project2Data/Data/test/paint-off/test_paintoff.jpg'
]

# Dictionary used to reference the true class labels
true_labels = {
    'Project2Data/Data/test/crack/test_crack.jpg': 'crack',
    'Project2Data/Data/test/missing-head/test_missinghead.jpg': 'missing-head',
    'Project2Data/Data/test/paint-off/test_paintoff.jpg': 'paint-off'
}

'''Step 3: Image Preprocessing Function'''

# The same preprocessing used during training must be applied here:
#   1. Resize image to 96x96 (same as training input) when applicable
#   2. Convert to grayscale when applicable 
#   3. Convert image to NumPy array
#   4. Normalize pixel values to range [0, 1]
#   5. Expand dimensions to match input shape expected by model: (96, 96, 1) or (500, 500, 3)

def preprocess_image(image_path, target_size=(96, 96)):
    img = load_img(image_path, target_size=target_size, color_mode='grayscale')
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    img_array /= 255.0  # normalize pixel intensity
    return img_array

'''Step 4: Prediction + Visualization Function'''

# This function:
#   - Preprocesses the image
#   - Runs the model to obtain predicted probabilities
#   - Identifies the most likely class
#   - Displays the image with both true and predicted labels
#   - Lists the predicted probability for each class
#   - Saves the resulting figure in the appropriate folder

def display_prediction(model, image_path, model_name):
    # Preprocess image for the model
    img_array = preprocess_image(image_path)
    
    # Obtain predictions — model outputs a probability for each class
    predictions = model.predict(img_array)
    predicted_probs = predictions[0] * 100  # Convert to percentage
    predicted_label = class_labels[np.argmax(predicted_probs)]
    true_label = true_labels[image_path]
    
    # Load the image again for display (grayscale)
    img = load_img(image_path, color_mode='grayscale')

    # Displaying the result
    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    
    # Title includes both model name and the classification result
    plt.title(f"{model_name}\nTrue: {true_label} | Predicted: {predicted_label}", fontsize=12, pad=20)

    # Print predicted class probabilities as text on the image
    y_pos = 25
    for label, prob in zip(class_labels, predicted_probs):
        plt.text(10, y_pos, f"{label}: {prob:.1f}%", color='purple', fontsize=11)
        y_pos += 30

    # Create directory to save prediction visualizations
    save_dir = f"AER850_Project2/{model_name.replace(' ', '_')}_Predictions"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the plotted figure
    filename = os.path.basename(image_path).replace(
        '.jpg', f'_{model_name.replace(" ", "_")}.png'
    )
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()


'''Step 5: Run Predictions for Each Model'''

print("\n================ MODEL 1 TEST RESULTS ================\n")
for image_path in image_paths:
    display_prediction(model1, image_path, "Model 1")

print("\n================ MODEL 2 TEST RESULTS ================\n")
for image_path in image_paths:
    display_prediction(model2, image_path, "Model 2")


'''--------------------------------------------------------------------
INTERPRETATION:
- The grayscale image is shown for each test example.
- The model’s predicted class and probability distribution across
  all classes are displayed beside the image.
- This visualization helps assess whether the model correctly identifies
  each fault and how confident it is in each prediction.

Model 1 (simpler CNN):
    → Expected to perform adequately but may show slightly less confidence
      or misclassify complex defects.

Model 2 (deeper CNN):
    → Expected to perform better on subtle patterns, providing higher
      confidence scores due to greater feature extraction capacity.

--------------------------------------------------------------------'''
