import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Define directories
base_dir = r'Path to base directory'
sideview_dir = os.path.join(base_dir, 'sideview') #"sideview","top view","validation" are just a name of the folders within the base directory"
topview_dir = os.path.join(base_dir, 'topview')
validation_dir = os.path.join(base_dir, 'validation')

# Function to load images from folder
def load_images_from_folder(folder, target_size=(64, 64)):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(folder, filename)
            try:
                img = load_img(img_path, target_size=target_size)
                img_array = img_to_array(img) / 255.0
                images.append(img_array)
            except Exception as e:
                print(f"Error loading image {img_path}: {str(e)}")
    return np.array(images)

# Load images and labels
sideview_images = load_images_from_folder(sideview_dir)
topview_images = load_images_from_folder(topview_dir)
validation_images = load_images_from_folder(validation_dir)

# Create labels for each category
sideview_labels = np.zeros(len(sideview_images))
topview_labels = np.ones(len(topview_images))

# Combine images and labels
images = np.concatenate((sideview_images, topview_images), axis=0)
labels = np.concatenate((sideview_labels, topview_labels), axis=0)

# Split data into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Validation set (assuming it might not have labels)
validation_labels = np.full(len(validation_images), -1)  # Dummy labels for validation set

# Define the model
num_classes = 2  # sideview and topview

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
try:
    model.fit(train_images, train_labels, epochs=10, validation_data=(validation_images, validation_labels))
except Exception as e:
    print(f"Error during model training: {str(e)}")

# Convert the model to TensorFlow Lite format
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the converted model in the Images folder
    model_save_path = os.path.join(base_dir, 'model.tflite')
    with open(model_save_path, 'wb') as f:
        f.write(tflite_model)

    print(f"Model saved to {model_save_path}")
except Exception as e:
    print(f"Error converting and saving model: {str(e)}")
