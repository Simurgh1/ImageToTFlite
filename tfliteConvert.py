import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Define directories
base_dir = r'Path to the base directory which contains the folders of images'
sideview_dir = os.path.join(base_dir, '<one of the views folder name>')
topview_dir = os.path.join(base_dir, "<other view's folder name>")

def load_images_from_folder(folder, target_size=(64, 64)):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(folder, filename)
            img = load_img(img_path, target_size=target_size)
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
    return np.array(images)

# Load and preprocess images
sideview_images = load_images_from_folder(sideview_dir)
topview_images = load_images_from_folder(topview_dir)

# Create labels for each category
sideview_labels = np.zeros(len(sideview_images))
topview_labels = np.ones(len(topview_images))

# Combine and split the data
images = np.concatenate((sideview_images, topview_images), axis=0)
labels = np.concatenate((sideview_labels, topview_labels), axis=0)

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

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
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model in the Images folder
model_save_path = os.path.join(base_dir, 'model.tflite')
with open(model_save_path, 'wb') as f:
    f.write(tflite_model)

print(f"Model saved to {model_save_path}")
