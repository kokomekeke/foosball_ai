import cv2
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Set directories
image_dir = 'dataset/train/images/'
label_dir = 'dataset/train/labels/'
test_image_dir = 'dataset/test/images/'  # Test images directory
test_label_dir = 'dataset/test/labels/'  # Test labels directory (if available)

# List all image files and label files for training
image_files = os.listdir(image_dir)
label_files = os.listdir(label_dir)

# Sort the files to ensure they correspond
image_files.sort()
label_files.sort()

images = []
labels = []

for img_file, label_file in zip(image_files, label_files):
    # Load image
    img_path = os.path.join(image_dir, img_file)
    img = image.load_img(img_path)

    original_w, original_h = img.size

    # Resize image (example: half the size of original image)
    ratio = 2  # Ratio for resizing
    x = original_w // ratio
    y = original_h // ratio
    img = image.load_img(img_path, target_size=(x, y))
    img_array = image.img_to_array(img)  # Convert image to array
    images.append(img_array)

    # Load corresponding label
    label_path = os.path.join(label_dir, label_file)
    label_data = []

    with open(label_path, 'r') as label_file_obj:
        label_data_raw = label_file_obj.read().strip().split()  # Split the line into separate values
        for i in range(0, len(label_data_raw), 5):
            class_index = int(label_data_raw[i])
            x_center_norm = float(label_data_raw[i + 1])
            y_center_norm = float(label_data_raw[i + 2])
            width_norm = float(label_data_raw[i + 3])
            height_norm = float(label_data_raw[i + 4])

            # Normalize bounding box coordinates
            x_center_norm /= 2
            y_center_norm /= 2
            width_norm /= 2
            height_norm /= 2

            label_data.append((class_index, x_center_norm, y_center_norm, width_norm, height_norm))

    labels.append(label_data)

# Convert images and labels to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Create a TensorFlow Dataset
dataset = tf.data.Dataset.from_tensor_slices((images, labels))

# Shuffle, batch, and prefetch the dataset for performance
dataset = dataset.shuffle(buffer_size=1000).batch(32).prefetch(tf.data.AUTOTUNE)

# Define YOLO-like model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(x, y, 3)),

    # Convolutional layers for feature extraction
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Flatten the output from conv layers
    tf.keras.layers.Flatten(),

    # Fully connected layers for bounding box and class predictions
    tf.keras.layers.Dense(128, activation='relu'),

    # Output layer for bounding box and class predictions
    tf.keras.layers.Dense(5, activation='sigmoid')  # 5 values: [class, x_center, y_center, width, height]
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Train the model
model.fit(dataset, epochs=5)

model.save("yolo_model.h5")

# --- Test --- #

# List all test image files
test_image_files = os.listdir(test_image_dir)

# Load test images into an array
test_images = []

for img_file in test_image_files:
    img_path = os.path.join(test_image_dir, img_file)
    img = image.load_img(img_path, target_size=(x, y))  # Resize to match model input
    img_array = image.img_to_array(img)  # Convert image to array
    test_images.append(img_array)

# Convert to numpy array
test_images = np.array(test_images)

# Normalize images
test_images = test_images / 255.0

# Make predictions on test images
predictions = model.predict(test_images)


# Show the first 5 predictions and their corresponding images
def plot_predictions(images, predictions, num_images=5):
    plt.figure(figsize=(15, 10))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i].astype("uint8"))
        predicted_class = np.argmax(predictions[i, 0])  # Get predicted class index
        plt.title(f"Predicted: {predicted_class}")
        plt.axis('off')
    plt.show()


# Display the first 5 test images with their predictions
plot_predictions(test_images, predictions, num_images=5)
