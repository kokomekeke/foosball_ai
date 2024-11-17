import concurrent

import cv2
import tensorflow as tf
import os
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Set directories
image_dir = 'dataset/train/images/'
label_dir = 'dataset/train/labels/'
test_image_dir = 'dataset/test/images/'  # Teszt képek könyvtára
test_label_dir = 'dataset/test/labels/'  # Teszt címkék könyvtára (ha vannak)

# List all image files and label files for training
image_files = os.listdir(image_dir)
label_files = os.listdir(label_dir)

# Sort the files if necessary to ensure they correspond
image_files.sort()
label_files.sort()

images = []
labels = []

for img_file, label_file in zip(image_files, label_files):
    # Load image
    img_path = os.path.join(image_dir, img_file)
    # img = image.load_img(img_path, target_size=(256, 256))
    img = image.load_img(img_path)

    original_w, original_h = img.size

    x = original_w // 2
    y = original_h // 2

    # Kép betöltése a kívánt méretre
    img = image.load_img(img_path, target_size=(x, y))

    img_array = image.img_to_array(img)  # Convert image to array
    images.append(img_array)

    # Load corresponding label
    label_path = os.path.join(label_dir, label_file)
    with open(label_path, 'r') as label_file_obj:
        label_data = label_file_obj.read().strip().split()  # Split the line into separate values
        for i in range(0, len(label_data), 5):
            class_index = int(label_data[i])
            x_center_norm = float(label_data[i + 1])
            y_center_norm = float(label_data[i + 2])
            width_norm = float(label_data[i + 3])
            height_norm = float(label_data[i + 4])

            # Compute bounding box coordinates
            x1_norm = x_center_norm - width_norm / 2  # Left
            y1_norm = y_center_norm - height_norm / 2  # Top
            x2_norm = x_center_norm + width_norm / 2  # Right
            y2_norm = y_center_norm + height_norm / 2  # Bottom

            x1_norm = x1_norm / 2
            y1_norm = y1_norm / 2
            x2_norm = x2_norm / 2
            y2_norm = y2_norm / 2

            labels.append((class_index, x1_norm, y1_norm, x2_norm, y2_norm))



# Convert to numpy arrays
images = np.array(images)
labels = np.array(labels)

print(labels[:10])

img = images[0]
color = (0, 255, 0)  # Zöld szín (BGR formátumban)
thickness = 2

for (_, x1, y1, x2, y2) in label_data:
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

cv2.imshow('Bekeretezett kép', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Convert labels to categorical if needed (e.g., one-hot encoding)
labels = tf.keras.utils.to_categorical(labels, num_classes=len(np.unique(labels)))

# Create a TensorFlow Dataset
dataset = tf.data.Dataset.from_tensor_slices((images, labels))

# Shuffle, batch, and prefetch the dataset for performance
dataset = dataset.shuffle(buffer_size=1000).batch(32).prefetch(tf.data.AUTOTUNE)

# Define a simple CNN model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(256, 256, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(labels)), activation='softmax')  # Output layer (number of classes)
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(dataset, epochs=5)


model.save("my_model.h5")
# --- Tesztelés ---

# List all test image files
test_image_files = os.listdir(test_image_dir)

# Load test images into an array
test_images = []

for img_file in test_image_files:
    img_path = os.path.join(test_image_dir, img_file)
    img = image.load_img(img_path, target_size=(256, 256))  # Resize to 256x256
    img_array = image.img_to_array(img)  # Convert image to array
    test_images.append(img_array)

# Convert to numpy array
test_images = np.array(test_images)

# Normalize images (same as training data preprocessing)
test_images = test_images / 255.0  # Normalize if needed

# Make predictions on test images
predictions = model.predict(test_images)


# Show the first 5 predictions and their corresponding images
def plot_predictions(images, predictions, num_images=5):
    plt.figure(figsize=(15, 10))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i].astype("uint8"))
        predicted_class = np.argmax(predictions[i])  # Get predicted class index
        plt.title(f"Predicted: {predicted_class}")
        plt.axis('off')
    plt.show()


# Display the first 5 test images with their predictions
plot_predictions(test_images, predictions, num_images=5)

# If you have true labels for the test images (optional):
# Load the true labels
test_labels = []
for label_file in os.listdir(test_label_dir):
    with open(os.path.join(test_label_dir, label_file), 'r') as label_file_obj:
        label_data = label_file_obj.read().strip().split()
        class_index = int(label_data[0])  # Assuming the first value is the class index
        test_labels.append(class_index)

# Convert test labels to categorical if needed (one-hot encoding)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=len(np.unique(labels)))

# Calculate accuracy on the test set
from sklearn.metrics import accuracy_score
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(test_labels, axis=1)
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Test Accuracy: {accuracy}")