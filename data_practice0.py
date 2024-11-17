import cv2
import os
from tensorflow.keras.preprocessing import image
import numpy as np

# Set directories
image_dir = 'dataset/train/images/'
label_dir = 'dataset/train/labels/'

# List image and label files
image_files = os.listdir(image_dir)
label_files = os.listdir(label_dir)

# Sort files to match images with labels
image_files.sort()
label_files.sort()

# Load the first image and label
image1 = image_files[9]
label_file = label_files[9]
imgfile = os.path.join(image_dir, image1)
labelfile = os.path.join(label_dir, label_file)

print(imgfile, labelfile)

# Initialize label list
labels = []

# Load the label data
with open(labelfile, 'r') as label_file_obj:
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

        # Store the bounding box
        labels.append((class_index, x1_norm, y1_norm, x2_norm, y2_norm))

# Load the image and convert to numpy array (OpenCV compatible)
img = image.load_img(imgfile)
original_w, original_h = img.size

# Get target size for resizing
x = original_w // 2
y = original_h // 2

# Resize image to half of its original size
img_resized = image.load_img(imgfile, target_size=(x, y))

# Convert to numpy array
img_resized = image.img_to_array(img_resized).astype(np.uint8)

# Get dimensions of resized image
img_height, img_width = img_resized.shape[:2]  # Get the dimensions of the resized image

# Draw rectangles based on labels
for (_, x1_norm, y1_norm, x2_norm, y2_norm) in labels:
    # Convert normalized coordinates to actual pixel values based on the original image size
    x1_pixel = int(x1_norm * original_w)  # Use the original image width
    y1_pixel = int(y1_norm * original_h)  # Use the original image height
    x2_pixel = int(x2_norm * original_w)  # Use the original image width
    y2_pixel = int(y2_norm * original_h)  # Use the original image height

    # Resize the bounding box to the new image size
    x1_pixel_resized = int(x1_pixel * (img_width / original_w))
    y1_pixel_resized = int(y1_pixel * (img_height / original_h))
    x2_pixel_resized = int(x2_pixel * (img_width / original_w))
    y2_pixel_resized = int(y2_pixel * (img_height / original_h))

    # Draw the rectangle on the resized image
    cv2.rectangle(img_resized, (x1_pixel_resized, y1_pixel_resized), (x2_pixel_resized, y2_pixel_resized), (0, 255, 0), 1)

# Display the image with rectangles
cv2.imshow('Resized Image with Bounding Boxes', img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
