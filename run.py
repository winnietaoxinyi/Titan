import label
import model
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf


# Function to load and preprocess an image  
def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (300, 300))  # Resize to 300x300 for SSD MobileNetV2
    image = image / 255.0  # Normalize
    return image


def pad_target_boxes(target_boxes, num_predicted_boxes=722):
    """
    Pad the target boxes to match the number of predicted boxes.

    Args:
        target_boxes: numpy array of shape (num_gt_boxes, 4) - Ground truth boxes.
        num_predicted_boxes: int - Number of predicted boxes.

    Returns:
        padded_target_boxes: numpy array of shape (num_predicted_boxes, 4)
    """
    num_gt_boxes = target_boxes.shape[0]
    
    if num_gt_boxes < num_predicted_boxes:
        # Pad with zeros (or another appropriate value)
        padding = np.zeros((num_predicted_boxes - num_gt_boxes, 4))
        padded_target_boxes = np.vstack([target_boxes, padding])
    else:
        # If there are more ground truth boxes than predicted boxes, truncate them
        padded_target_boxes = target_boxes[:num_predicted_boxes]
    
    return padded_target_boxes


# Processes clips listed in a .txt file 
def process_clips_for_training(txt_file_path, image_base_dir, label_base_dir):
    """
    Returns:
        images (np.array): Array of preprocessed images ready for training.
        all_formatted_locs (np.array): Array of padded bounding boxes for all clips.
        all_formatted_confs (np.array): Array of padded one-hot encoded class labels for all clips.
        num_classes (int): The number of object classes found across all clips.
    """

    images = []
    all_formatted_locs = []
    all_formatted_confs = []
    num_classes = None

    # Read the .txt file to get the list of clip names
    with open(txt_file_path, 'r') as file:
        clip_names = [line.strip() for line in file if line.strip()]

    for clip_name in clip_names:
        # Construct paths to the image directory and corresponding CSV file
        image_dir = os.path.join(image_base_dir, clip_name, 'images')
        # print("Constructed path:", image_dir)
        csv_file_path = os.path.join(label_base_dir, f'{clip_name}.csv')

        # Process the CSV file to get formatted_locs and formatted_confs
        formatted_locs, formatted_confs = label.process_csv_for_ssd(csv_file_path)

        # Load and preprocess images
        image_files = sorted([img for img in os.listdir(image_dir) if img.endswith('.png')])
        for img_file in image_files:
            img_path = os.path.join(image_dir, img_file)
            image = load_and_preprocess_image(img_path)
            if image is not None:
                images.append(image)

        # Append the processed data to the lists
        all_formatted_locs.extend(formatted_locs)
        all_formatted_confs.extend(formatted_confs)

        # Get the number of classes from formatted_confs if not already set
        if num_classes is None:
            num_classes = formatted_confs.shape[-1]

    # Convert lists to numpy arrays
    images = np.array(images)
    all_formatted_locs = tf.ragged.constant(all_formatted_locs)
    all_formatted_confs = tf.ragged.constant(all_formatted_confs)

    return images, all_formatted_locs, all_formatted_confs, num_classes


# Step 1: Process the data
txt_file_path = 'one.txt'
image_base_dir = r'D:\titan_data\dataset\images_anonymized'
label_base_dir = r'titan_0_4'

images, all_locs, all_confs, num_classes = process_clips_for_training(
    txt_file_path,
    image_base_dir,
    label_base_dir
)

# Step 2: Create the SSD MobileNetV2 model
model = model.create_ssd_mobilenetv2(num_classes, 2)

# Step 3: Compile the model
model.compile(
    optimizer='adam',
    loss={
        'confidence': 'categorical_crossentropy',  # Matches the output name 'confidence'
        'localization': 'mean_squared_error'  # Matches the output name 'localization'
    },
    metrics={'confidence': 'accuracy'}
)

all_locs = all_locs.to_tensor(default_value=0)
all_confs = all_confs.to_tensor(default_value=0)

# Step 4: Train the model
history = model.fit(
    x=images,  # The input images
    y=[all_confs, all_locs],  # The ground truth labels: class and bounding boxes
    epochs=10,  # Number of epochs
    batch_size=32,  # Batch size
    validation_split=0.2  # Optional: Use part of the data for validation
)

import matplotlib.pyplot as plt

# Plot training & validation loss values
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

# Step 5: Save the trained model
model.save('trained_ssd_mobilenetv2.h5')

print("Training completed and model saved.")