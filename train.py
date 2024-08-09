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


def pad_target_boxes(target_boxes, num_predicted_boxes, num_classes=722):
    """
    Pad the target boxes to match the number of predicted boxes.

    Args:
        target_boxes: numpy array of shape (num_gt_boxes, 4) - Ground truth boxes.
        num_predicted_boxes: int - Number of predicted boxes.
        num_classes: int - Number of classes.

    Returns:
        padded_target_boxes: numpy array of shape (num_predicted_boxes, 4)
        padded_target_confs: numpy array of shape (num_predicted_boxes, num_classes)
    """
    num_gt_boxes = target_boxes.shape[0]
    
    # Pad bounding boxes
    if num_gt_boxes < num_predicted_boxes:
        padding_boxes = np.zeros((num_predicted_boxes - num_gt_boxes, 4))
        padded_target_boxes = np.vstack([target_boxes, padding_boxes])
    else:
        padded_target_boxes = target_boxes[:num_predicted_boxes]

    # Pad class confidences
    if num_gt_boxes < num_predicted_boxes:
        padding_confs = np.zeros((num_predicted_boxes - num_gt_boxes, num_classes))
        padded_target_confs = np.vstack([np.ones((num_gt_boxes, num_classes)), padding_confs])
    else:
        padded_target_confs = np.ones((num_predicted_boxes, num_classes))

    return padded_target_boxes, padded_target_confs


def process_and_pad_data(txt_file_path, image_base_dir, label_base_dir, num_predicted_boxes=722):
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

        # Get the number of classes from formatted_confs if not already set
        if num_classes is None:
            num_classes = formatted_confs.shape[-1]

        # Pad the formatted_locs and formatted_confs to match the model's output shape
        for locs, confs in zip(formatted_locs, formatted_confs):
            padded_locs, padded_confs = pad_target_boxes(locs, num_predicted_boxes, num_classes)
            all_formatted_locs.append(padded_locs)
            all_formatted_confs.append(padded_confs)

    # Convert lists to numpy arrays
    images = np.array(images)
    all_formatted_locs = np.array(all_formatted_locs)
    all_formatted_confs = np.array(all_formatted_confs)

    return images, all_formatted_locs, all_formatted_confs, num_classes


# Step 1: Process the data
txt_file_path = 'one.txt'
image_base_dir = r'D:\titan_data\dataset\images_anonymized'
label_base_dir = r'titan_0_4'

images, all_locs, all_confs, num_classes = process_and_pad_data(
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

# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
# Step 4: Train the model
history = model.fit(
    x=images,  # The input images
    y=[all_confs, all_locs],  # The ground truth labels: class and bounding boxes
    epochs=10,  # Number of epochs
    batch_size=32,  # Batch size
    validation_split=0.2  # Optional: Use part of the data for validation
    # callbacks=[early_stopping]
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
model.save('trained_ssd_mobilenetv2_2.h5')

print("Training completed and model saved.")
