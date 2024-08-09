import numpy as np
import pandas as pd

def process_csv_for_ssd(csv_file_path, img_width=2704, img_height=1520):
    """
    Processes the CSV file to generate formatted bounding boxes (locs) and class labels (confs)
    for SSD training.

    Args:
        csv_file_path (str): Path to the CSV file containing annotations.
        img_width (int): The width of the images in pixels. Default is 1920.
        img_height (int): The height of the images in pixels. Default is 1080.

    Returns:
        formatted_locs (np.array): Array of padded bounding boxes.
        formatted_confs (np.array): Array of padded one-hot encoded class labels.
    """

    # Load CSV file
    annotations = pd.read_csv(csv_file_path)

    # Get all unique labels from the CSV and create a label map
    unique_labels = annotations['label'].unique()
    label_map = {label: idx + 1 for idx, label in enumerate(unique_labels)}  # Start indexing at 1
    num_classes = len(label_map) + 1  # Including background class

    def normalize_bbox(row):
        """Normalize bounding box coordinates to [0, 1]."""
        xmin = row['left'] / img_width
        ymin = row['top'] / img_height
        xmax = (row['left'] + row['width']) / img_width
        ymax = (row['top'] + row['height']) / img_height
        return [xmin, ymin, xmax, ymax]

    def one_hot_encode(label):
        """One-hot encode the class labels."""
        one_hot = [0] * num_classes
        one_hot[label_map[label]] = 1
        return one_hot

    # Initialize lists to hold the formatted data
    formatted_locs = []
    formatted_confs = []

    # Group annotations by 'frames' (image files)
    grouped = annotations.groupby('frames')

    # Find the maximum number of objects in any image
    max_objects = max(len(group) for name, group in grouped)

    for frame, group in grouped:
        bboxes = []
        confs = []
        for _, row in group.iterrows():
            bboxes.append(normalize_bbox(row))
            confs.append(one_hot_encode(row['label']))

        # Pad bboxes and confs to max_objects length
        while len(bboxes) < max_objects:
            bboxes.append([0, 0, 0, 0])  # Pad with zeros
            confs.append([0] * num_classes)  # Pad with zeros

        formatted_locs.append(bboxes)
        formatted_confs.append(confs)

    # Convert lists to numpy arrays
    formatted_locs = np.array(formatted_locs)
    formatted_confs = np.array(formatted_confs)

    return formatted_locs, formatted_confs

# Example usage:
# formatted_locs, formatted_confs = process_csv_for_ssd('clip_4.csv')
# print(len(formatted_locs))
