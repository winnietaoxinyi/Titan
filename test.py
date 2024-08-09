import os
import numpy as np
import tensorflow as tf
import pandas as pd
import cv2


def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (300, 300))  # Resize to the input size of the model
    image = image / 255.0  # Normalize
    return np.expand_dims(image, axis=0)  # Add batch dimension


import tensorflow as tf

class ResizeLayer(tf.keras.layers.Layer):
    def __init__(self, target_size, **kwargs):
        super(ResizeLayer, self).__init__(**kwargs)
        self.target_size = target_size

    def call(self, inputs):
        return tf.image.resize(inputs, self.target_size, method='bilinear')


def predict_on_images(model, txt_file_path, image_base_dir, output_dir, num_classes):
    # Read the .txt file to get the list of clip names
    with open(txt_file_path, 'r') as file:
        clip_names = [line.strip() for line in file if line.strip()]

    for clip_name in clip_names:
        # Construct path to the image directory
        image_dir = os.path.join(image_base_dir, clip_name, 'images')
        output_csv_path = os.path.join(output_dir, f'pred_{clip_name}.csv')

        predictions = []
        # Load and preprocess images
        image_files = sorted([img for img in os.listdir(image_dir) if img.endswith('.png')])
        for img_file in image_files:
            img_path = os.path.join(image_dir, img_file)
            image = process_image(img_path)
            if image is not None:
                output = model.predict(image)
                confs = output['confidence'][0]  # Remove batch dimension
                locs = output['localization'][0]  # Remove batch dimension

                # Debugging: Check the shape and content of locs
                # print(f"locs shape: {locs.shape}")
                # print(f"locs after removing batch dimension: {locs}")

                # Process the predictions (this will depend on your specific model's output format)
                # confs = confs[0]  # Remove batch dimension
                # locs = locs[0]  # Remove batch dimension

                # Convert predictions to a more interpretable format
                visited = set()
                for i in range(len(confs)):
                    label_index = np.argmax(confs[i])  # Get the predicted class index
                    # label_name = label_map.get(label_index, 'unknown')   Map to descriptive name
                    # if label_name != 'unknown':  Skip if it's an unknown label or background                   
                    if label_index < num_classes - 1 and label_index not in visited:
                        visited.add(label_index)
                        bbox = locs[i]
                        xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
                        # Convert to top, left, height, width based on original image size
                        top = ymin * 1520
                        left = xmin * 2704
                        height = (ymax - ymin) * 1520
                        width = (xmax - xmin) * 2704
                        predictions.append({
                            'frames': img_file,
                            'label': label_index,
                       #     'obj_track': obj_track_id,
                            'top': int(top),  # Scale to original image size if needed
                            'left': int(left),  # Scale to original image size if needed
                            'height': int(height),  # Scale to original image size if needed
                            'width': int(width)  # Scale to original image size if needed
                        })

        # Write predictions to CSV
        write_predictions_to_csv(predictions, output_csv_path)
        print(f'Predictions saved to {output_csv_path}')


def write_predictions_to_csv(predictions, output_csv_path):
    df = pd.DataFrame(predictions)
    df.to_csv(output_csv_path, index=False)


def infer_num_classes_from_model(model):
    # Assuming the model output for class predictions is named 'confidence'
    conf_output_shape = model.get_layer('confidence').output_shape
    # conf_output_shape will be something like (None, num_boxes, num_classes)
    num_classes = conf_output_shape[-1]
    return num_classes


model_path = 'trained_ssd_mobilenetv2_2.h5'  # Replace with your model path
txt_file_path = 'two.txt'  # Replace with your txt file path
base_dir = 'images_anonymized'  # Base directory containing clips
output_dir = 'pred'  # Directory to save the CSV files

# model = tf.keras.models.load_model(model_path)
model = tf.keras.models.load_model(model_path, custom_objects={'ResizeLayer': ResizeLayer})
# model.summary()
# num_classes = infer_num_classes_from_model(model)
# print(f"Inferred number of classes: {num_classes}")
num_classes = 19
predict_on_images(model, txt_file_path, base_dir, output_dir, num_classes)