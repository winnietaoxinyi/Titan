import os
import pickle
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd

# Read txt to get clips number
def read_clips_from_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    clips = [line.strip() for line in lines if line.strip()]
    return clips

# 读取官方CSV文件获取标签
def read_labels_from_csv(file_path):
    df = pd.read_csv(file_path)
    labels = {}
    for _, row in df.iterrows():
        frame_number = row['frames'].split('.')[0]  # 去掉扩展名
        obj_track_id = row['obj_track_id']
        labels[(frame_number, obj_track_id)] = {
            'label': row['label'],
            'bbox': (row['top'], row['left'], row['height'], row['width']),
            'obj_track_id': obj_track_id
        }
    return labels

# 背景减除
def background_subtraction(image):
    subtractor = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=25, detectShadows=False)
    mask = subtractor.apply(image)
    return mask

# 应用形态学操作
def apply_morphology(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

# 轮廓检测
def detect_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# 过滤轮廓
def filter_contours(contours, min_area=500):
    filtered = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    return filtered

# 加载和预处理图像，并进行对象检测
def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    return image

# 加载指定编号的剪辑中的所有图像并生成独立的CSV文件
def load_clips_images_with_labels(clip_ids, dataset_path, csv_base_path, model, label_map):
    for clip_id in clip_ids:
        images = []
        labels = []
        frame_data = []

        clip_folder = os.path.join(dataset_path, f'{clip_id}', 'images')
        csv_path = os.path.join(csv_base_path, f'{clip_id}.csv')
        output_csv_path = os.path.join(csv_base_path, f'{clip_id}_results.csv')  # 输出CSV文件路径

        print(f"Trying to access CSV file at: {csv_path}")

        if not os.path.exists(csv_path):
            print(f"CSV file does not exist: {csv_path}")
            continue

        labels_dict = read_labels_from_csv(csv_path)

        if os.path.exists(clip_folder):
            frame_data = []
            image_paths = [os.path.join(clip_folder, img) for img in sorted(os.listdir(clip_folder)) if
                           img.endswith('.png')]
            for img_path in image_paths:
                img = load_and_preprocess_image(img_path)
                if img is not None:
                    frame_number = os.path.basename(img_path).split('.')[0]
                    for (frame_num, obj_track_id), label_info in labels_dict.items():
                        if frame_num == frame_number:
                            images.append(img)
                            frame_data.append({
                                'frame_number': frame_number,
                                'obj_track_id': obj_track_id,
                                'label_info': label_info
                            })


            if images:  # 如果有图像
                images = np.array(images)
                predictions = model.predict(images)
                # predicted_labels = np.argmax(predictions, axis=1)
                print(f"Predictions shape: {predictions.shape}")

                # 确保predictions的形状是 (batch_size, 4)，即预测的边界框的四个坐标值
                if predictions.shape[1] != 4:
                    raise ValueError("Expected predictions to have 4 values per prediction (top, left, height, width).")
                # 将预测结果写入CSV文件
                output_data = []
                for i, (prediction, data) in enumerate(zip(predictions, frame_data)):
                    label_index = np.argmax(prediction[:len(label_map)])
                    label = list(label_map.keys())[list(label_map.values()).index(label_index)]
                    bbox = prediction[-4:]  # 获取最后4个元素作为边界框
                    frame_name = f'{data["frame_number"]}.png'

                    output_data.append([
                        frame_name,
                        label,
                        data['obj_track_id'],
                        bbox[0], bbox[1], bbox[2], bbox[3],
                    ])

                # # 将预测结果写入CSV文件
                # output_data = []
                # for i, (predicted_label, data) in enumerate(zip(predicted_labels, frame_data)):
                #     label_index = predicted_label
                #     label = list(label_map.keys())[list(label_map.values()).index(label_index)]
                #     bbox = predictions[i]
                #     frame_name = f'{data["frame_number"]}.png'
                #
                #     output_data.append([
                #         frame_name,
                #         label,
                #         data['obj_track_id'],
                #         bbox[0], bbox[1], bbox[2], bbox[3],
                #         # 添加属性信息，这里你可以添加需要的其他字段
                #     ])

                # 写入独立的CSV文件
                output_df = pd.DataFrame(output_data, columns=[
                    'frames', 'label', 'obj_track_id', 'top', 'left', 'height', 'width',
                    # 添加其他属性列
                ])
                output_df.to_csv(output_csv_path, index=False)
                print(f"Results saved to {output_csv_path}")


# 设置数据集路径
dataset_path = r'E:\download\titan_data\dataset\images_anonymized'
csv_base_path = r'E:\download\titan_data\dataset\titan_0_4'

# 读取测试剪辑编号
test_clips = read_clips_from_txt(r'E:\download\titan_data\dataset\splits\final1.txt')

# 加载标签映射
with open('label_map.pkl', 'rb') as f:
    label_map = pickle.load(f)

# 构建模型
def build_model(input_shape, num_classes):
    base_model = tf.keras.applications.ResNet50(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model((224, 224, 3), len(label_map))
model.build(input_shape=(None, 224, 224, 3))
model.load_weights('6_round_model.h5')

# 加载并处理每个clip，生成独立的CSV文件
load_clips_images_with_labels(test_clips, dataset_path, csv_base_path, model, label_map)
