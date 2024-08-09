import os
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle


# Read txt to get clips number
def read_clips_from_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    clips = [line.strip() for line in lines if line.strip()]
    return clips

# 读取官方CSV文件to get label
def read_labels_from_csv(file_path):

    df = pd.read_csv(file_path)
    # print('AAA')

    labels = {}
    for _, row in df.iterrows():
        frame_number = row['frames'].split('.')[0]  # 去掉扩展名
        obj_track_id = row['obj_track_id']
        labels[(frame_number,obj_track_id)] = {
            'label': row['label'],
            'bbox': (row['top'], row['left'], row['height'], row['width']),
            'obj_track_id': obj_track_id

            # 'attributes': {
            #     'Trunk Open': row.get('attributes.Trunk Open', ''),
            #     'Motion Status': row.get('attributes.Motion Status', ''),
            #     'Doors Open': row.get('attributes.Doors Open', ''),
            #     'Communicative': row.get('attributes.Communicative', ''),
            #     'Complex Contextual': row.get('attributes.Complex Contextual', ''),
            #     'Atomic Actions': row.get('attributes.Atomic Actions', ''),
            #     'Simple Context': row.get('attributes.Simple Context', ''),
            #     'Transporting': row.get('attributes.Transporting', ''),
            #     'Age': row.get('attributes.Age', '')
            # }
        }
    return labels

# 动态生成标签映射
def generate_label_map(labels):
    unique_labels = set(label_info['label'] for label_info in labels)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    return label_map

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
    mask = background_subtraction(image)
    mask = apply_morphology(mask)
    contours = detect_contours(mask)
    contours = filter_contours(contours)
    # 这里可以根据需要对检测到的轮廓进行处理
    image = cv2.resize(image, (224, 224))
    image = image.astype('float32')  / 255.0 #转化减少内存
    return image

# 加载指定编号的剪辑中的所有图像
def load_clips_images_with_labels(clip_ids, dataset_path, csv_base_path):
    images = []
    labels = []

    for clip_id in clip_ids:
        clip_folder = os.path.join(dataset_path, f'{clip_id}', 'images')
        csv_path = os.path.join(csv_base_path, f'{clip_id}.csv')

        # 打印尝试访问的文件路径
        print(f"Trying to access CSV file at: {csv_path}")

        if not os.path.exists(csv_path):
            print(f"CSV file does not exist: {csv_path}")
            continue

        labels_dict = read_labels_from_csv(csv_path)

        if os.path.exists(clip_folder):
            image_paths = [os.path.join(clip_folder, img) for img in sorted(os.listdir(clip_folder)) if
                            img.endswith('.png')]
            for img_path in image_paths:
                img = load_and_preprocess_image(img_path)
                if img is not None:
                    frame_number = os.path.basename(img_path).split('.')[0]
                    # 添加所有匹配的obj_track_id
                    for (frame_num, obj_track_id), label_info in labels_dict.items():
                        if frame_num == frame_number:
                            images.append(img)
                            labels.append(label_info)
    return np.array(images), labels


# 设置数据集路径
dataset_path = r'E:\download\titan_data\dataset\images_anonymized'
csv_base_path = r'E:\download\titan_data\dataset\titan_0_4'

# 读取训练剪辑编号
train_clips = read_clips_from_txt(r'E:\download\titan_data\dataset\splits\one round.txt')

# 加载训练数据
train_images, train_labels = load_clips_images_with_labels(train_clips, dataset_path, csv_base_path)

# 动态生成标签映射
label_map = generate_label_map(train_labels)
num_classes = len(label_map)

# 保存标签映射
with open('label_map.pkl', 'wb') as f:
    pickle.dump(label_map, f)

# 验证标签转换
train_labels_converted = []
for label_info in train_labels:
    try:
        label = label_map[label_info['label']]
        train_labels_converted.append(label)
    except KeyError:
        print(f"标签 {label_info['label']} 在标签映射中未找到。")
        continue

train_labels_converted = np.array(train_labels_converted)


# # 预处理标签
# def preprocess_labels(labels, label_map):
#     processed_labels = []
#     for label_info in labels:
#         label = label_map[label_info['label']]
#         bbox = label_info['bbox']
#         processed_labels.append({
#             'label': label,
#             'bbox': bbox,
#             # 'attributes': label_info['attributes']
#         })
#     return processed_labels
#
# train_labels = preprocess_labels(train_labels, label_map)

# 检查形状
print(f'Train images shape: {train_images.shape}')
print(f'Train labels shape: {len(train_labels)}')

# 构建和编译模型
base_model = tf.keras.applications.ResNet50(
    input_shape=(224, 224, 3),
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

# 加载已有的 2_round_model 模型
# model.load_weights('2_round_model.h5')

# 训练模型
print(len(train_labels))

labels_array = np.array([label_map[l['label']] for l in train_labels])
print(f'Converted Labels dtype: {labels_array.dtype}')
print(f'Converted Labels: {labels_array[:10]}')  # 打印前10个标签，检查它们是否正确转换
history = model.fit(train_images, labels_array, epochs=10, batch_size=32, validation_split=0.2)


model.save('3_round_model.h5')

history_df = pd.DataFrame(history.history)
history_df.to_csv('training_history.csv', index=False)
print("Model and training history saved.")

