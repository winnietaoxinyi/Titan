import tensorflow as tf
from tensorflow.keras import layers, models

class ResizeLayer(layers.Layer):
    def __init__(self, target_size, **kwargs):
        super(ResizeLayer, self).__init__(**kwargs)
        self.target_size = target_size

    def call(self, inputs):
        return tf.image.resize(inputs, self.target_size, method='bilinear')

def create_ssd_mobilenetv2(num_classes, num_anchors):
    base_model = tf.keras.applications.MobileNetV2(input_shape=[300, 300, 3], include_top=False)
    base_model.trainable = True

    feature_layer_names = [
        'block_13_expand_relu',
        # 'block_14_expand_relu',
        # 'block_15_expand_relu',
    ]
    feature_layers = [base_model.get_layer(name).output for name in feature_layer_names]

    target_shape = tf.keras.backend.int_shape(feature_layers[0])[1:3]
    resize_layer = ResizeLayer(target_size=target_shape)

    resized_features = [resize_layer(feature) for feature in feature_layers]

    confs = []
    locs = []
    for feature in resized_features:
        conf = layers.Conv2D(num_anchors * num_classes, (3, 3), padding='same')(feature)
        loc = layers.Conv2D(num_anchors * 4, (3, 3), padding='same')(feature)

        # Reshape outputs to match target shapes
        conf = layers.Reshape((-1, num_classes))(conf)  # (batch_size, num_boxes, num_classes)
        loc = layers.Reshape((-1, 4))(loc)  # (batch_size, num_boxes, 4)

        loc = layers.Activation('sigmoid')(loc)

        # Add Dropout after the convolutional layers
        conf = layers.Dropout(0.5)(conf)  # Dropout with a rate of 0.5
        loc = layers.Dropout(0.5)(loc)

        confs.append(conf)
        locs.append(loc)

    # Concatenate the outputs from all feature layers
    confs = layers.Concatenate(axis=1, name='confidence')(confs)
    locs = layers.Concatenate(axis=1, name='localization')(locs)

    model = models.Model(inputs=base_model.input, outputs={'confidence': confs, 'localization': locs})
    return model
