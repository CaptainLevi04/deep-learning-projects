"""
Brain Tumor Classification Architectures
Author: Amr Gamal
Organization: EME Internship Portfolio

Defines Custom Deep CNN baseline, MobileNetV2, and EfficientNetB0 models with
specialized classification heads for clinical MRI diagnostic tasks.
"""

from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, models


def build_custom_cnn(input_shape: Tuple[int, int, int] = (224, 224, 3), num_classes: int = 4) -> tf.keras.Model:
    """Builds a multi-stage Deep Convolutional Neural Network baseline from scratch."""
    inputs = layers.Input(shape=input_shape, name="input_image")
    x = layers.Rescaling(1.0 / 255.0)(inputs)

    # Conv Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Conv Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Conv Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Conv Block 4
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Dense Classification Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Custom_Deep_CNN")
    return model


def build_efficientnetb0(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 4,
    trainable_backbone: bool = False
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    Builds an EfficientNetB0 transfer learning architecture with custom dense head.
    Returns both the full model and the base feature extractor reference.
    """
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )
    base_model.trainable = trainable_backbone

    inputs = layers.Input(shape=input_shape, name="input_image")
    # Preprocessing is handled internally by EfficientNet
    x = base_model(inputs, training=trainable_backbone)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="EfficientNetB0_Classifier")
    return model, base_model


def build_mobilenetv2(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 4,
    trainable_backbone: bool = False
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """Builds a lightweight MobileNetV2 transfer learning architecture."""
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )
    base_model.trainable = trainable_backbone

    inputs = layers.Input(shape=input_shape, name="input_image")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=trainable_backbone)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_Classifier")
    return model, base_model
