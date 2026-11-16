"""
GTZAN Audio Spectrogram Music Genre Classifier
Author: Amr Gamal
Organization: EME Internship Portfolio

Implements an Audio Deep Learning pipeline treating Mel-Spectrogram visual representations
as 2D inputs for Deep Convolutional Networks. Benchmarks a scratch CNN against MobileNetV2
Transfer Learning for 10-genre musical classification.
"""

import argparse
import logging
import os
import sys
from typing import Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

GENRES = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
IMG_SHAPE = (224, 224, 3)


def build_mobilenet_audio_classifier(num_classes: int = 10) -> tf.keras.Model:
    """Builds MobileNetV2 with specialized classification head for audio spectrograms."""
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=IMG_SHAPE
    )
    base_model.trainable = True
    # Freeze bottom layers, fine-tune top 30
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    inputs = layers.Input(shape=IMG_SHAPE, name="spectrogram_input")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="genre_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="GTZAN_MobileNetV2")
    return model


def build_custom_audio_cnn(num_classes: int = 10) -> tf.keras.Model:
    """Builds a baseline scratch CNN for spectrogram feature comparison."""
    inputs = layers.Input(shape=IMG_SHAPE)
    x = layers.Rescaling(1.0 / 255.0)(inputs)

    for filters in [32, 64, 128, 256]:
        x = layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.2)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs=inputs, outputs=outputs, name="Custom_Spectrogram_CNN")


def main():
    parser = argparse.ArgumentParser(description="Audio Genre Classification via Mel-Spectrograms")
    parser.add_argument("--data_dir", type=str, default="gtzan_spectrograms", help="Path to spectrogram directory")
    parser.add_argument("--model_type", type=str, choices=["mobilenet", "custom_cnn"], default="mobilenet")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        logger.warning(f"Data directory '{args.data_dir}' not found on local path.")
        logger.info("Please download the GTZAN spectrogram dataset or run the interactive notebook.")
        return

    logger.info("Creating image dataset generators from spectrogram directories...")
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        validation_split=0.2,
        horizontal_flip=False  # Mel-spectrogram time axis should not be mirrored
    )

    train_gen = datagen.flow_from_directory(
        args.data_dir,
        target_size=IMG_SHAPE[:2],
        batch_size=args.batch_size,
        subset="training",
        class_mode="categorical"
    )

    val_gen = datagen.flow_from_directory(
        args.data_dir,
        target_size=IMG_SHAPE[:2],
        batch_size=args.batch_size,
        subset="validation",
        class_mode="categorical"
    )

    if args.model_type == "mobilenet":
        model = build_mobilenet_audio_classifier(num_classes=len(GENRES))
    else:
        model = build_custom_audio_cnn(num_classes=len(GENRES))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True)
    ]

    logger.info(f"Training {args.model_type} for {args.epochs} epochs...")
    history = model.fit(train_gen, validation_data=val_gen, epochs=args.epochs, callbacks=callbacks)

    val_loss, val_acc = model.evaluate(val_gen)
    logger.info(f"Final Validation Accuracy: {val_acc * 100:.2f}% | Loss: {val_loss:.4f}")


if __name__ == "__main__":
    main()
