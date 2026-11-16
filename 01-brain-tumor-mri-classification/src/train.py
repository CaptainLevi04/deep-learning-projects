"""
Brain Tumor Classification Training Pipeline
Author: Amr Gamal
Organization: EME Internship Portfolio

Orchestrates data preparation, multi-stage transfer learning (frozen feature extraction
followed by progressive fine-tuning), learning rate scheduling, and comprehensive evaluation.
"""

import argparse
import logging
import os
import sys

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from dataset import load_and_deduplicate_dataset, create_stratified_splits, build_tf_dataset, CLASS_NAMES
from models import build_custom_cnn, build_efficientnetb0, build_mobilenetv2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def train_model(args):
    # 1. Dataset Loading & Deduplication
    logger.info("Step 1: Loading & Deduplicating Dataset...")
    df = load_and_deduplicate_dataset(args.data_dir)
    train_df, val_df, test_df = create_stratified_splits(df)

    train_ds = build_tf_dataset(train_df, batch_size=args.batch_size, is_training=True)
    val_ds = build_tf_dataset(val_df, batch_size=args.batch_size, is_training=False)
    test_ds = build_tf_dataset(test_df, batch_size=args.batch_size, is_training=False)

    # 2. Model Instantiation
    logger.info(f"Step 2: Initializing Architecture '{args.arch}'...")
    if args.arch == "custom_cnn":
        model = build_custom_cnn()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=7, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
        ]
        logger.info("Training Custom CNN from scratch...")
        model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    elif args.arch == "efficientnetb0":
        model, base_model = build_efficientnetb0(trainable_backbone=False)

        # Stage 1: Feature Extraction
        logger.info("Stage 1: Training Classification Head (Backbone Frozen)...")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        stage1_callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
            tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True)
        ]
        model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=stage1_callbacks)

        # Stage 2: Fine-Tuning Top Backbone Layers
        logger.info("Stage 2: Unfreezing Top Layers for Fine-Tuning...")
        base_model.trainable = True
        # Fine-tune only top 30 layers
        for layer in base_model.layers[:-30]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        stage2_callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-7),
            tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
        ]
        model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=stage2_callbacks)

    # 3. Final Evaluation
    logger.info("=" * 60)
    logger.info("EVALUATING MODEL ON UNSEEN TEST PARTITION")
    logger.info("=" * 60)
    test_loss, test_acc = model.evaluate(test_ds)
    logger.info(f"Final Test Accuracy: {test_acc * 100:.2f}% | Test Loss: {test_loss:.4f}")

    # Generate Predictions & Detailed Report
    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    print("\nDetailed Clinical Diagnostic Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))


def main():
    parser = argparse.ArgumentParser(description="Brain Tumor MRI Classifier Trainer")
    parser.add_argument("--data_dir", type=str, default="brain_tumor_dataset", help="Path to dataset root folder")
    parser.add_argument("--arch", type=str, choices=["efficientnetb0", "custom_cnn", "mobilenetv2"],
                        default="efficientnetb0", help="Architecture to train")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Base learning rate")
    args = parser.parse_args()

    train_model(args)


if __name__ == "__main__":
    main()
