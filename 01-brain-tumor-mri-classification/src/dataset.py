"""
Brain Tumor MRI Dataset Pipeline & Data Deduplication
Author: Amr Gamal
Organization: EME Internship Portfolio

Handles clinical MRI image loading, MD5 hash deduplication across splits to eliminate
data leakage, stratified train/val/test splitting, and tf.data pipeline creation.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

CLASS_NAMES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
IMG_SIZE = (224, 224)


def compute_file_hash(filepath: Path) -> str:
    """Computes MD5 hash of an image file to detect duplicate scans."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def load_and_deduplicate_dataset(root_path: str) -> pd.DataFrame:
    """
    Crawls raw directory, catalogs all images, computes hash signatures,
    and removes cross-split or intra-split duplicate scans.
    """
    root = Path(root_path)
    records = []

    for class_idx, class_name in enumerate(CLASS_NAMES):
        for img_path in root.glob(f"**/{class_name}/*.*"):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                records.append({
                    "filepath": str(img_path),
                    "filename": img_path.name,
                    "class_name": class_name,
                    "label": class_idx,
                    "file_hash": compute_file_hash(img_path)
                })

    df = pd.DataFrame(records)
    initial_count = len(df)
    # Deduplicate based on MD5 checksum
    df = df.drop_duplicates(subset=["file_hash"]).reset_index(drop=True)
    dedup_count = len(df)
    logger.info(f"Loaded {initial_count} images. Removed {initial_count - dedup_count} duplicates ({dedup_count} unique remaining).")
    return df


def create_stratified_splits(
    df: pd.DataFrame,
    test_size: float = 0.10,
    val_size: float = 0.10,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Creates leakage-free stratified partitions for train, validation, and test."""
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["label"]
    )
    val_ratio_adjusted = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_ratio_adjusted, random_state=random_state, stratify=train_val_df["label"]
    )

    logger.info(f"Split distributions: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")
    return train_df, val_df, test_df


def parse_image_tf(filename: tf.Tensor, label: tf.Tensor, img_size: Tuple[int, int] = IMG_SIZE) -> Tuple[tf.Tensor, tf.Tensor]:
    """Decodes and resizes an MRI image tensor."""
    image_raw = tf.io.read_file(filename)
    image = tf.io.decode_image(image_raw, channels=3, expand_animations=False)
    image = tf.image.resize(image, img_size)
    return image, label


def build_tf_dataset(
    df: pd.DataFrame,
    batch_size: int = 32,
    is_training: bool = False,
    img_size: Tuple[int, int] = IMG_SIZE
) -> tf.data.Dataset:
    """Builds an optimized, prefetching tf.data.Dataset."""
    paths = df["filepath"].values
    labels = df["label"].values

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(lambda p, l: parse_image_tf(p, l, img_size), num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        dataset = dataset.shuffle(buffer_size=min(len(df), 1024), seed=42)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset
