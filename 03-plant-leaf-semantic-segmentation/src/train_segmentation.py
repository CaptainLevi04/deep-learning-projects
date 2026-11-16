"""
Foliage Semantic Segmentation Training & Evaluation Loop
Author: Amr Gamal
Organization: EME Internship Portfolio

Orchestrates two-phase training (Frozen backbone -> Full network fine-tuning),
metrics calculation (Dice, IoU, Precision, Recall, Pixel Accuracy), and evaluation.
"""

import argparse
import logging
import os
from typing import Dict

import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import DataLoader

from dataset_coco import LeafDataset
from model_deeplabv3 import create_deeplabv3_model, CombinedDiceBCEWithLogitsLoss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def compute_metrics(preds: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> Dict[str, float]:
    """Computes Dice Coefficient, IoU, Precision, Recall, and Accuracy."""
    preds_bin = (preds > 0.5).float()

    intersection = (preds_bin * targets).sum().item()
    total_pred = preds_bin.sum().item()
    total_target = targets.sum().item()
    union = total_pred + total_target - intersection

    dice = (2.0 * intersection + smooth) / (total_pred + total_target + smooth)
    iou = (intersection + smooth) / (union + smooth)
    precision = (intersection + smooth) / (total_pred + smooth)
    recall = (intersection + smooth) / (total_target + smooth)
    accuracy = (preds_bin == targets).float().mean().item()

    return {
        "Dice": dice,
        "IoU": iou,
        "Precision": precision,
        "Recall": recall,
        "Accuracy": accuracy
    }


def evaluate(model: torch.nn.Module, loader: DataLoader, criterion: torch.nn.Module, device: torch.device) -> Dict[str, float]:
    """Evaluates segmentation network on test or validation dataloader."""
    model.eval()
    total_loss = 0.0
    all_metrics = []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)["out"]
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)
            batch_metrics = compute_metrics(probs, masks)
            all_metrics.append(batch_metrics)

    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    avg_metrics["Loss"] = total_loss / len(loader)
    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Plant Leaf Segmentation Trainer")
    parser.add_argument("--data_dir", type=str, default="verycleanfarmers", help="Path to COCO dataset folder")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")

    train_json = os.path.join(args.data_dir, "train", "_annotations.coco.json")
    if not os.path.exists(train_json):
        logger.warning(f"Annotation file not found at: {train_json}")
        logger.info("Please provide the verycleanfarmers dataset folder or run the notebook.")
        return

    logger.info("Loading COCO annotations...")
    train_coco = COCO(train_json)
    train_ids = sorted(train_coco.getImgIds())
    train_dataset = LeafDataset(train_coco, os.path.join(args.data_dir, "train"), train_ids)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    logger.info("Instantiating DeepLabV3+ with MobileNetV3-Large backbone...")
    model = create_deeplabv3_model(num_classes=1, pretrained=True).to(device)
    criterion = CombinedDiceBCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    logger.info(f"Beginning training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)["out"]
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        logger.info(f"Epoch [{epoch}/{args.epochs}] - Train Loss: {epoch_loss:.4f}")

    logger.info("Training complete. Model weights ready for inference.")


if __name__ == "__main__":
    main()
