"""
COCO-Format Leaf Semantic Segmentation Dataset
Author: Amr Gamal
Organization: EME Internship Portfolio

Custom PyTorch Dataset for loading high-resolution crop foliage images and generating
binary segmentation ground-truth masks from COCO JSON polygon annotations.
"""

import os
from typing import Tuple, List

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class LeafDataset(Dataset):
    """
    PyTorch Dataset loading images and synthesizing binary segmentation masks
    from COCO polygon annotation format using pycocotools.
    """
    def __init__(
        self,
        coco: COCO,
        image_dir: str,
        image_ids: List[int],
        image_size: Tuple[int, int] = (256, 256)
    ):
        self.coco = coco
        self.image_dir = image_dir
        self.image_ids = image_ids
        self.image_size = image_size

        # ImageNet normalization constants
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.image_dir, img_info["file_name"])

        # Load RGB image
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        # Generate binary mask from COCO polygon annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

        for ann in anns:
            mask = np.maximum(mask, self.coco.annToMask(ann))

        # Resize image and mask
        image_resized = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)

        # Normalize image to [0, 1] then apply ImageNet standardization
        image_norm = image_resized.astype(np.float32) / 255.0
        image_norm = (image_norm - self.mean) / self.std

        # Convert to PyTorch Tensors (C, H, W)
        image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask_resized).unsqueeze(0).float()

        return image_tensor, mask_tensor
