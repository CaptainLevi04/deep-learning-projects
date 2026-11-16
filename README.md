# Applied Deep Learning Portfolio Repository

A high-performance portfolio of production-grade Deep Learning systems spanning **Computer Vision**, **Medical Imaging Diagnostics**, **Audio Signal Processing**, and **Precision Agriculture Semantic Segmentation**.

Developed during the **EME Training Program** by **Amr Gamal**.

---

## 🚀 Key Engineering Highlights & SOTA Results

- **Clinical Brain Tumor MRI Classification (90.76% Accuracy)**:
  - Diagnosed data leakage and distribution shift in standard public partitions.
  - Implemented an **MD5 checksum hash deduplication pipeline** to purge cross-split redundant scans.
  - Executed a **two-stage transfer learning fine-tuning protocol** with **EfficientNetB0**, boosting test accuracy from 72.61% to **90.76%**.
- **PyTorch Precision Agriculture Segmentation (98.07% Dice / 96.21% IoU)**:
  - Engineered an end-to-end foliage segmentation system using **DeepLabV3+ with MobileNetV3-Large** and Atrous Spatial Pyramid Pooling (ASPP).
  - Designed a custom **Combined Dice + BCE Loss** and trained on custom COCO annotations ([`amrgamal007/verycleanfarmers`](https://www.kaggle.com/datasets/amrgamal007/verycleanfarmers)).
  - Reached **98.14% pixel accuracy** across 2.4 million test pixels.
- **Audio Music Genre Classification via Mel-Spectrograms (75.68% Val Accuracy)**:
  - Formulated audio classification as a 2D Computer Vision task by generating **Mel-Spectrogram frequency-time representations** of 1,000 GTZAN tracks.
  - Benchmarked scratch CNN against **MobileNetV2 Transfer Learning**, achieving a massive jump from random baseline to **75.68% validation accuracy** across 10 musical genres.

---

## 📂 Project Catalog

| Directory | Domain | Modality | Framework | Model Architecture | Primary Benchmark |
|---|---|---|---|---|---|
| [`01-brain-tumor-mri-classification`](./01-brain-tumor-mri-classification) | Medical Diagnostics | Clinical MRI Scans | TensorFlow / Keras | **EfficientNetB0** (Fine-Tuned) | **90.76% Test Accuracy** |
| [`02-gtzan-music-genre-classification`](./02-gtzan-music-genre-classification) | Audio Information Retrieval | Mel-Spectrograms | TensorFlow / Keras | **MobileNetV2** (Fine-Tuned) | **75.68% Val / 71.62% Test Acc** |
| [`03-plant-leaf-semantic-segmentation`](./03-plant-leaf-semantic-segmentation) | Precision Agriculture | COCO Drone/Field Imagery | PyTorch / Torchvision | **DeepLabV3+ (MobileNetV3)** | **98.07% Dice / 96.21% IoU** |

---

## 🛠️ Technology Stack & Architectures

- **Frameworks**: PyTorch 2.2+, TensorFlow 2.15+, Keras, Torchvision
- **Computer Vision & Audio**: OpenCV (`cv2`), PIL, `pycocotools`, Mel-Spectrogram transforms
- **Core Architectures**:
  - **EfficientNetB0** (Compound scaling, inverted residual bottleneck blocks)
  - **DeepLabV3+** (Atrous Spatial Pyramid Pooling for multi-scale context)
  - **MobileNetV3-Large & MobileNetV2** (Depthwise separable convolutions for edge deployment)
  - Custom Deep Convolutional Networks with Batch Normalization & Dropout
- **Optimization Techniques**:
  - Two-Stage Transfer Learning (Frozen Feature Extraction $\to$ Partial Layer Unfreezing)
  - Learning Rate Scheduling (`ReduceLROnPlateau`, Adam, AdamW with weight decay)
  - Hybrid Objective Formulations (Dice Loss + Binary Cross-Entropy)
  - MD5 Data Deduplication & Leakage Prevention

---

## 📁 Repository Layout

```
deep-learning-projects/
├── 01-brain-tumor-mri-classification/
│   ├── README.md                            # Medical imaging documentation & leakage analysis
│   ├── brain_tumor_classification.ipynb     # Interactive research notebook with all visual outputs
│   └── src/
│       ├── dataset.py                       # MD5 deduplication & tf.data streaming
│       ├── models.py                        # EfficientNetB0, MobileNetV2 & Custom CNN
│       └── train.py                         # Two-stage progressive fine-tuning pipeline
├── 02-gtzan-music-genre-classification/
│   ├── README.md                            # Spectrogram & audio ML documentation
│   ├── music_genre_spectrogram_classification.ipynb # Spectrogram modeling notebook
│   └── src/
│       └── train_audio_mobilenet.py         # Modular spectrogram training script
├── 03-plant-leaf-semantic-segmentation/
│   ├── README.md                            # PyTorch semantic segmentation documentation
│   ├── leaf_semantic_segmentation_deeplabv3.ipynb # PyTorch DeepLabV3+ notebook
│   └── src/
│       ├── dataset_coco.py                  # PyTorch COCO polygon parsing
│       ├── model_deeplabv3.py               # DeepLabV3+ model & Combined Dice Loss
│       └── train_segmentation.py            # PyTorch training & metric evaluation loop
├── .gitignore                               # PyTorch, TensorFlow & data ignore rules
├── requirements.txt                         # Pinned dependencies
└── README.md                                # Deep learning portfolio index
```

---

## 💻 Environment Setup & Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/amrgamal007/deep-learning-projects.git
cd deep-learning-projects
```

### 2. Set up virtual environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Execute a pipeline
```bash
# Train Brain Tumor EfficientNetB0 classifier
python 01-brain-tumor-mri-classification/src/train.py --arch efficientnetb0 --epochs 15

# Train Foliage Segmentation with PyTorch DeepLabV3+
python 03-plant-leaf-semantic-segmentation/src/train_segmentation.py --epochs 10
```

---

## 👤 Author & Acknowledgments

- **Engineer**: Amr Gamal ([@amrgamal007](https://github.com/amrgamal007))
- **Program**: EME Artificial Intelligence & Deep Learning Training Program
