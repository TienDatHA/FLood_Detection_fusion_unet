# Fusion_fLood — User Guide

This document describes how to set up the environment, configure, train, evaluate, and run inference for the Fusion_fLood project.

## 🐳 Quick Start with Docker (Recommended)

For the fastest setup with all dependencies pre-configured:

```bash
# Clone and setup
git clone https://github.com/TienDatHA/FLodd_Detection_fusion_unet.git
cd FLodd_Detection_fusion_unet

# Build and run
./docker-helper.sh build
./docker-helper.sh dev    # For development with Jupyter
./docker-helper.sh train  # For training

# See complete Docker guide
open DOCKER_SETUP.md
```

## Overview

This repository contains code to train and evaluate a U-Net model combined with ResNet50 for flood detection based on Sentinel-1 (Sen1Flood11). Main files:

- `config.py` — configuration for paths, image size, batch size, etc.
- `flood.py` — main script for train/evaluate (with custom callbacks, TensorBoard setup, logging).
- `model.py` — model definition (Resnet50_UNet).
- `utils.py` — load data, generator, inference functions, etc.
- `inference_all.py`, `inference_only.py`, `inference_change_detection.py` — inference scripts (for different purposes).
- `merge.py`, `DEM_JRC_post.py`, `Visual_DEM.py` — data processing utilities / post-processing.

## Model Architecture

### Fusion ResNet50-UNet Architecture Overview

The `Resnet50_UNet` model uses a **Dual-Branch Encoder with Shared Decoder** architecture to simultaneously process two types of input data (multi-modal fusion):

```
Input 1 (SAR/Optical) ──► ResNet50 Encoder 1 ──┐
                                                │
                                               Add ──► U-Net Decoder ──► Flood Mask
                                                │
Input 2 (DEM/JRC)     ──► ResNet50 Encoder 2 ──┘
```

### Component Details

#### 1. **Dual ResNet50 Encoders**

**Encoder 1** (`get_resnet50_encoder`):
- Processes primary input (typically SAR imagery from Sentinel-1)
- Standard ResNet50 architecture with ImageNet pretrained weights
- Output feature maps at 4 different scales: `f11, f12, f13, f14`

**Encoder 2** (`get_resnet50_encoder2`):
- Processes auxiliary input (DEM, JRC water data, or topographic information)
- Similar ResNet50 but completely independent (different layer names to avoid conflicts)
- Output feature maps: `f21, f22, f23, f24`

**ResNet50 structure for each encoder:**
```
Input (512x512x3)
    ↓
Conv1 + BN + ReLU + MaxPool        → f1 (128x128x64)
    ↓
Stage 2: 3 blocks [64,64,256]      → f2 (64x64x256)  
    ↓
Stage 3: 4 blocks [128,128,512]    → f3 (32x32x512)
    ↓  
Stage 4: 6 blocks [256,256,1024]   → f4 (16x16x1024)
```

#### 2. **Feature Fusion Layer**

Combines features from two encoders using **Element-wise Addition**:
```python
f1 = Add()([f11, f21])  # 128x128x64
f2 = Add()([f12, f22])  # 64x64x256  
f3 = Add()([f13, f23])  # 32x32x512
f4 = Add()([f14, f24])  # 16x16x1024
```

#### 3. **U-Net Decoder with Skip Connections**

Decoder upsampling and combines features from multiple scales:

```
f4 (16x16x1024)
    ↓ Conv3x3(512) + BN + ReLU
    ↓ UpSample2x → (32x32x512)
    ↓ Concat with f3 → (32x32x1024)
    ↓ Conv3x3(256) + BN + ReLU
    ↓ UpSample2x → (64x64x256)  
    ↓ Concat with f2 → (64x64x512)
    ↓ Conv3x3(128) + BN + ReLU
    ↓ UpSample2x → (128x128x128)
    ↓ Concat with f1 → (128x128x192) [if l1_skip_conn=True]
    ↓ Conv3x3(64) + BN + ReLU  
    ↓ UpSample2x → (256x256x64)
    ↓ Conv1x1(n_classes) + Sigmoid → (256x256x1)
```

**Note:** Model output (256x256) is smaller than input (512x512) due to no final upsampling to full resolution.

#### 4. **Technical Specifications**

- **Input Shape**: 2 inputs, each input (512, 512, 3)
- **Output Shape**: (256, 256, 1) with sigmoid activation
- **Pretrained Weights**: ImageNet for ResNet50 backbone
- **Skip Connections**: U-Net style concatenation in decoder
- **Batch Normalization**: After each Conv layer
- **Activation**: ReLU in hidden layers, Sigmoid in output

#### 5. **Architecture Advantages**

1. **Multi-modal Fusion**: Effectively combines SAR imagery with DEM/JRC data
2. **Pretrained Feature Extraction**: Leverages ImageNet weights for better feature extraction
3. **Multi-scale Features**: Skip connections preserve details at multiple scales
4. **Dual-branch Design**: Allows learning specialized features for each input type
5. **Flexible Input**: Can disable level 1 skip connection if needed (`l1_skip_conn`)

#### 6. **Usage in Code**

```python
# In flood.py
n_classes = 1  # Binary segmentation (flood/no-flood)
in_img = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))    # SAR input  
in_inf = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))    # DEM/JRC input

model = Resnet50_UNet(n_classes, in_img, in_inf, l1_skip_conn=True)
model.compile(optimizer, loss=total_loss, metrics=metrics)
```

#### 7. **Parameters and Memory**

- **Total Parameters**: ~46M (2 × ResNet50 + Decoder)
- **Trainable Parameters**: Depends on freezing strategy in `flood.py`  
- **Memory Usage**: ~8-12GB VRAM for batch_size=1 with 512×512 input
- **Training Strategy**: Phase 1 (freeze encoders) → Phase 2 (unfreeze all)

This architecture is particularly effective for flood detection problems because it combines spectral information (SAR) with topographic information (DEM) and historical water data (JRC).

> Note: The current `config.py` file has `ROOT_PATH` hard-coded. If your machine doesn't have that path, please edit `config.py` (see "Configuration" section below) before running.

## Environment Requirements

- Operating System: Linux (check with `uname -a`).
- Python 3.8+ (recommended 3.8–3.10). Can use `venv` or conda.
- GPU (if using GPU training) and CUDA/cuDNN drivers compatible with your TensorFlow version.

Suggested Python packages (example for quick installation):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install tensorflow numpy scikit-image opencv-python matplotlib tqdm pathlib segmentation-models
```

Specific versions depend on your system; if you encounter compatibility errors, try changing the `tensorflow` version.

## Configuration Before Running

Open the `config.py` file and check the following fields:

- `ROOT_PATH` — root path containing the dataset and/or where you want to save weights/outputs.
  - Default in repo: `/mnt/hdd2tb/Uni-Temporal-Flood-Detection-Sentinel-1_Frontiers22`.
  - If your dataset is not there, change this value to an appropriate path or create an alias (e.g., create symlink).
- `SEN1FLOODS_PATH`, `IMG_PATH`, `LABEL_PATH`, `DEM_PATH`, `JRC_PATH` — sub-paths built from `ROOT_PATH`.
- `WEIGHT_PATH` and `OUT_FOLDER` — where to save weights and outputs.
- `WEIGHT_FILE` — default checkpoint filename (`standard_checkpoint.h5`).
- `IMG_HEIGHT` and `IMG_WIDTH` — input size for model (default 512x512). Check `model.py` if model requires specific multiples (e.g., multiple of 32).
- `train_batchSize`, `val_batchSize` — batch size; default is 1 (to avoid OOM). You can increase if GPU has enough memory.

Currently `config.py` creates `WEIGHT_PATH` and `OUT_FOLDER` directories at import level (via `mkdir`). If you want to avoid side-effects during import, you can change it to call directory creation from the main script before writing files.

## Data Types / Dataset Directory Structure

The repository expects the Sen1Flood11 dataset to have a structure similar to the paths in `config.py` (`Sen1Flood11/v1.1/...`). The `IMG_PATH` and `LABEL_PATH` variables point to directories containing images and corresponding labels (HandLabeled in current configuration).

If you use a different location, change `ROOT_PATH` or corresponding paths in `config.py`.

## Training

The main script for training is `flood.py`. Command-line:

```bash
# Run training
python3 flood.py train

# If you want to disable Smart Early Stopping
python3 flood.py train --no-early-stopping

# Customize patience parameters from CLI
python3 flood.py train --early-stopping-patience 20 --overfitting-patience 12
```

Details of `flood.py` behavior:
- When calling `train`, the script will:
  - Call `load_data()` from `utils.py` to load `train_x, train_y, val_x, val_y`.
  - Create dataset generators `Cust_DatasetGenerator` with `train_batchSize` and `val_batchSize` from `config.py`.
  - Initialize model `Resnet50_UNet` from `model.py`, compile with optimizer and loss (combination of Dice + Focal loss), set scheduler `ReduceLROnPlateau`.
  - Phase 1: freeze encoder, train 2 epochs (according to current code).
  - Phase 2: unfreeze all, train for 100 more epochs (can be modified in code).
  - Callbacks: TensorBoard, BestModelLogger (save multiple types of checkpoints), ModelCheckpoint (save `standard_checkpoint.h5` based on `val_loss`), SmartEarlyStopping (optional).

Outputs after training:
- `training_logs/` — where logs and checkpoints are saved (`best_val_loss_checkpoint.h5`, `best_iou_checkpoint.h5`, `best_performance_checkpoint.h5`, `training_log.txt`).
- `WEIGHT_PATH/` — where default `standard_checkpoint.h5` is saved (according to `config.py`).

Training tips:
- If OOM: reduce `train_batchSize` or `IMG_HEIGHT`/`IMG_WIDTH` in `config.py`.
- Let TensorFlow use GPU memory growth (already included in `flood.py`).
- Check TensorBoard logs: from repo root run

```bash
tensorboard --logdir training_logs
```

## Evaluation

To evaluate the model on the loaded validation set, use:

```bash
python3 flood.py evaluate
```

Behavior:
- `evaluate` will try to find checkpoints in order: `training_logs/best_performance_checkpoint.h5`, `training_logs/best_iou_checkpoint.h5`, `training_logs/best_val_loss_checkpoint.h5`, then fallback to `WEIGHT_PATH / WEIGHT_FILE`.
- If no checkpoint is found, the script will print `No checkpoint found!` and exit.
- Evaluation results (IoU, intersection/union) are written to `evaluation_logs` folder and `evaluation_log.txt` file.

## Inference / Prediction Generation

The repository has several inference-related scripts:

- `inference_only.py` — can be used to run inference on individual images/samples.
- `inference_all.py` — run inference on entire dataset (depending on internal code).
- `inference_change_detection.py` — inference for change-detection problems (if appropriate data is available).

Each script may have its own CLI; if not, you can import functions from `utils.py` (`Inference`) or call `evaluate_fusion()` function from `flood.py` (this function also performs inference and writes `Pred_Mask` in `WEIGHT_PATH`).

Example: running evaluate will call `log_evaluation_results()` and create outputs in `WEIGHT_PATH/Pred_Mask`.

## Saving Weights and Checkpoints

- Default checkpoint: `WEIGHT_PATH / WEIGHT_FILE`, default name `standard_checkpoint.h5`.
- Best model logger will save additional checkpoint files by metric: `best_val_loss_checkpoint.h5`, `best_iou_checkpoint.h5`, `best_performance_checkpoint.h5`.

## Debug / Troubleshooting

- "No checkpoint found" error: check `training_logs/` and `WEIGHT_PATH` for any `.h5` files.
- OOM error: reduce batch size; enable memory growth (already included in `flood.py`).
- Dataset path error: edit `ROOT_PATH` in `config.py` or place dataset in default path.
- `segmentation_models` import error: `segmentation-models` package needs to be installed and framework set `sm.set_framework('tf.keras')` (already included in `flood.py`). Often need to install `efficientnet` when using EfficientNet backbones.

## Minor Improvement Suggestions

- Move directory creation (`mkdir`) from import-time in `config.py` to an `ensure_dirs()` function and call it at entrypoint (`flood.py`) to avoid side-effects when importing the module in unit tests or other scripts.
- Allow overriding `ROOT_PATH` with environment variable `PROJECT_ROOT` for easier switching between machines/local/cluster.

## Other Useful Commands

```bash
# Run training in background with nohup
nohup python3 flood.py train > train.out 2>&1 &

# Run evaluate and log output
python3 flood.py evaluate > eval.out 2>&1
```

## Contact
If you need additional support (debugging specific errors, updating config to use environment variables, or creating a patch to move mkdir out of import), let me know — I can modify `config.py` and update the repo directly.

---
This file outlines the necessary steps to run the project; if you want me to create a standard `requirements.txt` or modify `config.py` to avoid side-effects during import, I can proceed with that.
