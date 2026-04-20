# SIMPLE-IVUS

A modular framework for IVUS (Intravascular Ultrasound) image segmentation, supporting multi-stage training, anatomical constraints, and K-Fold cross-validation.

## Prerequisites
- Python 3.8+
- PyTorch 1.10+
- CUDA 11.0+ (optional but recommended)

## Installation
```bash
git clone https://github.com/ZK1523/SIMPLE-IVUS.git
cd SIMPLE-IVUS
pip install -r requirements.txt
```

## Project Structure
```
data/          # Dataset loading & augmentation
models/        # Model architecture adapters (U-Net, etc.)
losses/        # Specialized loss functions for IVUS
utils/         # Metrics (Dice, IoU) & helpers
train1.py      # Multi-stage progressive training
train2.py      # Baseline training
```

## Quick Start

### 1. Progressive multi-stage training
```bash
python train1.py \
    --data-root /path/to/your/dataset \
    --output-dir ./experiments/progressive_training
```

### 2. Baseline training (single stage)
```bash
python train2.py \
    --data-root /path/to/your/dataset \
    --output-dir ./experiments/baseline \
    --batch-size 8 \
    --epochs 200
```


