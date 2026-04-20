# SIMPLE-IVUS
A modular framework for IVUS (Intravascular Ultrasound) image segmentation, supporting multi-stage training, anatomical constraints, and K-Fold cross-validation.

## Prerequisites
- Python 3.8+
- PyTorch 1.10+
- CUDA 11.0+

## Installation
Clone this repository:
```bash
git clone https://github.com/ZK1523/SIMPLE-IVUS.git
cd SIMPLE-IVUS
Install the required dependencies:

Bash

pip install -r requirements.txt
Project Structure
data/: Dataset handling and augmentation modules.
models/: Model architecture adapters.
losses/: Specialized loss functions for IVUS segmentation.
utils/: Metrics calculation (Dice, IoU) and helpers.
train1.py: Multi-stage progressive training script.
train2.py: Additional training configurations.
Quick Start
1. Training (Single Stage)
To run the baseline experiment:

Bash

python train2.py \
    --data-root /path/to/your/dataset \
    --output-dir ./experiments/baseline \
    --batch-size 8 \
    --epochs 200
2. Training (Progressive Multi-Stage)
To run the three-stage progressive training:

Bash

python train1.py \
    --data-root /path/to/your/dataset \
    --output-dir ./experiments/progressive_training
Requirements
The requirements.txt includes:

torch, torchvision
numpy, scipy
scikit-learn
tqdm
pillow
