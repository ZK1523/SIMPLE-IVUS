import os
import json
import argparse
import numpy as np
from tqdm import tqdm
import torch
from pathlib import Path
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import warnings
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt

warnings.filterwarnings('ignore')

NUM_CLASSES = 7
CLASS_NAMES = ['1', '2', '3', '4', '5', '6', '7']

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from data.kfold_dataset_multilabel import IVUSDatasetKFold
from utils.metrics_multilabel import MetricsCalculator, format_metrics
from losses.loss_multilabel import create_loss
from models.adapters import AdapterFactory

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_kfold_splits(processed_dir, n_splits=5, seed=42):
    images_dir = Path(processed_dir) / 'images'
    all_files = sorted([f.stem for f in images_dir.glob('*.png')])
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(all_files)):
        folds.append({
            'fold': fold_idx + 1,
            'train': [all_files[i] for i in train_idx],
            'val': [all_files[i] for i in val_idx]
        })
    return folds

def train_one_epoch(model, loader, criterion, optimizer, epoch, stage_name, device):
    model.train()
    epoch_losses = {'total': 0}
    pbar = tqdm(loader, desc=f'[{stage_name}] Epoch {epoch+1}', ncols=100)
    for images, masks in pbar:
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        if isinstance(outputs, tuple): outputs = outputs[0]
        loss_result = criterion(outputs, masks)
        loss, loss_dict = (loss_result if isinstance(loss_result, tuple) else (loss_result, {'total': loss_result.item()}))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        for key in loss_dict:
            if key in epoch_losses: epoch_losses[key] += loss_dict[key]
            else: epoch_losses[key] = loss_dict[key]
    return {k: v / len(loader) for k, v in epoch_losses.items()}

def validate_fast(model, loader, device, metrics_calculator):
    model.eval()
    CLASSES_TO_COMPUTE = [0, 1, 2, 4, 5, 6]
    class_thresholds = torch.tensor([0.5] * 7).to(device)
    for cls_idx in CLASSES_TO_COMPUTE: class_thresholds[cls_idx] = 0.5
    
    batch_metrics = {'dice': [], 'iou': [], 'per_class_dice': [[] for _ in range(6)], 'per_class_iou': [[] for _ in range(6)]}
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            outputs = model(images)
            probs = (outputs[0] if isinstance(outputs, tuple) else outputs).sigmoid()
            for sample_idx in range(images.shape[0]):
                sample_dice, sample_iou = [], []
                for list_idx, cls_idx in enumerate(CLASSES_TO_COMPUTE):
                    pred_cls = (probs[sample_idx, cls_idx] > class_thresholds[cls_idx]).float().unsqueeze(0)
                    target_cls = masks[sample_idx, cls_idx].unsqueeze(0)
                    d = metrics_calculator.calculate_dice_gpu(pred_cls, target_cls).item()
                    i = metrics_calculator.calculate_iou_gpu(pred_cls, target_cls).item()
                    sample_dice.append(d)
                    sample_iou.append(i)
                    batch_metrics['per_class_dice'][list_idx].append(d)
                    batch_metrics['per_class_iou'][list_idx].append(i)
                batch_metrics['dice'].append(np.mean(sample_dice))
                batch_metrics['iou'].append(np.mean(sample_iou))
    
    final_metrics = {'mdice': np.mean(batch_metrics['dice']), 'miou': np.mean(batch_metrics['iou'])}
    for i, name in enumerate(['1', '2', '3', '4', '5', '6']):
        final_metrics[name] = {'dice': np.mean(batch_metrics['per_class_dice'][i]), 'iou': np.mean(batch_metrics['per_class_iou'][i])}
    return final_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='data_root = os.getenv('DATA_ROOT', str(BASE_DIR / 'data'))')
    parser.add_argument('--mask-dir', type=str, default='masks_multilabel')
    parser.add_argument('--output-dir', type=str, default='./experiments/exp1')
    parser.add_argument('--model', type=str, default='unet')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-folds', type=int, default=5)
    args = parser.parse_args()
    
    setup_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    folds = get_kfold_splits(args.data_root, args.n_folds)

if __name__ == '__main__':
    main()
