import os
import json
import argparse
import numpy as np
from tqdm import tqdm
import torch
from pathlib import Path
from sklearn.model_selection import KFold
import warnings

from data.kfold_dataset_multilabel import IVUSDatasetKFold
from utils.metrics_multilabel import MetricsCalculator
from losses.loss_multilabel import create_single_stage_loss
from models.adapters import AdapterFactory

warnings.filterwarnings('ignore')
NUM_CLASSES = 7
CLASS_NAMES = ['1', '2', '3', '4', '5', '6', '7']

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

def apply_anatomical_constraints(pred_probs, thresholds):
    B, C, H, W = pred_probs.shape
    constrained = torch.zeros_like(pred_probs)
    for b in range(B):
        wy = (pred_probs[b, 2] > thresholds[2]).float()
        constrained[b, 2] = wy
        
        zm = ((pred_probs[b, 0] * (1 - wy)) > thresholds[0]).float()
        constrained[b, 0] = zm
        
        nm = ((pred_probs[b, 1] * zm) > thresholds[1]).float()
        constrained[b, 1] = nm
        
        plaque_region = zm * (1 - nm)
        for i in [4, 5, 6]:
            constrained[b, i] = ((pred_probs[b, i] * plaque_region) > thresholds[i]).float()
        if C > 3:
            constrained[b, 3] = (pred_probs[b, 3] > thresholds[3]).float()
    return constrained

def validate(model, loader, device, calc, thresholds):
    model.eval()
    metrics_log = {'dice': [], 'iou': [], 'per_class': [[] for _ in range(7)]}
    
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            probs = model(images).sigmoid()
            preds = apply_anatomical_constraints(probs, torch.tensor(thresholds).to(device))
            
            for i in range(images.shape[0]):
                for cls in [0, 1, 2, 4, 5, 6]:
                    d = calc.calculate_dice_gpu(preds[i, cls:cls+1], masks[i, cls:cls+1])
                    metrics_log['per_class'][cls].append(d.item())
                    
    results = {
        name: np.mean(metrics_log['per_class'][i]) 
        for i, name in enumerate(['1', '2', '3', '4', '5', '6', '7']) 
        if i in [0, 1, 2, 4, 5, 6]
    }
    return results

def train_single_fold(fold_data, args, device):
    fold_dir = Path(args.output_dir) / f"fold_{fold_data['fold']}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    model = AdapterFactory.get(args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.max_lr)
    criterion = create_single_stage_loss()
    calc = MetricsCalculator()
    
    best_dice = 0.0
    for epoch in range(args.epochs):
        model.train()
        for images, masks in DataLoader(IVUSDatasetKFold(fold_data['train']), batch_size=args.batch_size):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            val_res = validate(model, DataLoader(IVUSDatasetKFold(fold_data['val'])), device, calc, args.thresholds)
            mdice = np.mean([v for k, v in val_res.items() if k != ''])
            if mdice > best_dice:
                best_dice = mdice
                torch.save(model.state_dict(), fold_dir / 'best_model.pth')
    return best_dice

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='./data')
    parser.add_argument('--output-dir', type=str, default='./results')
    parser.add_argument('--model', type=str, default='unet')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--max-lr', type=float, default=1e-3)
    parser.add_argument('--thresholds', type=float, nargs=6, default=[0.5]*6)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    setup_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    folds = get_kfold_splits(args.data_root)
    
    for fold in folds:
        train_single_fold(fold, args, device)

if __name__ == '__main__':
    main()
