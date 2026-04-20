import torch
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion
from typing import List, Optional, Dict, Any

class MetricsCalculator:
    def __init__(
        self, 
        num_classes: int = 7, 
        class_names: Optional[List[str]] = None,
        exclude_classes: Optional[List[int]] = None
    ):
        self.num_classes = num_classes
        self.exclude_classes = exclude_classes or []
        self.class_range = [i for i in range(num_classes) if i not in self.exclude_classes]
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]

    def calculate_dice(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
        pred = pred.flatten()
        target = target.flatten()
        intersection = (pred * target).sum()
        return float((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))
    
    def calculate_iou(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
        pred = pred.flatten()
        target = target.flatten()
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        return float((intersection + smooth) / (union + smooth))
    
    def calculate_binary_metrics(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> tuple:
        pred = pred.flatten()
        target = target.flatten()
        tp = ((pred == 1) & (target == 1)).sum().float()
        tn = ((pred == 0) & (target == 0)).sum().float()
        fp = ((pred == 1) & (target == 0)).sum().float()
        fn = ((pred == 0) & (target == 1)).sum().float()
        
        recall = (tp + smooth) / (tp + fn + smooth)
        precision = (tp + smooth) / (tp + fp + smooth)
        accuracy = (tp + tn + smooth) / (tp + tn + fp + fn + smooth)
        return float(recall), float(precision), float(accuracy)
    
    def calculate_hd95(self, pred: np.ndarray, target: np.ndarray) -> float:
        pred = pred.astype(bool)
        target = target.astype(bool)
        
        if pred.sum() == 0 and target.sum() == 0:
            return 0.0
        if pred.sum() == 0 or target.sum() == 0:
            return 100.0
        
        pred_dt = distance_transform_edt(~pred)
        target_dt = distance_transform_edt(~target)
        
        pred_border = pred ^ binary_erosion(pred)
        target_border = target ^ binary_erosion(target)
        
        if pred_border.sum() == 0 or target_border.sum() == 0:
            return 0.0
        
        distances = np.concatenate([pred_dt[target_border], target_dt[pred_border]])
        return float(np.percentile(distances, 95))
    
    def calculate_metrics(self, preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
        metrics = {'classes': {}, 'average': {}}
        results = {k: [] for k in ['dice', 'iou', 'recall', 'precision', 'accuracy', 'hd95']}
        
        for cls_idx in self.class_range:
            pred_cls = preds[:, cls_idx]
            target_cls = targets[:, cls_idx]
            
            dice = self.calculate_dice(pred_cls, target_cls)
            iou = self.calculate_iou(pred_cls, target_cls)
            recall, precision, acc = self.calculate_binary_metrics(pred_cls, target_cls)
            
            hd95_list = [
                self.calculate_hd95(pred_cls[b].cpu().numpy(), target_cls[b].cpu().numpy()) 
                for b in range(preds.shape[0])
            ]
            hd95 = np.mean(hd95_list)
            
            cls_name = self.class_names[cls_idx]
            metrics['classes'][cls_name] = {
                'dice': dice, 'iou': iou, 'recall': recall, 
                'precision': precision, 'accuracy': acc, 'hd95': hd95
            }
            
            for k in results.keys():
                results[k].append(locals()[k] if k != 'hd95' else hd95)
        
        for k, v in results.items():
            metrics['average'][k] = np.mean(v)
            
        return metrics

