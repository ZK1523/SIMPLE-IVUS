import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.smooth = smooth
        self.class_weights = class_weights
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred = pred.view(pred.size(0), pred.size(1), -1)
        target = target.view(target.size(0), target.size(1), -1)
        
        intersection = (pred * target).sum(dim=2)
        union = pred.sum(dim=2) + target.sum(dim=2)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        if self.class_weights is not None:
            weights = self.class_weights.view(1, -1)
            return ((1 - dice) * weights).sum() / weights.sum()
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pred_prob = torch.sigmoid(pred)
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_weight * bce_loss
        
        if self.class_weights is not None:
            weights = self.class_weights.view(1, -1, 1, 1)
            focal_loss = focal_loss * weights
        return focal_loss.mean()


class FalsePositivePenalty(nn.Module):
    def __init__(self, penalty_weight: float = 2.0, target_classes: List[int] = [3]):
        super().__init__()
        self.penalty_weight = penalty_weight
        self.target_classes = target_classes
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_prob = torch.sigmoid(pred)
        total_penalty = 0.0
        for cls_idx in self.target_classes:
            pred_cls = pred_prob[:, cls_idx]
            target_cls = target[:, cls_idx]
            false_positive = ((pred_cls > 0.5).float() * (1 - target_cls)).sum()
            total_predictions = (pred_cls > 0.5).float().sum() + 1e-7
            total_penalty += (false_positive / total_predictions)
        return total_penalty * self.penalty_weight / len(self.target_classes)


class ComboLoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5, focal_weight: float = 0.3, 
                 bce_weight: float = 0.2, fp_penalty_weight: float = 0.0, 
                 class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.dice_loss = DiceLoss(class_weights=class_weights)
        self.focal_loss = FocalLoss(class_weights=class_weights)
        self.fp_penalty = FalsePositivePenalty(penalty_weight=fp_penalty_weight)
        
        self.bce_class_weights = class_weights.view(1, -1, 1, 1) if class_weights is not None else None
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.bce_weight = bce_weight
        self.fp_penalty_weight = fp_penalty_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple:
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        bce_raw = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        bce = (bce_raw * self.bce_class_weights).mean() if self.bce_class_weights is not None else bce_raw.mean()
        
        fp_loss = self.fp_penalty(pred, target) if self.fp_penalty_weight > 0 else torch.tensor(0.0, device=pred.device)
        total_loss = self.dice_weight * dice + self.focal_weight * focal + self.bce_weight * bce + fp_loss
        
        loss_dict = {'dice': dice.item(), 'focal': focal.item(), 'bce': bce.item(), 'total': total_loss.item()}
        if self.fp_penalty_weight > 0: loss_dict['fp_penalty'] = fp_loss.item()
        return total_loss, loss_dict


class SingleLabelLossWrapper(nn.Module):
    def __init__(self, multi_label_loss_fn: nn.Module, num_classes: int = 7):
        super().__init__()
        self.multi_label_loss_fn = multi_label_loss_fn
        self.num_classes = num_classes
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_onehot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        return self.multi_label_loss_fn(pred, target_onehot)[0]

def create_loss_function(is_single_label: bool = False, device: str = 'cuda', fp_penalty_weight: float = 1.0) -> nn.Module: 
    class_weights = torch.tensor([1.0, 1.5, 4.0, 0.0, 3.0, 6.0, 30.0], dtype=torch.float32, device=device)
    
    combo_loss = ComboLoss(
        dice_weight=0.6,
        focal_weight=0.25,
        bce_weight=0.15,
        fp_penalty_weight=fp_penalty_weight,
        class_weights=class_weights
    )
    
    return SingleLabelLossWrapper(combo_loss) if is_single_label else combo_loss
