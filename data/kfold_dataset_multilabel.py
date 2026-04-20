import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Optional


LABEL_PRIORITY = {0: 0, 1: 1, 2: 2, 4: 4, 5: 5, 6: 6}

class IVUSDatasetKFold(Dataset):
    
    def __init__(self, 
                 file_list: List[str], 
                 images_dir: str, 
                 masks_dir: str, 
                 transform: Optional[callable] = None, 
                 is_single_label: bool = False):
        
        self.file_list = file_list
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.is_single_label = is_single_label
        
        print(f"Dataset initialized:")
        print(f"Mode: {'Single-Label (Baseline)' if is_single_label else 'Multi-Label (Proposed)'}")
        print(f"Samples: {len(file_list)}")
    
    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, idx: int):
        file_id = self.file_list[idx]
        
        img_path = self.images_dir / f'{file_id}.png'
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        mask_path = self.masks_dir / f'{file_id}.npy'
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        mask = np.load(mask_path)
        
        if mask.ndim == 2:
            mask = mask[..., np.newaxis]
        if mask.shape[-1] == 8:
            mask = mask[..., 1:]
        
        if self.is_single_label:
            single_mask = np.zeros(mask.shape[:2], dtype=np.int64)
            for ch_idx, target_val in LABEL_PRIORITY.items():
                single_mask[mask[..., ch_idx] == 1] = target_val
            mask = single_mask
        
        if self.transform:
            image_gray, mask = self.transform(image_gray, mask)
        
        if image_gray.ndim == 3:
            image_gray = image_gray[..., 0]
        image_rgb = np.stack([image_gray, image_gray, image_gray], axis=0)
        image_tensor = torch.from_numpy(image_rgb).float() / 255.0
        
        if self.is_single_label:
            mask_tensor = torch.from_numpy(mask).long()
        else:
            mask = np.transpose(mask, (2, 0, 1))
            mask_tensor = torch.from_numpy(mask).float()
        
        self._validate_shapes(image_tensor, mask_tensor)
        
        return image_tensor, mask_tensor

    def _validate_shapes(self, image_tensor: torch.Tensor, mask_tensor: torch.Tensor):
        assert image_tensor.shape == (3, 512, 512), \
            f"Image shape mismatch: {image_tensor.shape}"
        
        if self.is_single_label:
            assert mask_tensor.shape == (512, 512), \
                f"Single-label mask shape mismatch: {mask_tensor.shape}"
        else:
            assert mask_tensor.shape == (7, 512, 512), \
                f"Multi-label mask shape mismatch: {mask_tensor.shape}"
