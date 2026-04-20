import cv2
import numpy as np
import random
from scipy.ndimage import map_coordinates, gaussian_filter
from typing import Tuple, Optional

class BaselineAugmentation:
    """No augmentation (Baseline)"""
    def __call__(self, image, mask):
        return image, mask


class BasicAugmentation:
    """Basic augmentation supporting multi-channel masks"""
    def __init__(self, prob=0.8):
        self.prob = prob
    
    def __call__(self, image, mask):
        if random.random() > self.prob:
            return image, mask
        
        if len(image.shape) == 3 and image.shape[-1] == 3:
            image_rgb = image.copy()
        else:
            if len(image.shape) == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = image.copy()
        
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            h, w = image_rgb.shape[:2]
            center = (w//2, h//2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            image_rgb = cv2.warpAffine(image_rgb, M, (w, h), 
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT_101)
            
            if mask is not None:
                if mask.ndim == 2:
                    mask = cv2.warpAffine(mask, M, (w, h),
                                         flags=cv2.INTER_NEAREST,
                                         borderMode=cv2.BORDER_REFLECT_101)
                else: 
                    mask_channels = []
                    for c in range(mask.shape[-1]):
                        mask_c = cv2.warpAffine(mask[..., c], M, (w, h),
                                               flags=cv2.INTER_NEAREST,
                                               borderMode=cv2.BORDER_REFLECT_101)
                        mask_channels.append(mask_c)
                    mask = np.stack(mask_channels, axis=-1)
        
        if random.random() > 0.5:
            image_rgb = cv2.flip(image_rgb, 1)
            if mask is not None:
                mask = cv2.flip(mask, 1)
        
        if random.random() > 0.5:
            image_rgb = cv2.flip(image_rgb, 0)
            if mask is not None:
                mask = cv2.flip(mask, 0)
        
        if random.random() > 0.5:
            alpha = random.uniform(0.9, 1.1)
            beta = random.uniform(-10, 10)
            
            if image_rgb.dtype == np.uint8:
                image_rgb = np.clip(alpha * image_rgb + beta, 0, 255).astype(np.uint8)
            else:
                image_rgb = np.clip(alpha * image_rgb + beta, 0, 1.0)
        
        if random.random() > 0.7:
            noise = np.random.normal(0, 5, image_rgb.shape)
            if image_rgb.dtype == np.uint8:
                image_rgb = np.clip(image_rgb + noise, 0, 255).astype(np.uint8)
            else:
                image_rgb = np.clip(image_rgb + noise, 0, 1.0)
        
        if image.shape[-1] != 3 and len(image_rgb.shape) == 3 and image_rgb.shape[-1] == 3:
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        return image_rgb, mask


class IVUSSpecificAugmentationFixed:
    """Optimized IVUS-specific augmentation to preserve anatomical structures"""
    def __init__(self, prob=0.5):
        self.prob = prob
        self.clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    
    def _safe_elastic_transform(self, image, mask, alpha=2, sigma=1):
        if image is None:
            return image, mask
        
        shape = image.shape
        h, w = shape[:2]
        
        dx = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
        
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        indices = np.array([(y + dy).ravel(), (x + dx).ravel()])
        
        if len(shape) == 3:
            image_channels = []
            for c in range(shape[2]):
                channel = map_coordinates(image[..., c], indices, order=1, mode='reflect')
                image_channels.append(channel.reshape(h, w))
            image_transformed = np.stack(image_channels, axis=-1).astype(image.dtype)
        else:
            image_transformed = map_coordinates(image, indices, order=1, mode='reflect')
            image_transformed = image_transformed.reshape(shape).astype(image.dtype)
        
        if mask is not None:
            if mask.ndim == 2:
                mask_transformed = map_coordinates(mask, indices, order=0, mode='reflect')
                mask_transformed = mask_transformed.reshape(shape[:2]).astype(mask.dtype)
            else:
                mask_channels = []
                for c in range(mask.shape[-1]):
                    mask_c = map_coordinates(mask[..., c], indices, order=0, mode='reflect')
                    mask_channels.append(mask_c.reshape(h, w))
                mask_transformed = np.stack(mask_channels, axis=-1).astype(mask.dtype)
        else:
            mask_transformed = mask
        
        return image_transformed, mask_transformed
    
    def __call__(self, image, mask):
        if random.random() > self.prob:
            return image, mask
        
        if len(image.shape) == 3 and image.shape[-1] == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image.copy()
        
        if random.random() > 0.7:
            alpha = random.uniform(1, 2)
            sigma = random.uniform(0.5, 1)
            image_gray, mask = self._safe_elastic_transform(image_gray, mask, alpha, sigma)
        
        if random.random() > 0.5:
            hist, _ = np.histogram(image_gray.flatten(), 256, [0,256])
            cdf = hist.cumsum()
            cdf_normalized = cdf / cdf.max()
            if cdf_normalized[200] < 0.85:
                image_gray = self.clahe.apply(image_gray)
        
        if random.random() > 0.7:
            h, w = image_gray.shape
            center_x, center_y = w//2, h//2
            radius = random.randint(3, 8)
            y, x = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            shadow_mask = dist_from_center <= radius
            image_gray[shadow_mask] = np.clip(image_gray[shadow_mask] * 0.3, 0, 255)
        
        if random.random() > 0.6:
            speckle = np.random.normal(1.0, 0.05, image_gray.shape)
            image_gray = np.clip(image_gray * speckle, 0, 255).astype(np.uint8)
        
        return image_gray, mask


class FullAugmentationFixed:
    """Combined augmentation strategy"""
    def __init__(self, prob=0.5):
        self.prob = prob
        self.basic_aug = BasicAugmentation(prob=prob)
        self.ivus_aug = IVUSSpecificAugmentationFixed(prob=prob*0.3)
    
    def __call__(self, image, mask):
        image, mask = self.basic_aug(image, mask)
        if random.random() < 0.3:
            image, mask = self.ivus_aug(image, mask)
        return image, mask


class AdaptiveAugmentation:
    """Image-aware adaptive augmentation"""
    def __init__(self, prob=0.7):
        self.prob = prob
        self.basic_aug = BasicAugmentation(prob=0.5)
        self.ivus_aug = IVUSSpecificAugmentationFixed(prob=0.3)
    
    def _analyze_image(self, image):
        if len(image.shape) == 3 and image.shape[-1] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        hist, _ = np.histogram(gray.flatten(), 256, [0,256])
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf.max()
        contrast_score = cdf_normalized[200]
        return {
            'needs_clahe': contrast_score < 0.8,
            'needs_elastic': True,
            'needs_basic': True
        }
    
    def __call__(self, image, mask):
        if random.random() > self.prob:
            return image, mask
        
        analysis = self._analyze_image(image)
        image, mask = self.basic_aug(image, mask)
        
        if analysis['needs_clahe'] and random.random() < 0.5:
            if len(image.shape) == 3 and image.shape[-1] == 3:
                image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
                image_gray = clahe.apply(image_gray)
                image = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
            else:
                clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
                image = clahe.apply(image)
        
        if analysis['needs_elastic'] and random.random() < 0.3:
            image, mask = IVUSSpecificAugmentationFixed()._safe_elastic_transform(
                image, mask, alpha=1, sigma=0.5
            )
        return image, mask

if __name__ == "__main__":
    pass


