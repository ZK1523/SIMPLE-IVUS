import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Optional

class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 7,
        encoder_name: str = 'resnet34',
        encoder_weights: Optional[str] = None,  
        use_pretrained: bool = False  
    ):
        super(UNet, self).__init__()
        
        if not use_pretrained:
            encoder_weights = None
            
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights, 
            in_channels=in_channels,
            classes=num_classes,
            activation=None 
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def build_unet(in_channels: int = 3, num_classes: int = 7, **kwargs) -> UNet:
    """Factory function to build UNet."""
    return UNet(
        in_channels=in_channels,
        num_classes=num_classes,
        encoder_weights=None,  
        use_pretrained=False,   
        **kwargs
    )
