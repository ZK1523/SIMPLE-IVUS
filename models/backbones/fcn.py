import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class FCN(nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = None,
        in_channels: int = 3,
        num_classes: int = 7,  
        **kwargs
    ):
        super().__init__()   
        self.model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes, 
            **kwargs
        )
    
    def forward(self, x):
        return self.model(x)

__all__ = ['FCN']
