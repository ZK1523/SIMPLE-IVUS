import torch.nn as nn
import segmentation_models_pytorch as smp

class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, encoder_name='resnet34', encoder_weights='imagenet'): 
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=num_classes,
            activation=None   
        )
    
    def forward(self, x):
        return self.model(x)
