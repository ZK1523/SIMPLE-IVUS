import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except ImportError:
    HAS_SMP = False
    print("Warning: segmentation_models_pytorch not installed")


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, 
                 encoder_name='resnet34', encoder_weights='imagenet'):
        super().__init__()
        
        if HAS_SMP:
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=num_classes,
                decoder_attention_type='scse',  
                activation=None
            )
        else:
            raise ImportError("segmentation_models_pytorch required for AttentionUNet")
    
    def forward(self, x):
        return self.model(x)
