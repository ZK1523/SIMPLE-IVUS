import torch.nn as nn
try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except ImportError:
    HAS_SMP = False
    print("Warning: segmentation_models_pytorch not installed")

class DeepLabV3Plus(nn.Module):
    def __init__(self, in_channels=3, num_classes=7,  
                 encoder_name='resnet101', encoder_weights=None):
        super().__init__()
        if HAS_SMP:
            self.model = smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=num_classes,
                activation=None  
            )
        else:
            raise ImportError("segmentation_models_pytorch required for DeepLabV3Plus")
    
    def forward(self, x):
        return self.model(x)
