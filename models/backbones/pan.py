import torch.nn as nn
import segmentation_models_pytorch as smp

class PAN(nn.Module):
    def __init__(
        self, 
        in_channels: int = 3, 
        num_classes: int = 7, 
        encoder_name: str = 'resnet50', 
        encoder_weights: Optional[str] = None
    ):
        super().__init__()
        self.model = smp.PAN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=None  
        )
    
    def forward(self, x):
        return self.model(x)

