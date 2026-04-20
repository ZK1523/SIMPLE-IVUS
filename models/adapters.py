import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
from abc import ABC, abstractmethod

class BaseAdapter(nn.Module, ABC):
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.model = self._build_model()
        if config.get('pretrained_path'):
            self.load_pretrained(config['pretrained_path'])
    
    @abstractmethod
    def _build_model(self) -> nn.Module:
        pass
    
    @abstractmethod
    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        pass
    
    def postprocess(self, output: torch.Tensor, original_size: Tuple[int, int]) -> torch.Tensor:
        if output.shape[2:] != original_size:
            output = nn.functional.interpolate(
                output, size=original_size, mode='bilinear', align_corners=False
            )
        return output
    
    def forward(self, x: torch.Tensor, return_original_size: bool = True) -> torch.Tensor:
        original_size = x.shape[2:]
        x_processed = self.preprocess(x)
        output = self.model(x_processed)
        if return_original_size:
            output = self.postprocess(output, original_size)
        return output
    
    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return self.forward(image)

    def load_pretrained(self, path: str, strict: bool = False):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        self.model.load_state_dict(state_dict, strict=strict)

    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        pass
    
    def to(self, device):
        super().to(device)
        if hasattr(self, 'model'):
            self.model = self.model.to(device)
        return self

class UNetAdapter(BaseAdapter):
    
    def _build_model(self) -> nn.Module:
        from .backbones.unet import UNet
        encoder_weights = self.config.get('encoder_weights', 'imagenet')
        if self.config.get('use_pretrained', True) is False or encoder_weights is None:
            encoder_weights = None
        
        return UNet(
            in_channels=self.config.get('in_channels', 3),
            num_classes=self.config['num_classes'],
            use_pretrained=self.config.get('use_pretrained', True),
            encoder_name=self.config.get('encoder_name', 'resnet34'),
            encoder_weights=encoder_weights
        )
    
    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        return image
    
    def get_default_config(self) -> Dict[str, Any]:
        return {'in_channels': 3, 'batch_size': 16}

class TransUNetAdapter(BaseAdapter):
    
    def _build_model(self) -> nn.Module:
        from .backbones.transunet import TransUNet
        return TransUNet(
            img_size=self.config.get('img_size', 224),
            num_classes=self.config['num_classes']
        )
    
    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        target_size = self.config.get('img_size', 224)
        if image.shape[2:] != (target_size, target_size):
            image = nn.functional.interpolate(image, size=(target_size, target_size))
        return image
    
    def get_default_config(self) -> Dict[str, Any]:
        return {'img_size': 224, 'batch_size': 4}


class SegNetAdapter(BaseAdapter): 
    def _build_model(self) -> nn.Module:
        from .backbones.segnet import SegNet
        return SegNet(
            in_channels=self.config.get('in_channels', 3),
            num_classes=self.config['num_classes']
        )
    
    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        return image
    
    def get_default_config(self) -> Dict[str, Any]:
        return {'in_channels': 3, 'batch_size': 8}


class AttentionUNetAdapter(BaseAdapter):
    
    def _build_model(self) -> nn.Module:
        from .backbones.attention_unet import AttentionUNet
        encoder_weights = self.config.get('encoder_weights', 'imagenet')
        if self.config.get('use_pretrained', True) is False or encoder_weights is None:
            encoder_weights = None
        
        return AttentionUNet(
            in_channels=self.config.get('in_channels', 3),
            num_classes=self.config['num_classes'],
            encoder_weights=encoder_weights
        )
    
    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        return image
    
    def get_default_config(self) -> Dict[str, Any]:
        return {'in_channels': 3, 'batch_size': 16}


class UNetPlusPlusAdapter(BaseAdapter):
    
    def _build_model(self) -> nn.Module:
        from .backbones.unetplusplus import UNetPlusPlus
        return UNetPlusPlus(
            in_channels=self.config.get('in_channels', 3),
            num_classes=self.config['num_classes']
        )
    
    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        return image
    
    def get_default_config(self) -> Dict[str, Any]:
        return {'in_channels': 3, 'batch_size': 16}


class UNETRAdapter(BaseAdapter):
    
    def _build_model(self) -> nn.Module:
        from .backbones.unetr import UNETR
        return UNETR(
            img_size=self.config.get('img_size', 224),
            in_channels=self.config.get('in_channels', 3),
            num_classes=self.config['num_classes']
        )
    
    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        return image
    
    def get_default_config(self) -> Dict[str, Any]:
        return {'img_size': 224, 'in_channels': 3, 'batch_size': 4}


class DeepLabV3Adapter(BaseAdapter):
    
    def _build_model(self) -> nn.Module:
        from .backbones.deeplabv3 import DeepLabV3Plus
        return DeepLabV3Plus(
            in_channels=self.config.get('in_channels', 3),
            num_classes=self.config['num_classes']
        )
    
    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        return image
    
    def get_default_config(self) -> Dict[str, Any]:
        return {'in_channels': 3, 'batch_size': 16}


class FCNAdapter(BaseAdapter):
    
    def _build_model(self) -> nn.Module:
        from .backbones.fcn import FCN
        print(f"🔧 [FCN] Building model with in_channels={self.config.get('in_channels', 3)}, num_classes={self.config['num_classes']}")
        model = FCN(
            in_channels=self.config.get('in_channels', 3),
            num_classes=self.config['num_classes']
        )
        print(f"✓ [FCN] Model built successfully")
        return model
    
    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        original_shape = image.shape
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
            print(f"📊 [FCN] Preprocessed: {original_shape} -> {image.shape}")
        return image
    
    def get_default_config(self) -> Dict[str, Any]:
        return {'in_channels': 3, 'batch_size': 16}

class PSPNetAdapter(BaseAdapter):
    
    def _build_model(self) -> nn.Module:
        from .backbones.pspnet import PSPNet
        return PSPNet(
            in_channels=self.config.get('in_channels', 3),
            num_classes=self.config['num_classes']
        )
    
    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        return image
    
    def get_default_config(self) -> Dict[str, Any]:
        return {'in_channels': 3, 'batch_size': 16}


class LinkNetAdapter(BaseAdapter):
    
    def _build_model(self) -> nn.Module:
        from .backbones.linknet import LinkNet
        return LinkNet(
            in_channels=self.config.get('in_channels', 3),
            num_classes=self.config['num_classes']
        )
    
    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        return image
    
    def get_default_config(self) -> Dict[str, Any]:
        return {'in_channels': 3, 'batch_size': 16}


class PANAdapter(BaseAdapter):
    
    def _build_model(self) -> nn.Module:
        from .backbones.pan import PAN
        return PAN(
            in_channels=self.config.get('in_channels', 3),
            num_classes=self.config['num_classes']
        )
    
    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        return image
    
    def get_default_config(self) -> Dict[str, Any]:
        return {'in_channels': 3, 'batch_size': 16}


class MANetAdapter(BaseAdapter):
    
    def _build_model(self) -> nn.Module:
        from .backbones.manet import MANet
        return MANet(
            in_channels=self.config.get('in_channels', 3),
            num_classes=self.config['num_classes']
        )
    
    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        return image
    
    def get_default_config(self) -> Dict[str, Any]:
        return {'in_channels': 3, 'batch_size': 16}


class HRNetAdapter(BaseAdapter):
    
    def _build_model(self) -> nn.Module:
        from .backbones.hrnet import HRNet
        return HRNet(
            in_channels=self.config.get('in_channels', 3),
            num_classes=self.config['num_classes']
        )
    
    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        return image
    
    def get_default_config(self) -> Dict[str, Any]:
        return {'in_channels': 3, 'batch_size': 16}


class OCRNetAdapter(BaseAdapter):
    
    def _build_model(self) -> nn.Module:
        from .backbones.ocrnet import OCRNet
        return OCRNet(
            in_channels=self.config.get('in_channels', 3),
            num_classes=self.config['num_classes']
        )
    
    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        return image
    
    def get_default_config(self) -> Dict[str, Any]:
        return {'in_channels': 3, 'batch_size': 16}


class DANetAdapter(BaseAdapter):
    
    def _build_model(self) -> nn.Module:
        from .backbones.danet import DANet
        return DANet(
            in_channels=self.config.get('in_channels', 3),
            num_classes=self.config['num_classes']
        )
    
    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        return image
    
    def get_default_config(self) -> Dict[str, Any]:
        return {'in_channels': 3, 'batch_size': 16}


class SegFormerAdapter(BaseAdapter):
    
    def _build_model(self) -> nn.Module:
        from .backbones.segformer import SegFormer
        return SegFormer(
            in_channels=self.config.get('in_channels', 3),
            num_classes=self.config['num_classes']
        )
    
    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        return image
    
    def get_default_config(self) -> Dict[str, Any]:
        return {'in_channels': 3, 'batch_size': 8}


class MedTAdapter(BaseAdapter):
    
    def _build_model(self) -> nn.Module:
        from .backbones.medt import MedT
        return MedT(
            img_size=self.config.get('img_size', 224),
            num_classes=self.config['num_classes']
        )
    
    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        return image
    
    def get_default_config(self) -> Dict[str, Any]:
        return {'img_size': 224, 'batch_size': 4}



class AdapterFactory:
    
    _adapters = {}
    
    @classmethod
    def create(cls, model_name: str, config: Dict[str, Any] = None) -> BaseAdapter:
        if model_name not in cls._adapters:
            available = ', '.join(sorted(cls._adapters.keys()))
            raise ValueError(f"Model '{model_name}' not supported. Available: {available}")
        
        adapter_cls = cls._adapters[model_name]
        temp_config = {'num_classes': config.get('num_classes', 7) if config else 7}
        default_config = adapter_cls(temp_config).get_default_config()
        final_config = {**default_config, **(config or {})}
        
        return adapter_cls(final_config)
    
    @classmethod
    def register(cls, name: str, adapter_cls: type):
        if not issubclass(adapter_cls, BaseAdapter):
            raise TypeError(f"{adapter_cls} must inherit from BaseAdapter")
        cls._adapters[name] = adapter_cls
        print(f"✓ Registered: {name}")
    
    @classmethod
    def list_models(cls):
        return sorted(cls._adapters.keys())


AdapterFactory.register('unet', UNetAdapter)
AdapterFactory.register('transunet', TransUNetAdapter)
AdapterFactory.register('segnet', SegNetAdapter)
AdapterFactory.register('attention_unet', AttentionUNetAdapter)
AdapterFactory.register('unetplusplus', UNetPlusPlusAdapter)
AdapterFactory.register('unetr', UNETRAdapter)
AdapterFactory.register('deeplabv3plus', DeepLabV3Adapter)
AdapterFactory.register('fcn', FCNAdapter)
AdapterFactory.register('pspnet', PSPNetAdapter)
AdapterFactory.register('linknet', LinkNetAdapter)
AdapterFactory.register('pan', PANAdapter)
AdapterFactory.register('manet', MANetAdapter)
AdapterFactory.register('hrnet', HRNetAdapter)
AdapterFactory.register('ocrnet', OCRNetAdapter)
AdapterFactory.register('danet', DANetAdapter)
AdapterFactory.register('segformer', SegFormerAdapter)
AdapterFactory.register('medt', MedTAdapter)
