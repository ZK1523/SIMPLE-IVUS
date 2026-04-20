from dataclasses import dataclass, field
from typing import List, Tuple
import os
from pathlib import Path
@dataclass
class Config:
    BASE_DIR = Path(__file__).resolve().parent.parent
    data_root = os.getenv('DATA_ROOT', str(BASE_DIR / 'data'))
    use_multilabel = os.getenv('USE_MULTILABEL', 'true').lower() == 'true'  
    @property
    def processed_dir_name(self):
        return 'processed' if self.use_multilabel else 'processed_correct'
    
    @property
    def image_dir(self):
        return f"{self.data_root}/{self.processed_dir_name}/images"
    
    @property
    def mask_dir(self):
        if self.use_multilabel:
            return f"{self.data_root}/{self.processed_dir_name}/masks_multilabel"  
        else:
            return f"{self.data_root}/{self.processed_dir_name}/masks"
    
    json_dir = os.getenv('JSON_DIR', str(BASE_DIR / 'data' / 'json'))
    
    @property
    def processed_dir(self):
        return f"{self.data_root}/{self.processed_dir_name}"
    
    @property
    def splits_dir(self):
        return f"{self.data_root}/splits"
    

    @property
    def num_classes(self):
        return 7 if self.use_multilabel else 8  
    
    class_names = [
        'background',  
        '1',         
        '2',          
        '3',         
        '4',         
        '5',         
        '6',        
        '7'           
    ]
    
    
    anatomy_constraints = {
        '11': True,
        '22': True,
        '33': True,
        '44': True
    }
    
    draw_order = ['1', '2', '3', '4', '5', '6', '7']
    
    @property
    def rare_classes(self):
        return [4, 5, 6] if self.use_multilabel else [5, 6, 7]
    
    @property
    def semi_rare_classes(self):
        return [3] if self.use_multilabel else [4]
    
    @property
    def common_classes(self):
        return [0, 1, 2] if self.use_multilabel else [1, 2, 3]
    
    image_size = (512, 512)
    image_channels = 1  

    seed = 42
    device = 'cuda'
    num_workers = 4
    batch_size = 8
    num_epochs = 100
    
    learning_rate = 1e-4
    lr_scheduler = 'cosine'

    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    
    use_weighted_sampling = True
    oversample_rare_classes = True
    use_focal_loss = False  
    use_dice_loss = True
    
    augmentation_prob = 0.8
    
    loss_weights = {
        'segmentation': 1.0,
        'classification': 0.3,
        'boundary': 0.2
    }
    
    validate_anatomy = True
    save_debug_images = True
    
    output_dir = os.getenv('OUTPUT_DIR', str(BASE_DIR / 'outputs'))
    checkpoint_dir = f"{output_dir}/checkpoints"
    log_dir = f"{output_dir}/logs"
    debug_dir = f"{output_dir}/debug"

config = Config()

def get_class_id(class_name: str) -> int:
    try:
        idx = config.class_names.index(class_name)
        if config.use_multilabel and idx > 0:
            return idx - 1  
        return idx
    except ValueError:
        raise ValueError(f"Unknown class: {class_name}")

def get_class_name(class_id: int) -> str:
    if config.use_multilabel:
        class_id += 1  
    
    if 0 <= class_id < len(config.class_names):
        return config.class_names[class_id]
    else:
        raise ValueError(f"Invalid class ID: {class_id}")

def print_config():

    
    if config.use_multilabel:
        for i in range(7):
            name = config.class_names[i + 1]
            weight = config.class_weights[i]
            rarity = ""
            if i in config.rare_classes:
                rarity = " [极稀有]"
            elif i in config.semi_rare_classes:
                rarity = " [稀有]"
            elif i in config.common_classes:
                rarity = " [常见]"
            print(f"   {i}: {name:<15} (weight={weight:.1f}){rarity}")
    else:
        for i, name in enumerate(config.class_names):
            weight = config.class_weights[i]
            rarity = ""
            if i in config.rare_classes:
                rarity = " [极稀有]"
            elif i in config.semi_rare_classes:
                rarity = " [稀有]"
            elif i in config.common_classes:
                rarity = " [常见]"
            print(f"   {i}: {name:<15} (weight={weight:.1f}){rarity}")
    
    for key, value in config.anatomy_constraints.items():
        print(f"   {key}: {value}")
    


if __name__ == '__main__':
    print_config()

