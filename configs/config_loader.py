import yaml
from pathlib import Path
from configs.config import config as base_config

def load_model_config(model_name):

    config_path = Path(__file__).parent / 'model_configs.yaml'
    with open(config_path) as f:
        configs = yaml.safe_load(f)
    return configs['models'].get(model_name, {})

def get_merged_config(model_name, args):           
    config = {
        'num_classes': base_config.num_classes,
        'img_size': base_config.image_size[0],
        'in_channels': base_config.image_channels,
        'class_names': base_config.class_names,
        'class_weights': base_config.class_weights,
    }
    
    model_config = load_model_config(model_name)
    config.update(model_config)

    if hasattr(args, 'batch_size') and args.batch_size:
        config['batch_size'] = args.batch_size
    
    return config
