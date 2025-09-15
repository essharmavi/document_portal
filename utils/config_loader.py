import yaml
import os

def load_config() :
    """
    Load configuration from a YAML file.
    
    Returns:
        dict: Configuration settings.
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))  # goes up to project root
    config_path = os.path.join(base_dir, "config", "config.yaml")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    return config