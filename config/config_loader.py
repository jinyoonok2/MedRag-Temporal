import os
import yaml

random_seed = 42

class ConfigObject:
    def __init__(self, config_dict):
        for k, v in config_dict.items():
            setattr(self, k, ConfigObject(v) if isinstance(v, dict) else v)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key):
        return hasattr(self, key)

    def __repr__(self):
        return str(self.__dict__)


def load_config(yaml_filename):
    """
    Loads a YAML config and returns a ConfigObject.
    Automatically locates the file relative to this script or the working directory.
    """
    # Determine directory of this file or fallback to current working dir
    if "__file__" in globals():
        config_dir = os.path.dirname(__file__)
    else:
        config_dir = os.getcwd()

    config_path = os.path.join(config_dir, yaml_filename)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    return ConfigObject(config_dict)
