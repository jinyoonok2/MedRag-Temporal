# utils/openclip_utils.py

import open_clip
from config import load_config

def get_pretrained_model(cfg):
    """
    Loads and returns the OpenCLIP model architecture based on the configured pretrained name.
    """
    model, _ = open_clip.create_model_from_pretrained(cfg.pretrained)
    return model

def get_preprocess_transform(cfg):
    """
    Returns the preprocessing transform associated with the OpenCLIP model.
    """
    _, preprocess = open_clip.create_model_from_pretrained(cfg.pretrained)
    return preprocess

def get_tokenizer(cfg):
    """
    Returns the tokenizer function associated with the OpenCLIP model.
    """
    return open_clip.get_tokenizer(cfg.pretrained)
