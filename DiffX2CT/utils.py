# ファイル名: utils.py

import random
import numpy as np
import torch
import yaml
import os

def load_config():
    """YAML設定ファイルを読み込む"""
    config_file_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "config.yml"))
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)