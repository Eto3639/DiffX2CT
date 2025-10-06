# ファイル名: utils.py

import random
import numpy as np
import torch
import yaml

def load_config(config_path="/workspace/config.yml"):
    """YAML設定ファイルを読み込む"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)