import os
import yaml
import torch
import shutil
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def setup_device(use_cuda: bool = True):
    """设置计算设备"""
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device

def save_checkpoint(state: Dict[str, Any], is_best: bool, checkpoint_dir: str):
    """保存模型检查点"""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pth')
    torch.save(state, checkpoint_path)

    if is_best:
        best_path = os.path.join(checkpoint_dir, 'checkpoint_best.pth')
        shutil.copy(checkpoint_path, best_path)
