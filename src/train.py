import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import argparse
from tqdm import tqdm

from models.base_model import BaseModel
from utils.utils import load_config, setup_device, save_checkpoint

def train(model, train_loader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0

    with tqdm(train_loader, desc='Training') as pbar:
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(train_loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml', help='配置文件路径')
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 设置设备
    device = setup_device()

    # 创建模型
    model = BaseModel(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        num_classes=config['model']['num_classes']
    ).to(device)

    # 设置优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    # TODO: 添加数据加载器
    # train_loader = DataLoader(...)

    # 训练循环
    for epoch in range(config['training']['epochs']):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        print(f'Epoch {epoch+1}/{config["training"]["epochs"]}, Loss: {train_loss:.4f}')

        # 保存检查点
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            },
            is_best=False,  # TODO: 根据验证集性能确定是否是最佳模型
            checkpoint_dir=config['logging']['save_dir']
        )

if __name__ == '__main__':
    main()
