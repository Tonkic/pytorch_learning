# PyTorch学习项目

这是一个基于PyTorch的深度学习项目模板。

## 项目结构

```
pytorch_learning/
│
├── src/                    # 源代码目录
│   ├── data/              # 数据处理相关代码
│   ├── models/            # 模型定义
│   ├── utils/             # 工具函数
│   └── train.py           # 训练脚本
│
├── data/                  # 数据集存储目录
│   ├── raw/              # 原始数据
│   └── processed/        # 处理后的数据
│
├── models/               # 保存训练好的模型
│   └── checkpoints/     # 模型检查点
│
├── configs/             # 配置文件目录
│   └── config.yaml     # 配置文件
│
├── requirements.txt    # 项目依赖
└── README.md          # 项目说明
```

## 环境配置

1. 创建虚拟环境（推荐使用conda）：
```bash
conda create -n pytorch_env python=3.11
conda activate pytorch_env
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用说明

1. 准备数据：
   - 将数据集放入 `data/raw` 目录
   - 运行数据预处理脚本

2. 训练模型：
```bash
python src/train.py --config configs/config.yaml
```

3. 评估模型：
```bash
python src/evaluate.py --model-path models/checkpoints/latest.pth
```

