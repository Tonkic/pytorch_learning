<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# PyTorch项目指南

这是一个PyTorch深度学习项目。在编写代码时，请遵循以下准则：

1. 代码风格：
   - 使用清晰的变量和函数命名
   - 添加适当的类型提示
   - 为函数和类添加docstring
   - 遵循PEP 8规范

2. 项目结构：
   - 模型定义放在 `src/models` 目录
   - 数据处理相关代码放在 `src/data` 目录
   - 工具函数放在 `src/utils` 目录
   - 配置文件放在 `configs` 目录

3. 最佳实践：
   - 使用设备无关的代码（CPU/GPU兼容）
   - 实现适当的错误处理
   - 使用配置文件管理超参数
   - 记录实验过程和结果
   - 定期保存模型检查点
