from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu

class Expert(nn.Module):
    """单个专家网络模块

    Args:
        input_dim (int): 输入维度
        hidden_dim (int): 隐藏层维度
        output_dim (int): 输出维度
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, input_dim)

        Returns:
            torch.Tensor: 输出张量，形状为 (batch_size, output_dim)
        """
        return self.net(x)

class MoE(nn.Module):
    """混合专家模型

    Args:
        input_dim (int): 输入维度
        num_experts (int): 专家数量
        top_k (int): 每个样本选择的专家数量
        expert_capacity (int): 每个专家可处理的最大样本数
        hidden_dim (int): 专家网络隐藏层维度
        output_dim (int): 输出维度
    """
    def __init__(self,
                 input_dim: int,
                 num_experts: int,
                 top_k: int,
                 expert_capacity: int,
                 hidden_dim: int,
                 output_dim: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_capacity = expert_capacity

        # 路由网络
        self.gate = nn.Linear(input_dim, num_experts)

        # 专家集合
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim)
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """前向传播

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, input_dim)

        Returns:
            Tuple[torch.Tensor, float]:
                - 输出张量，形状为 (batch_size, output_dim)
                - 辅助损失值（训练模式下为重要性损失和负载均衡损失之和，评估模式下为0）
        """
        batch_size, input_dim = x.shape
        device = x.device

        # 路由计算
        logits = self.gate(x)
        probs = torch.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, self.top_k, dim=-1)

        # 计算辅助损失
        aux_loss = self._compute_auxiliary_loss(probs, topk_indices) if self.training else 0.0

        # 专家分配和计算
        outputs = self._dispatch_to_experts(x, topk_indices, topk_probs, batch_size, device)

        return outputs, aux_loss

    def _compute_auxiliary_loss(self,
                              probs: torch.Tensor,
                              topk_indices: torch.Tensor) -> float:
        """计算辅助损失

        Args:
            probs (torch.Tensor): 路由概率
            topk_indices (torch.Tensor): top-k专家索引

        Returns:
            float: 辅助损失值
        """
        # 重要性损失（专家利用率均衡）
        importance = probs.sum(0)
        importance_loss = torch.var(importance) / (self.num_experts ** 2)

        # 负载均衡损失（样本分配均衡）
        mask = torch.zeros_like(probs, dtype=torch.bool)
        mask.scatter_(1, topk_indices, True)
        routing_probs = probs * mask
        expert_usage = mask.float().mean(0)
        routing_weights = routing_probs.mean(0)
        load_balance_loss = self.num_experts * (expert_usage * routing_weights).sum()

        return importance_loss + load_balance_loss

    def _dispatch_to_experts(self,
                           x: torch.Tensor,
                           topk_indices: torch.Tensor,
                           topk_probs: torch.Tensor,
                           batch_size: int,
                           device: torch.device) -> torch.Tensor:
        """分发样本到专家并计算输出

        Args:
            x (torch.Tensor): 输入张量
            topk_indices (torch.Tensor): top-k专家索引
            topk_probs (torch.Tensor): top-k路由概率
            batch_size (int): 批次大小
            device (torch.device): 计算设备

        Returns:
            torch.Tensor: 专家计算结果
        """
        flat_indices = topk_indices.view(-1)
        flat_probs = topk_probs.view(-1)
        sample_indices = torch.arange(batch_size, device=device)[:, None]\
                            .expand(-1, self.top_k).flatten()

        outputs = torch.zeros(batch_size, self.experts[0].net[-1].out_features,
                            device=device)

        for expert_idx in range(self.num_experts):
            expert_mask = flat_indices == expert_idx
            expert_samples = sample_indices[expert_mask]
            expert_weights = flat_probs[expert_mask]

            if len(expert_samples) > self.expert_capacity:
                expert_samples = expert_samples[:self.expert_capacity]
                expert_weights = expert_weights[:self.expert_capacity]

            if len(expert_samples) == 0:
                continue

            expert_input = x[expert_samples]
            expert_output = self.experts[expert_idx](expert_input)
            weighted_output = expert_output * expert_weights.unsqueeze(-1)

            outputs.index_add_(0, expert_samples, weighted_output)

        return outputs

# 测试示例
if __name__ == "__main__":
    input_dim = 128
    output_dim = 256
    num_experts = 8
    top_k = 2
    expert_capacity = 32
    hidden_dim = 512
    batch_size = 64

    # add
    device = torch.device("npu:4" if torch.npu.is_available() else "cpu")
    moe = MoE(input_dim, num_experts, top_k, expert_capacity, hidden_dim, output_dim).to(device)
    x = torch.randn(batch_size, input_dim).to(device)

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
        msprof_tx=False,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
        l2_cache=False,
        op_attr=False,
        data_simplification=False,
        record_op_args=False,
        gc_detect_threshold=None
    )

    with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU
            ],
            schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=1),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./moe_stand_npu_result"),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_modules=False,
            with_flops=False,
            experimental_config=experimental_config) as prof:
        # 训练模式
        for _ in range(10):
            moe.train()
            output, loss = moe(x)
            print(f"Using device: {x.device}")
            print(f"Training output shape: {output.shape}")      # torch.Size([64, 256])
            print(f"Training auxiliary loss: {loss.item():.4f}")     # 示例值，如 0.1234
            prof.step()

    print("=" * 80)

    # 推理模式
    moe.eval()
    output, _ = moe(x)
    print(f"Eval output shape: {output.shape}")     # torch.Size([64, 256])