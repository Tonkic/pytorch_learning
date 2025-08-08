import torch
import torch_npu
from src.models.moe_stand_npu import MoE

def test_moe():
    # 模型参数
    input_dim = 128
    output_dim = 256
    num_experts = 8
    top_k = 2
    expert_capacity = 32
    hidden_dim = 512
    batch_size = 64

    # 设备配置
    device = torch.device("npu:4" if torch.npu.is_available() else "cpu")
    
    # 创建模型和测试数据
    moe = MoE(input_dim, num_experts, top_k, expert_capacity, hidden_dim, output_dim).to(device)
    x = torch.randn(batch_size, input_dim).to(device)

    # NPU性能分析配置
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

    # 性能分析
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
        
        # 训练模式测试
        for _ in range(10):
            moe.train()
            output, loss = moe(x)
            print(f"Using device: {x.device}")
            print(f"Training output shape: {output.shape}")
            print(f"Training auxiliary loss: {loss.item():.4f}")
            prof.step()

    print("=" * 80)

    # 推理模式测试
    moe.eval()
    output, _ = moe(x)
    print(f"Eval output shape: {output.shape}")

if __name__ == "__main__":
    test_moe()