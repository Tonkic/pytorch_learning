import torch

# 假设这是神经网络输出
def y_theta(x1, x2):
    # 这里只是一个简单的例子，真实情况是神经网络模型
    return x1**2 + torch.sin(x2)

# 定义输入，并开启梯度追踪
x1 = torch.tensor(1.2, requires_grad=True)
x2 = torch.tensor(0.7, requires_grad=True)

# 前向计算
y = y_theta(x1, x2)

# 一阶偏导
dy_dx1 = torch.autograd.grad(y, x1, create_graph=True)[0]
dy_dx2 = torch.autograd.grad(y, x2, create_graph=True)[0]

# 二阶偏导
d2y_dx1_2 = torch.autograd.grad(dy_dx1, x1)[0]
d2y_dx2_2 = torch.autograd.grad(dy_dx2, x2)[0]

print(f"y={y.item():.4f}")
print(f"d²y/dx1²={d2y_dx1_2.item():.4f}")
print(f"d²y/dx2²={d2y_dx2_2.item():.4f}")

# PDE残差
residual = d2y_dx1_2 + d2y_dx2_2 - y
print(f"PDE残差={residual.item():.4e}")
