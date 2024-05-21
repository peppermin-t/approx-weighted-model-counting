import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal

# 定义一个模型类，继承自 nn.Module
class GaussianModel(nn.Module):
    def __init__(self, dim):
        super(GaussianModel, self).__init__()
        # 初始化均值向量 mu 和协方差矩阵的对角元素 log_std
        self.mu = nn.Parameter(torch.randn(dim))
        self.log_std = nn.Parameter(torch.randn(dim))

    def forward(self, y):
        std = torch.exp(self.log_std)
        cov_matrix = torch.diag(std**2)  # 仅使用对角协方差矩阵
        dist = MultivariateNormal(self.mu, cov_matrix)
        log_prob = dist.log_prob(y)
        return log_prob

# 假设我们有一个向量 y
torch.manual_seed(0)
y = torch.randn(100, 2)  # 100 个 2 维样本

# 实例化模型
model = GaussianModel(dim=2)

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练循环
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()

    # 计算负对数似然
    log_prob = model(y)
    nll = -log_prob.sum()

    # 反向传播
    nll.backward()

    # 更新参数
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], NLL: {nll.item():.4f}')

# 打印最终参数
print(f'Estimated mu: {model.mu.data}')
print(f'Estimated log_std: {model.log_std.data}')
print(f'Estimated std: {torch.exp(model.log_std).data}')
