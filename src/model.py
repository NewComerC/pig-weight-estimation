import torch
import torch.nn as nn
from torchvision import models

class PigWeightEstimator(nn.Module):
    """
    基于 ResNet-18 的生猪体重估计模型。
    使用预训练的 ResNet-18 作为特征提取器，并替换其顶部的全连接层为回归头。
    """
    def __init__(self, num_features=512):
        super(PigWeightEstimator, self).__init__()
        # 1. 加载预训练的 ResNet-18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # 2. 冻结所有参数，仅训练回归头 (可选，但推荐在迁移学习中进行)
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
            
        # 3. 替换最后的全连接层为回归头
        # ResNet-18 的最后一个全连接层输入特征数是 512
        num_ftrs = self.backbone.fc.in_features
        
        # 移除原有的全连接层
        self.backbone.fc = nn.Identity()
        
        # 定义新的回归头
        self.regression_head = nn.Sequential(
            nn.Linear(num_ftrs, num_features),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(num_features, 1) # 输出一个连续的体重值
        )

    def forward(self, x):
        # 特征提取
        x = self.backbone(x)
        # 体重预测
        x = self.regression_head(x)
        return x.squeeze(1) # 确保输出是 (batch_size,) 形状
        
if __name__ == '__main__':
    # 简单的模型测试
    model = PigWeightEstimator()
    print(f"Model architecture: {model}")
    
    # 模拟输入数据 (Batch Size=4, Channels=3, Height=224, Width=224)
    dummy_input = torch.randn(4, 3, 224, 224)
    
    # 前向传播
    output = model(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Predicted weights (first 4): {output.detach().numpy()}")
    
    # 检查是否可以计算损失 (回归任务使用 MSE Loss)
    dummy_target = torch.randn(4) * 50 + 50 # 模拟 50-100kg 的体重
    loss_fn = nn.MSELoss()
    loss = loss_fn(output, dummy_target)
    print(f"Dummy MSE Loss: {loss.item():.4f}")
