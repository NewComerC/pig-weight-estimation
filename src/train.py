import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
import time

# 导入自定义模块
from model import PigWeightEstimator
from dataset import PigWeightDataset

# --- 配置参数 ---
class Config:
    # 训练参数
    EPOCHS = 5          # 调试阶段使用较少的 Epoch
    BATCH_SIZE = 4      # 调试阶段使用较小的 Batch Size
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # 数据集参数
    DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    IS_DUMMY_DATA = True # 调试阶段使用虚拟数据
    DUMMY_SAMPLES = 100  # 虚拟数据集大小
    
    # 模型参数
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'pig_weight_estimator.pth')
    
    # 设备
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 图像预处理
    TRANSFORMS = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def train_model(config):
    print(f"--- 训练配置 ---")
    print(f"设备: {config.DEVICE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"使用虚拟数据: {config.IS_DUMMY_DATA}")
    print("----------------")

    # 1. 数据加载
    # 假设 80% 训练，20% 验证
    if config.IS_DUMMY_DATA:
        train_samples = int(config.DUMMY_SAMPLES * 0.8)
        val_samples = config.DUMMY_SAMPLES - train_samples
        
        train_dataset = PigWeightDataset(config.DATA_ROOT, config.TRANSFORMS, is_dummy=True, num_samples=train_samples)
        val_dataset = PigWeightDataset(config.DATA_ROOT, config.TRANSFORMS, is_dummy=True, num_samples=val_samples)
    else:
        # TODO: 真实数据加载逻辑
        full_dataset = PigWeightDataset(config.DATA_ROOT, config.TRANSFORMS, is_dummy=False)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    # 2. 模型、损失函数和优化器
    model = PigWeightEstimator().to(config.DEVICE)
    # 回归任务使用均方误差 (MSE) 作为损失函数
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    # 3. 训练循环
    best_val_loss = float('inf')
    
    start_time = time.time()
    
    for epoch in range(config.EPOCHS):
        # 训练阶段
        model.train()
        running_loss = 0.0
        
        # 使用 tqdm 显示进度条
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]", unit="batch")
        
        for images, weights in train_bar:
            images = images.to(config.DEVICE)
            weights = weights.to(config.DEVICE)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, weights)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            train_bar.set_postfix(loss=loss.item())

        epoch_train_loss = running_loss / len(train_dataset)

        # 验证阶段
        model.eval()
        running_loss = 0.0
        
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Val]", unit="batch")
        
        with torch.no_grad():
            for images, weights in val_bar:
                images = images.to(config.DEVICE)
                weights = weights.to(config.DEVICE)

                outputs = model(images)
                loss = criterion(outputs, weights)

                running_loss += loss.item() * images.size(0)
                val_bar.set_postfix(loss=loss.item())

        epoch_val_loss = running_loss / len(val_dataset)
        
        # 评估指标：均方根误差 (RMSE)
        epoch_val_rmse = torch.sqrt(torch.tensor(epoch_val_loss)).item()

        print(f"Epoch {epoch+1} finished. Train Loss: {epoch_train_loss:.4f}, Val Loss (MSE): {epoch_val_loss:.4f}, Val RMSE: {epoch_val_rmse:.2f} kg")

        # 保存最佳模型
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), config.MODEL_PATH)
            print(f"Model saved to {config.MODEL_PATH} with Val Loss: {best_val_loss:.4f}")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n训练完成。总耗时: {total_time:.2f} 秒")
    print(f"最佳验证损失 (MSE): {best_val_loss:.4f}")
    print(f"最佳验证 RMSE: {torch.sqrt(torch.tensor(best_val_loss)).item():.2f} kg")
    
    return total_time

if __name__ == '__main__':
    # 确保在 CPU 上运行，即使有 CUDA
    if Config.DEVICE.type == 'cuda':
        print("警告: 发现 CUDA 设备，但为调试目的，将强制使用 CPU。")
        Config.DEVICE = torch.device("cpu")
        
    # 运行训练
    cpu_time = train_model(Config)
    
    # 打印 CPU 调试时间
    print(f"\nCPU 调试运行时间: {cpu_time:.2f} 秒")
    
    # 简单的模型加载和测试
    print("\n--- 模型加载和测试 ---")
    try:
        loaded_model = PigWeightEstimator().to(Config.DEVICE)
        loaded_model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
        loaded_model.eval()
        
        # 随机取一个验证集样本进行预测
        val_dataset = PigWeightDataset(Config.DATA_ROOT, Config.TRANSFORMS, is_dummy=True, num_samples=int(Config.DUMMY_SAMPLES * 0.2))
        image, true_weight = val_dataset[0]
        
        # 预测
        with torch.no_grad():
            predicted_weight = loaded_model(image.unsqueeze(0).to(Config.DEVICE))
            
        print(f"真实体重: {true_weight.item():.2f} kg")
        print(f"预测体重: {predicted_weight.item():.2f} kg")
        print(f"误差: {abs(predicted_weight.item() - true_weight.item()):.2f} kg")
        
    except FileNotFoundError:
        print("模型文件未找到，跳过加载测试。")
    except Exception as e:
        print(f"模型加载或测试失败: {e}")
