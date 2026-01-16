import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import random

class PigWeightDataset(Dataset):
    """
    生猪体重估计数据集。
    
    在没有实际数据集文件时，此实现使用虚拟数据进行测试。
    一旦 PIGRGB-Weight 数据集下载完成，可以修改 load_data_list 方法来加载真实数据。
    """
    def __init__(self, root_dir, transform=None, is_dummy=True, num_samples=100):
        self.root_dir = root_dir
        self.transform = transform
        self.is_dummy = is_dummy
        self.num_samples = num_samples
        
        if self.is_dummy:
            # 虚拟模式：生成随机数据点
            self.data_list = self._generate_dummy_data(num_samples)
        else:
            # 真实模式：加载实际数据列表
            self.data_list = self._load_real_data(root_dir)
            
        if not self.data_list:
            raise RuntimeError("Dataset is empty. Check root_dir or dummy settings.")

    def _generate_dummy_data(self, num_samples):
        """生成虚拟的文件路径和体重标签"""
        dummy_data = []
        # 模拟体重范围 30kg 到 150kg
        weights = np.random.uniform(30.0, 150.0, num_samples)
        for i in range(num_samples):
            # 虚拟路径，仅用于测试逻辑
            dummy_path = f"dummy_path/pig_{i:04d}.png"
            dummy_data.append({'path': dummy_path, 'weight': weights[i]})
        return dummy_data

    def _load_real_data(self, root_dir):
        """
        加载 PIGRGB-Weight 真实数据列表的占位符。
        
        需要根据 PIGRGB-Weight 的实际目录结构（如 RGB_9579/fold1/）来解析文件路径和体重。
        """
        print(f"Loading real data from {root_dir}...")
        # TODO: 实现 PIGRGB-Weight 数据集的真实加载逻辑
        # 示例：遍历目录，从文件名中提取体重
        real_data = []
        # for folder in os.listdir(os.path.join(root_dir, 'RGB_9579')):
        #     for filename in os.listdir(os.path.join(root_dir, 'RGB_9579', folder)):
        #         if filename.endswith('.png'):
        #             try:
        #                 weight = float(filename.split('kg')[0])
        #                 real_data.append({'path': os.path.join(root_dir, 'RGB_9579', folder, filename), 'weight': weight})
        #             except:
        #                 continue
        return real_data

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        weight = torch.tensor(item['weight'], dtype=torch.float32)
        
        if self.is_dummy:
            # 虚拟模式：返回一个随机生成的图像张量
            # 模拟 ResNet-18 要求的 3x224x224 尺寸
            image = torch.randn(3, 224, 224)
        else:
            # 真实模式：加载图像
            image = Image.open(item['path']).convert('RGB')
            if self.transform:
                image = self.transform(image)
        
        # 体重估计是回归任务，标签是一个标量
        return image, weight

if __name__ == '__main__':
    # 简单的数据集测试
    
    # 1. 定义图像预处理
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 2. 创建虚拟数据集
    dummy_dataset = PigWeightDataset(root_dir='/home/ubuntu/pig-weight-estimation/data', 
                                     transform=data_transforms, 
                                     is_dummy=True, 
                                     num_samples=10)
    
    print(f"Dummy Dataset Size: {len(dummy_dataset)}")
    
    # 3. 测试加载单个样本
    img, weight = dummy_dataset[0]
    print(f"Sample 0 - Image Tensor Shape: {img.shape}")
    print(f"Sample 0 - Weight: {weight.item():.2f} kg")
    
    # 4. 使用 DataLoader
    from torch.utils.data import DataLoader
    dummy_dataloader = DataLoader(dummy_dataset, batch_size=4, shuffle=True)
    
    print("\nTesting DataLoader:")
    for i, (images, weights) in enumerate(dummy_dataloader):
        print(f"Batch {i+1} - Images Shape: {images.shape}, Weights Shape: {weights.shape}")
        if i >= 1: # 只测试两个批次
            break
