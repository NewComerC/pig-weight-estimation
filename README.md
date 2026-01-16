# Pig Weight Estimation (纯视觉生猪体重评估)

本项目旨在开发一个基于计算机视觉的非接触式生猪体重评估模型。通过分析生猪的背部 RGB 图像，利用深度学习技术实现高精度的体重预测。

## 🚀 项目亮点
- **非接触式**: 减少生猪应激，提高防疫安全性。
- **纯视觉**: 仅需普通 RGB 摄像头，成本低廉。
- **端到端**: 从图像输入到体重输出的完整流水线。

## 📂 目录结构
- `src/`: 核心源代码（模型定义、训练脚本、数据加载）。
- `data/`: 数据集存放目录（已在 .gitignore 中排除）。
- `models/`: 训练好的模型权重。
- `docs/`: 项目文档和实施计划。

## 🛠️ 快速开始 (CPU 调试)
1. 克隆仓库:
   ```bash
   git clone https://github.com/NewComerC/pig-weight-estimation.git
   cd pig-weight-estimation
   ```
2. 安装依赖:
   ```bash
   pip install -r requirements.txt
   ```
3. 运行调试训练:
   ```bash
   python src/train.py
   ```

## 📅 实施计划
详细的算力预估和 3 天完成计划请参阅 [Implementation Plan](docs/implementation_plan.md)。

## 📊 数据集调研
本项目首选使用 [PIGRGB-Weight](https://github.com/maweihong/PIGRGB-Weight) 数据集进行训练。
