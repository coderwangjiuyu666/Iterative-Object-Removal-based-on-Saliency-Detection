# 基于显著性检测的迭代物体移除

Iterative Object Removal based on Saliency Detection

本项目是课程《机器学习》的结课大作业，完成时间为2025.12

## 📖 项目简介

本项目是一个端到端的图像显著物体自动检测与移除系统，无需手动标注和固定物体类别限制，通过**超像素分割+MLP深度学习**实现像素级显著性检测，结合U2Net、SAM分割模型与Lama-cleaner图像修复技术，实现迭代式的显著物体自动移除。系统可广泛应用于摄影后期处理、图像内容编辑、冗余物体消除等场景，完整覆盖数据预处理、模型训练、推理预测、可视化调试全流程。

## ✨ 核心特性

1. **轻量化显著性检测方案**：基于SLIC超像素分割降低计算复杂度，结合VGG16提取多维度特征，通过3层MLP实现超像素级前景/背景分类，无需大规模标注数据
2. **迭代式物体移除逻辑**：每次仅移除当前图像中最显著的物体，修复后重新进行显著性检测，循环执行，避免多物体重叠导致的移除不彻底、修复失真问题
3. **多模型协同优化**：U2Net生成初始物体掩码 → SAM模型精细化掩码边界 → Lama-cleaner实现高保真背景修复，全流程自动化
4. **完整的工程化能力**：
   1. 预计算缓存机制，避免训练时重复特征提取，大幅提升训练效率
   2. 内置丰富的可视化调试工具，可输出每一步迭代的显著性图、掩码、修复结果
   3. 支持单张图像推理与批量图像处理
   4. 训练集/验证集自动划分，Focal Loss解决前景背景样本不均衡问题
5. **完善的异常处理**：内置伪目标过滤、边界检查、显著性对比度校验，避免无效迭代与错误修复

## 🏗️ 系统整体架构

系统分为两大核心模块，模块间无缝衔接，形成完整的自动化处理闭环：

```Plain
原始图像 → 【显著性检测模块】 → 显著性热力图
    ↓
【迭代物体移除模块】
    ↓
U2Net初始掩码生成 → 伪目标过滤 → SAM掩码精细化 → 形态学优化 → Lama图像修复
    ↓
修复后图像重新输入显著性检测模块，循环执行
    ↓
满足停止条件 → 输出最终无物体的干净图像
```

### 显著性检测模块

1. **超像素分割**：SLIC算法将图像划分为300个语义连贯的超像素区域，提取掩码、边界框、邻域等信息

2. **多维度特征提取**：基于预训练VGG16，提取单超像素特征、邻域合并特征、全局图像特征，拼接为12288维特征向量

3. **MLP分类预测**：3层全连接网络输出每个超像素的前景概率，生成像素级显著性热力图

4. **后处理优化**：高斯模糊平滑、形态学去噪、归一化，输出最终显著性图（如下所示）

   <img src="/docImage/saliency_1.jpg" style="zoom:67%;" />

### 迭代物体移除模块

1. **掩码生成与过滤**：U2Net生成初始物体掩码，通过显著性一致性、边界重叠、对比度校验过滤伪目标

2. **掩码精细化**：SAM模型基于初始掩码与边界框，优化物体分割边界，提升掩码精度

3. **掩码后处理**：形态学膨胀、空洞填充、边缘平滑，适配图像修复的输入要求

4. **图像修复**：Lama-cleaner基于掩码擦除显著物体，实现背景的无缝填充

5. **迭代控制**：支持最大迭代次数、显著性阈值、重叠率等多维度停止条件控制

   ![](/docImage/Snipaste_2026-03-15_15-29-50.png)

## 📋 环境要求

- **Python版本**：Python 3.9

- **CUDA支持**：推荐CUDA 11.8（CPU可运行，GPU大幅提升处理速度）

- **操作系统**：Windows/Linux/macOS

- **核心依赖**：详见`requirements.txt`，核心库如下：

  - ```Plain
    torch>=2.5.1
    torchvision>=0.20.1
    opencv-python>=4.12.0
    scikit-image>=0.24.0
    numpy>=2.0.1
    lama-cleaner==1.2.5
    segment-anything==1.0
    pillow>=11.1.0
    matplotlib>=3.9.0
    tqdm>=4.67.0
    ```

## 🛠️ 安装步骤

### 1. 克隆仓库

```Bash
git clone https://github.com/coderwangjiuyu666/Iterative-Object-Removal-based-on-Saliency-Detection.git
cd Saliency-Based-Iterative-Object-Removal
```

### 2. 创建并激活Conda环境

```Bash
conda create -n <your_env_name> python=3.9
conda activate <your_env_name>
```

### 3. 安装依赖

```Bash
pip install -r requirements.txt
```

### 4. 下载预训练模型

1. **U2Net预训练模型**：下载[u2net.pth](https://github.com/xuebinqin/U-2-Net/releases/download/v1.0/u2net.pth)，放入`./checkpoints/`目录
2. **SAM预训练模型**：下载[sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)，放入`./checkpoints/`目录
3. **预训练显著性检测模型**：可使用项目训练好的`saliency_model4.pth`，放入`./pths/`目录；也可自行训练生成（详见`SaliencyDetection.py`）

### 5. 准备数据集（训练用）

本项目使用DUTS-TR数据集进行训练，下载地址：[DUTS Dataset](http://saliencydetection.net/duts/)

- 下载后解压，将图像放入`./DUTS-TR/DUTS-TR-Image/`
- 掩码标签放入`./DUTS-TR/DUTS-TR-Mask/`

## 🚀 快速开始

### 一、数据预计算（仅首次运行需执行）

预计算超像素特征与标签，缓存为npz文件，避免训练时重复计算，大幅提升训练效率。

```Bash
python pretrain.py
```

执行后会自动生成`./DUTS-TR/precomputed_data/`目录，存储预计算的特征文件，同时输出数据集的前景/背景标签分布统计。

### 二、显著性检测模型训练

```Bash
python SaliencyDetection.py
```

- 自动加载预计算的缓存数据，按9:1划分训练集/验证集
- 训练完成后，最优模型与最终模型会保存至`./pths/`目录
- 训练日志与loss/F1曲线会自动保存至`./training_logs/`目录
- 同时会对`./test_images/`目录下的图像进行批量推理，生成显著性热力图至`./output_results/`

### 三、迭代物体移除

#### 1. 单张/批量图像处理

将需要处理的图像放入`./test_images_for_remove/`目录，执行：

```Bash
python Remove.py
```

- 自动批量处理目录内的所有图像
- 每轮迭代的中间结果保存至`./iterative_removal_results/`
- 最终修复结果保存至`./batch_output_u2net_sam/`
- 同时生成每一张图像的移除历史记录与调试可视化文件

#### 2. 核心参数调整

在`Remove.py`的主函数中，可调整以下核心参数：

- `max_objects`：最大迭代移除的物体数量，默认6
- `saliency_avg_stop_thresh`：平均显著性停止阈值，默认20.0，低于该值停止迭代
- 超像素分割参数：`num_superpixels`超像素数量、`compactness`紧凑度
- 掩码膨胀参数：`er_rate`形态学膨胀核大小

## 📁 项目结构

```Plain
├── pretrain.py              # 数据预计算脚本，预提取超像素特征与标签并缓存
├── SaliencyDetection.py     # 显著性检测核心代码，包含模型定义、训练、推理全流程
├── Remove.py                # 迭代物体移除主脚本，集成U2Net+SAM+Lama全流程
├── requirements.txt         # 项目依赖清单
├── checkpoints/             # 预训练模型存放目录（U2Net、SAM）
├── pths/                    # 训练好的显著性检测MLP模型存放目录
├── DUTS-TR/                 # 训练数据集目录
│   ├── DUTS-TR-Image/       # 训练集原图
│   ├── DUTS-TR-Mask/        # 训练集掩码标签
│   └── precomputed_data/    # 预计算特征缓存目录
├── test_images/             # 显著性检测图像输入目录
├── test_images_for_remove/  # 迭代移除图像输入目录
├── output_results/          # 显著性检测推理结果输出目录
├── batch_output_u2net_sam/  # 批量物体移除最终结果输出目录
├── iterative_removal_results/ # 迭代过程中间结果目录
├── debug_masks/             # 掩码调试可视化输出目录
├── u2net_visualization/     # U2Net检测结果可视化目录
└── training_logs/           # 模型训练日志与曲线目录
```

## 📚 参考与致谢

本项目参考了以下优秀的开源项目与研究成果，在此向相关作者表示感谢：

- [U-2-Net](https://github.com/xuebinqin/U-2-Net)：显著性检测与物体分割模型
- [Segment Anything](https://github.com/facebookresearch/segment-anything)：SAM通用分割模型
- [lama-cleaner](https://github.com/guocuixia/lama-cleaner)：图像修复与物体擦除工具
- 论文《Fine-grained perception in panoramic scenes: a novel task, dataset, and method for object importance ranking》
- 论文《Saliency detection: a spectral residual approach》

