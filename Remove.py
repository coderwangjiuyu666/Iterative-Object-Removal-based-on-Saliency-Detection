import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import io, morphology
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tqdm import tqdm
import glob
import multiprocessing
from scipy.ndimage import binary_dilation
import copy
import requests
from io import BytesIO
import base64
from typing import Union, Tuple
import zipfile
import subprocess
import sys
import tempfile
import time

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # 根据输入尺寸自动选择更快算法
# 设置字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题


def visualize_debug_mask(image, saliency_map, mask, refined_mask, bbox,image_name=None,iteration=None):
    """
    可视化显著性图、原始mask、refined mask、bbox
    """
    x1, y1, x2, y2 = bbox

    fig, axs = plt.subplots(2, 3, figsize=(18, 12))  # 增加列数

    axs[0, 0].imshow(image)
    axs[0, 0].add_patch(
        plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=2)
    )
    axs[0, 0].set_title("Image + BBox")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(saliency_map, cmap="hot")
    axs[0, 1].set_title("Saliency Map")
    axs[0, 1].axis("off")

    # 新增：显示阈值处理后的二值图
    threshold = np.percentile(saliency_map, 70)
    binary_map = (saliency_map >= threshold).astype(np.uint8) * 255
    axs[0, 2].imshow(binary_map, cmap="gray")
    axs[0, 2].set_title(f"Binary Map (threshold={threshold:.1f})")
    axs[0, 2].axis("off")

    axs[1, 0].imshow(mask, cmap="gray")
    axs[1, 0].set_title("Original Mask")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(refined_mask, cmap="gray")
    axs[1, 1].set_title("Refined Mask for Lama")
    axs[1, 1].axis("off")

    # 新增：显示显著性图的直方图
    axs[1, 2].hist(saliency_map.flatten(), bins=50, color='blue', alpha=0.7)
    axs[1, 2].axvline(x=threshold, color='red', linestyle='--', label=f'阈值={threshold:.1f}')
    axs[1, 2].set_title("Saliency Histogram")
    axs[1, 2].set_xlabel("显著性值")
    axs[1, 2].set_ylabel("频率")
    axs[1, 2].legend()
    axs[1, 2].grid(True, alpha=0.3)

    os.makedirs("debug_masks", exist_ok=True)
    # 生成带图片名前缀的文件名
    if image_name:
        save_name = f"debug_masks/{image_name}_{iteration}.jpg"
    else:
        save_name = f"debug_masks/{iteration}.jpg"
    fig.savefig(save_name, dpi=150)
    plt.close(fig)


def visualize_removal_process(original, current, saliency_map, mask, iteration,image_name=None):
    """可视化移除过程"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].imshow(original)
    axes[0, 0].set_title(f"原始图像")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(saliency_map, cmap='hot')
    axes[0, 1].set_title(f"显著性图 - 迭代 {iteration}")
    axes[0, 1].axis('off')

    axes[1, 0].imshow(mask, cmap='gray')
    axes[1, 0].set_title(f"当前掩码 - 迭代 {iteration}")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(current)
    axes[1, 1].set_title(f"当前图像 - 迭代 {iteration}")
    axes[1, 1].axis('off')

    plt.tight_layout()
    os.makedirs("debug_masks", exist_ok=True)

    if image_name:
        save_name = f"debug_masks/{image_name}_removal_process_iter_{iteration}.jpg"
    else:
        save_name = f"debug_masks/removal_process_iter_{iteration}.jpg"

    plt.savefig(save_name, dpi=150)
    plt.close()


def visualize_u2net_results(image, mask, iteration,image_name=None):
    """可视化U2Net检测结果"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 原始图像
    axes[0].imshow(image)
    axes[0].set_title(f"原始图像 - 迭代 {iteration}")
    axes[0].axis('off')

    # 在原始图像上叠加检测结果
    image_with_overlay = image.copy()
    overlay = np.zeros_like(image)
    overlay[:, :, 1] = mask * 255  # 绿色通道表示前景
    image_with_overlay = cv2.addWeighted(image_with_overlay, 0.7, overlay, 0.3, 0)

    axes[1].imshow(image_with_overlay)
    axes[1].set_title(f"U2Net检测结果 - 迭代 {iteration}")
    axes[1].axis('off')

    plt.tight_layout()
    os.makedirs("u2net_visualization", exist_ok=True)
    # 生成带图片名前缀的文件名
    if image_name:
        save_name = f'u2net_visualization/{image_name}_iteration_{iteration}.jpg'
    else:
        save_name = f'u2net_visualization/_iteration_{iteration}.jpg'
    plt.savefig(save_name, dpi=150)
    plt.close()


def visualize_u2net_masks(image, mask, iteration,image_name=None):
    """可视化U2Net检测的掩码"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 显示原始图像
    axes[0].imshow(image)
    axes[0].set_title(f"原始图像")
    axes[0].axis('off')

    # 显示掩码
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title(f"U2Net掩码 (面积: {np.sum(mask):.0f})")
    axes[1].axis('off')

    plt.tight_layout()
    os.makedirs("u2net_visualization_masks", exist_ok=True)
    # 生成带图片名前缀的文件名
    if image_name:
        save_name = f'u2net_visualization_masks/{image_name}_iteration_{iteration}.jpg'
    else:
        save_name = f'u2net_visualization_masks/_iteration_{iteration}.jpg'
    plt.savefig(save_name, dpi=150)
    plt.close()


# --------------------------
# U2Net模型定义
# --------------------------
class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout


def _upsample_like(src, tar):
    src = F.interpolate(src, size=tar.shape[2:], mode='bilinear')
    return src


### RSU-7 ###
class RSU7(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = _upsample_like(hx6d, hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-6 ###
class RSU6(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-5 ###
class RSU5(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-4 ###
class RSU4(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-4F ###
class RSU4F(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))

        return hx1d + hxin


import torch.nn.functional as F


class U2NET(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(U2NET, self).__init__()

        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 256, 512)

        # decoder
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6, out_ch, 1)

    def forward(self, x):
        hx = x

        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        # decoder
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)
        d1 = F.sigmoid(d1)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)
        d2 = F.sigmoid(d2)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)
        d3 = F.sigmoid(d3)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)
        d4 = F.sigmoid(d4)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)
        d5 = F.sigmoid(d5)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)
        d6 = F.sigmoid(d6)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
        d0 = F.sigmoid(d0)

        return d0, d1, d2, d3, d4, d5, d6


class U2NetDetector:
    """
    U2Net检测器，用于显著性检测
    """

    def __init__(self, model_path):
        print("正在加载U2Net模型...")
        try:
            # 创建模型
            self.model = U2NET(in_ch=3, out_ch=1)

            # 加载预训练权重
            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint)
            self.model.to(device)
            self.model.eval()

            print("U2Net模型加载成功！")
        except FileNotFoundError:
            print(f"错误：找不到U2Net检查点文件: {model_path}")
            raise

    def detect(self, image):
        """
        使用U2Net进行显著性检测
        """
        print("使用U2Net进行显著性检测...")

        # 预处理图像
        original_size = image.shape[:2]
        image_resized = cv2.resize(image, (320, 320))

        # 转换为tensor (H, W, C) RGB -> (C, H, W) tensor
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float().unsqueeze(0).to(device)
        image_tensor = image_tensor / 255.0  # 归一化到[0,1]

        # 运行检测
        with torch.no_grad():
            outputs = self.model(image_tensor)
            pred = outputs[0][0, 0]  # 获取第一个输出的显著性图
            pred = torch.sigmoid(pred)  # 应用sigmoid激活函数
            pred = pred.cpu().numpy()

        # 转换回原始尺寸
        pred_resized = cv2.resize(pred, (original_size[1], original_size[0]))

        # 二值化处理
        threshold = np.percentile(pred_resized, 95)
        threshold = max(threshold, 0.5)  # 强制最小阈值=0.4（pred_resized是[0,1]范围）
        mask = (pred_resized >= threshold).astype(np.uint8)

        # 计算边界框
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            bbox = (x, y, x + w, y + h)
        else:
            # 如果没找到轮廓，返回全零掩码和无效边界框
            bbox = (0, 0, 0, 0)
            mask = np.zeros(original_size, dtype=np.uint8)

        # 计算掩码面积
        area = np.sum(mask)

        result = {
            'mask': mask,
            'bbox': bbox,
            'area': area,
            'saliency_map': pred_resized
        }


        print(f"U2Net检测完成，掩码面积: {area}")
        return result


# --------------------------
# 2. SAM集成（真实模型）
# --------------------------
class SAMOptimizer:
    """
    真实SAM（Segment Anything Model）优化器，用于手动优化掩码
    """

    def __init__(self, model_type="vit_h", checkpoint_path=None):
        print("初始化真实SAM模型...")

        try:
            from segment_anything import sam_model_registry, SamPredictor

            # 设置默认checkpoint路径
            if checkpoint_path is None:
                checkpoint_path = "./checkpoints/sam_vit_h_4b8939.pth"

            # 加载SAM模型
            self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            self.sam.to(device=device)
            self.predictor = SamPredictor(self.sam)

            print("SAM模型加载成功！")
        except ImportError:
            print("错误：无法导入segment-anything。请先安装SAM库。")
            raise
        except FileNotFoundError:
            print(f"错误：找不到SAM检查点文件。请下载对应的SAM模型文件。")
            raise

    def optimize_mask(self, image, mask, bbox):
        """
        使用真实SAM优化掩码精度
        """
        print("使用真实SAM进行掩码优化...")

        # 设置图像到SAM预测器
        self.predictor.set_image(image)

        x1, y1, x2, y2 = bbox

        # 使用边界框提示进行分割
        input_boxes = torch.tensor([[x1, y1, x2, y2]], device=self.predictor.device)
        transformed_boxes = self.predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])

        # 使用SAM进行精确分割
        masks, _, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        # 获取优化后的掩码
        optimized_mask = masks[0].cpu().numpy().astype(np.uint8)
        # 将优化后的掩码调整回原始图像尺寸
        optimized_mask_resized = cv2.resize(optimized_mask[0], (image.shape[1], image.shape[0]),
                                            interpolation=cv2.INTER_NEAREST)
        return optimized_mask_resized


# --------------------------
# 3. 超像素处理工具类
# --------------------------
class SuperpixelProcessor:
    def __init__(self, num_superpixels=300, compactness=10, sigma=1.0):
        self.num_superpixels = num_superpixels
        self.compactness = compactness
        self.sigma = sigma

    def get_superpixels(self, image):
        """获取超像素分割结果及相关信息"""
        segments = slic(
            img_as_float(image),
            n_segments=self.num_superpixels,
            compactness=self.compactness,
            sigma=self.sigma,
            start_label=0
        )
        num_segments = segments.max() + 1

        # 提取每个超像素的信息
        sp_info = []
        for i in range(num_segments):
            mask = segments == i
            y, x = np.where(mask)
            if len(y) == 0:
                continue

            # 计算边界框
            y1, y2 = y.min(), y.max()
            x1, x2 = x.min(), x.max()
            bbox = (x1, y1, x2, y2)

            # 计算中心坐标
            centroid = (np.mean(x), np.mean(y))

            # 计算邻域超像素
            neighbors = self._find_neighbors(segments, i, bbox)

            sp_info.append({
                'id': i,
                'mask': mask,
                'bbox': bbox,
                'centroid': centroid,
                'neighbors': neighbors
            })

        return segments, sp_info

    def _find_neighbors(self, segments, sp_id, bbox, scale=1.5):
        """通过扩展边界框寻找邻域超像素"""
        x1, y1, x2, y2 = bbox
        h, w = segments.shape[:2]

        # 扩展边界框
        w_expand = (x2 - x1) * (scale - 1) / 2
        h_expand = (y2 - y1) * (scale - 1) / 2
        x1_e = max(0, int(x1 - w_expand))
        y1_e = max(0, int(y1 - h_expand))
        x2_e = min(w - 1, int(x2 + w_expand))
        y2_e = min(h - 1, int(y2 + h_expand))

        # 提取扩展区域内的超像素
        region = segments[y1_e:y2_e + 1, x1_e:x2_e + 1]
        neighbors = np.unique(region)
        neighbors = neighbors[neighbors != sp_id]  # 排除自身
        return neighbors.tolist()

    def _get_neighbors_bbox(self, neighbor_ids, sp_info):
        """根据邻域超像素 id 列表和 sp_info 计算最小包围框（用于合并邻域patch）"""
        if not neighbor_ids:
            return (0, 0, 0, 0)
        bboxes = [sp_info[i]['bbox'] for i in neighbor_ids if 0 <= i < len(sp_info)]
        if not bboxes:
            return (0, 0, 0, 0)
        x1 = min(b[0] for b in bboxes)
        y1 = min(b[1] for b in bboxes)
        x2 = max(b[2] for b in bboxes)
        y2 = max(b[3] for b in bboxes)
        return (x1, y1, x2, y2)

    def get_superpixel_label(self, sp_mask, gt_mask):
        """根据与GT的交并比计算超像素标签"""
        intersection = np.logical_and(sp_mask, gt_mask).sum()
        union = sp_mask.sum()
        if union == 0:
            return 0
        ratio = intersection / union
        return 1 if ratio > 0.5 else 0


# --------------------------
# 4. 特征提取器 (VGG16)
# --------------------------
class VGGFeatureExtractor:
    def __init__(self):
        # 加载预训练VGG16并提取fc6层特征
        vgg = models.vgg16(pretrained=True)
        self.features = vgg.features
        self.classifier = nn.Sequential(*list(vgg.classifier.children())[:3])  # 到fc6

        # 固定参数
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = False

        self.features.to(device)
        self.classifier.to(device)
        self.features.eval()
        self.classifier.eval()

        # 图像预处理
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract(self, image):
        """提取图像特征 (4096-D)"""
        img = Image.fromarray((image * 255).astype(np.uint8))  # 转换为PIL图像
        img = self.preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = self.features(img)
            feat = feat.view(feat.size(0), -1)
            feat = self.classifier(feat)
        return feat.squeeze().cpu().numpy()

    def extract_patch(self, image, bbox):
        """提取边界框区域特征"""
        x1, y1, x2, y2 = bbox
        patch = image[y1:y2 + 1, x1:x2 + 1]
        return self.extract(patch)


# --------------------------
# 5. MLP分类器
# --------------------------
class MLPClassifier(nn.Module):
    def __init__(self):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(12288, 1024),
            nn.BatchNorm1d(1024),  # 加入BatchNorm，稳定特征分布
            nn.ReLU(),
            nn.Dropout(0.5),  # 提高Dropout，防止过拟合

            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.model(x)


# --------------------------
# 6. 本地 Lama Cleaner 集成（兼容新版 lama-cleaner）
# --------------------------
from lama_cleaner.model_manager import ModelManager

# 尝试导入正确的配置类
try:
    from lama_cleaner.schema import Config

    print("使用 lama_cleaner.schema.Config")
    HAS_SCHEMA_CONFIG = True
except ImportError:
    # 如果 schema 模块不存在或 Config 不存在，则使用字典
    print("lama_cleaner.schema.Config 不存在，使用字典配置")
    HAS_SCHEMA_CONFIG = False


class LocalLamaCleaner:
    def __init__(self, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        print(f"正在加载 Lama 模型到 {device}...")
        self.model = ModelManager(name="lama", device=device)
        print("Lama 模型加载完成！")

    def remove_object(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        通用且兼容性最高的 lama-cleaner inpaint 调用。
        会自动判断 lama-cleaner 的版本和返回类型，并进行统一处理。
        """

        # 1. 预处理 image
        if image.dtype != np.uint8:
            image = np.clip(image * 255, 0, 255).astype(np.uint8)

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.shape[-1] == 4:
            image = image[:, :, :3]

        # 2. 预处理 mask
        if mask.dtype == bool:
            mask = mask.astype(np.uint8) * 255
        elif mask.max() <= 1:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)

        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # 3. 准备配置（尽量简单，避免触发不兼容字段）
        if HAS_SCHEMA_CONFIG:
            # 新版 lama-cleaner 的完整 Config（pydantic v2）
            config = Config(
                ldm_steps=50,
                ldm_sampler="euler",

                # 必填项，绝对不能缺
                hd_strategy="crop",
                hd_strategy_crop_margin=64,
                hd_strategy_crop_trigger_size=512,
                hd_strategy_resize_limit=1536,

                # 其他可选项
                prompt="严格延续原图的纹理方向、色调和亮度，无缝填充背景，保持细节一致性",
                sd_mask_blur=5,
                sd_strength=0.65,
                sd_noise=0.05,
                seed=-1
            )
        else:
            # 旧版 lama-cleaner 使用 dict 配置
            config = {
                "ldm_steps": 25,
                "ldm_sampler": "ddim",
                "hd_strategy": "crop",
                "hd_strategy_crop_margin": 62,
                "hd_strategy_crop_trigger_size": 512,
                "hd_strategy_resize_limit": 1024,
                "prompt": "",
                "sd_mask_blur": 3,
                "sd_strength": 0.8,
                "sd_noise": 0.0,
                "seed": -1,
            }

        # 4. 执行 inpaint
        try:
            out = self.model(image, mask, config)

            # 5. 兼容不同 lama-cleaner 版本的返回格式
            # 5.1 新版 lama-cleaner 返回 dict
            if isinstance(out, dict):
                if "image" in out:
                    result = out["image"]
                    print("lama-cleaner 返回 dict['image']")
                else:
                    raise ValueError("lama-cleaner 返回了 dict，但没有 image 字段。")

            # 5.2 有些版本返回一个对象：需要调用 .inpaint()
            elif hasattr(out, "inpaint"):
                print("lama-cleaner 使用 out.inpaint()")
                result = out.inpaint()

            # 5.3 老版本直接返回图像 ndarray
            elif isinstance(out, np.ndarray):
                print("lama-cleaner 返回 ndarray")
                result = out

            else:
                raise TypeError(f"lama-cleaner 返回了未知类型：{type(out)}")

        except Exception as e:
            print("lama-cleaner 调用失败：", e)
            print("返回原图以避免中断。")
            return image

        # 6. 后处理输出
        if result.dtype != np.uint8:
            if result.max() <= 1.0:
                result = (result * 255).astype(np.uint8)
            else:
                result = np.clip(result, 0, 255).astype(np.uint8)

        if result.ndim == 2:
            result = np.stack([result] * 3, axis=-1)
        elif result.shape[-1] == 1:
            result = np.repeat(result, 3, axis=-1)
        elif result.shape[-1] == 4:
            result = result[:, :, :3]

        # 确保大小与原图一致
        if result.shape[:2] != image.shape[:2]:
            result = cv2.resize(result, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)

        return result


# 初始化本地 Lama 模型（只需一次）
local_lama = LocalLamaCleaner(device=str(device))  # device 是你之前定义的 torch.device


# --------------------------
# 7. 显著性检测推理函数
# --------------------------
def infer_saliency(image, model, sp_processor, feature_extractor):
    """
    对输入图像进行显著性检测
    """
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    h, w = image.shape[:2]

    # 获取超像素
    segments, sp_info = sp_processor.get_superpixels(image)

    # 提取整图特征
    global_feat = feature_extractor.extract(image)

    # 为每个超像素预测显著性
    saliency_map = np.zeros((h, w), dtype=np.float32)
    model.eval()

    with torch.no_grad():
        for i, sp in enumerate(sp_info):
            # 提取特征
            sp_feat = feature_extractor.extract_patch(image, sp['bbox'])

            if sp['neighbors']:
                neighbor_bbox = sp_processor._get_neighbors_bbox(sp['neighbors'], sp_info)
                neighbor_feat = feature_extractor.extract_patch(image, neighbor_bbox)
            else:
                neighbor_feat = np.zeros_like(sp_feat)

            feat = np.concatenate([sp_feat, neighbor_feat, global_feat])
            feat = torch.tensor(feat).to(device).float().unsqueeze(0)

            # 预测
            output = model(feat)
            prob = torch.softmax(output, dim=1)[0, 1].item()  # 前景概率

            # 赋值给超像素区域
            saliency_map[sp['mask']] = prob

    # 后处理：平滑和归一化
    saliency_map = (saliency_map * 255).astype(np.uint8)  # 概率映射到0-255灰度
    saliency_map = cv2.GaussianBlur(saliency_map, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    saliency_map = cv2.morphologyEx(saliency_map, cv2.MORPH_OPEN, kernel)
    saliency_map = np.clip(saliency_map * 255, 0, 255).astype(np.uint8)

    return image, saliency_map, segments, sp_info


def compute_object_saliency_score(saliency_map, mask):
    """
    计算物体掩码内的显著性像素总和（Sum）- Sal-to-Rank转换
    """
    # 只考虑掩码区域内的显著性值
    masked_saliency = saliency_map[mask.astype(bool)]
    # 计算总和作为重要性评分
    saliency_sum = np.sum(masked_saliency) if len(masked_saliency) > 0 else 0
    # 计算平均值作为辅助评分
    saliency_avg = np.mean(masked_saliency) if len(masked_saliency) > 0 else 0

    return saliency_sum, saliency_avg


def iterative_object_removal_with_u2net_sam(image_path, u2net_detector, model, sp_processor, feature_extractor,
                                            max_objects=5):
    """
    使用U2Net+SAM+Lama的迭代物体移除流程
    """
    # 提取图片基础名作为前缀
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 准备输出目录（确保存在）
    output_dir = "iterative_removal_results"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("debug_masks", exist_ok=True)
    os.makedirs("u2net_visualization", exist_ok=True)

    # 加载原始图像
    original_image = io.imread(image_path)
    if original_image.ndim == 2:
        original_image = np.stack([original_image] * 3, axis=-1)

    current_image = original_image.copy()
    removal_history = []  # 存储移除过程的历史
    processed_masks = []  # 存储已处理物体的掩码，用于排除

    # 初始化SAM
    sam_optimizer = SAMOptimizer()

    print(f"开始迭代擦除图像中的最显著物体（使用U2Net+SAM+Lama）...")

    for iteration in range(max_objects):
        print(f"\n=== 迭代 {iteration + 1}: 确定当前最显著物体 ===")

        # 进行显著性检测
        image_result, saliency_map, segments, sp_info = infer_saliency(
            current_image, model, sp_processor, feature_extractor
        )

        if np.max(saliency_map) < 50:
            print(f"当前图像最大显著性过低 ({np.max(saliency_map)}/255)，停止迭代.")
            break

        # 步骤1：使用U2Net检测物体
        print("步骤1: 使用U2Net生成显著性掩码")
        detection = u2net_detector.detect(current_image)
        # 在 detect() 后立刻加一层 显著性一致性检查：
        sal = detection['saliency_map']
        mask = detection['mask']

        # 1. 掩码内 vs 掩码外 显著性对比
        fg_mean = sal[mask == 1].mean() if mask.sum() > 0 else 0
        bg_mean = sal[mask == 0].mean()
        print("掩码内平均显著性: {:.2f}, 掩码外平均显著性: {:.2f}".format(fg_mean, bg_mean))
        # 2. 对比度不足，直接判定为“伪目标”
        if fg_mean < bg_mean * 1.02:
            print("U2Net detected edge-like pseudo foreground, skip.")
            break

        # 3. 掩码边界检查：如果掩码过多接触边界，也视为伪目标
        h, w = mask.shape
        border = np.zeros_like(mask)
        border[:10, :] = 1
        border[-10:, :] = 1
        border[:, :10] = 1
        border[:, -10:] = 1

        border_overlap = np.logical_and(mask, border).sum() / mask.sum()

        if border_overlap > 0.3:
            print("Mask touches image border too much, treat as background.")
            break

        # 可视化U2Net检测结果
        visualize_u2net_results(current_image, detection['mask'], iteration + 1, base_name)
        visualize_u2net_masks(current_image, detection['mask'], iteration + 1, base_name)


        if detection['area'] == 0:
            print("未检测到显著物体，停止迭代。")
            break

        # 排除已经处理过的区域
        overlap_too_much = False
        for processed_mask in processed_masks:
            intersection = np.logical_and(detection['mask'].astype(bool), processed_mask.astype(bool))
            if np.sum(intersection) > 0.4 * np.sum(detection['mask']):  # 如果重叠超过40%，则跳过
                overlap_too_much = True
                break

        if overlap_too_much:
            print("检测到的物体与已处理区域重叠过多，停止迭代。")
            break

        # 计算显著性得分
        saliency_sum, saliency_avg = compute_object_saliency_score(saliency_map, detection['mask'])

        print(
            f"发现显著物体，显著性总分: {saliency_sum:.3f}, "
            f"平均显著性: {saliency_avg:.3f}, "
            f"面积: {detection['area']:.1f}")
        # 新增：基于物体平均显著性停止迭代
        if saliency_avg < saliency_avg_stop_thresh:
            print(f"检测到的物体平均显著性过低 ({saliency_avg:.1f}/255 < {saliency_avg_stop_thresh})，停止迭代。")
            break
        # if saliency_avg < 0.15 * 255:
        #     print("U2Net mask not supported by learned saliency, stop.")
        #     break
        # 步骤2：使用SAM优化检测到的掩码
        print("步骤2: 使用SAM优化显著物体的分割掩码")
        optimized_mask = sam_optimizer.optimize_mask(current_image, detection['mask'], detection['bbox'])

        # 记录本次移除的信息
        removal_record = {
            'iteration': iteration + 1,
            'saliency_sum': saliency_sum,
            'saliency_avg': saliency_avg,
            'bbox': detection['bbox'],
            'original_image': current_image.copy(),
            'saliency_map': saliency_map.copy(),
            'detection_mask': detection['mask'],
            'optimized_mask': optimized_mask
        }
        removal_history.append(removal_record)

        # 步骤3：掩码预处理（提升擦除准确性）- 形态学膨胀
        print("步骤3: 对优化后的掩码进行形态学膨胀（ER=3）")
        refined_mask = refine_mask_for_inpaint(optimized_mask, detection['bbox'], current_image.shape, er_rate=3)

        # 保存调试信息
        visualize_debug_mask(current_image, saliency_map, optimized_mask, refined_mask, detection['bbox'],
                            image_name=base_name,iteration=iteration + 1)

        # 将当前处理的掩码添加到已处理列表
        processed_masks.append(refined_mask)

        # 步骤4：物体擦除（核心操作）- 使用LAMA工具
        print("步骤4: 使用LAMA工具擦除显著物体")
        current_image = remove_object_with_lama_cleaner(current_image, refined_mask)

        # 步骤5：可视化当前状态
        print("步骤5: 保存当前迭代结果")
        visualize_removal_process(original_image, current_image, saliency_map, refined_mask, iteration + 1,image_name=base_name)

        # 保存中间结果（文件名前缀包含 base_name）
        result_path = os.path.join(output_dir, f"{base_name}_iteration_{iteration + 1}_result_u2net_sam.jpg")
        io.imsave(result_path, current_image)
        print(f"迭代 {iteration + 1} 结果已保存到: {result_path}")

    print(f"\n=== 迭代擦除完成 ===")
    print(f"总共处理了 {len(removal_history)} 个显著物体。")

    # 保存最终结果（文件名前缀包含 base_name）
    final_result_path = os.path.join(output_dir, f"{base_name}_final_cleaned_image_u2net_sam.jpg")
    io.imsave(final_result_path, current_image)
    print(f"最终清理后的图像已保存到: {final_result_path}")

    return current_image, removal_history


def refine_mask_for_inpaint(mask, bbox, image_shape, er_rate):
    """
    改进版 Mask 优化逻辑：使用更小的形态学膨胀（ER=3）扩展掩码边缘
    """
    h, w = image_shape[:2]
    mask = mask.astype(np.uint8)

    # 确保 mask 是 0/1
    mask = (mask > 0).astype(np.uint8)

    # 计算物体尺寸
    obj_w = bbox[2] - bbox[0]
    obj_h = bbox[3] - bbox[1]
    obj_size = max(obj_w, obj_h)

    # --- 步骤3：形态学膨胀（ER=3），进一步减少掩码边缘扩展 ---
    # 根据ER参数设置膨胀核大小
    dilate_kernel_size = er_rate  # ER=3，进一步减小膨胀程度
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size))
    mask = cv2.dilate(mask, kernel_dilate, iterations=1)

    # --- 修复 1: 填充空洞 (改用 Morphological Closing) ---
    # 根据物体大小决定闭运算核的大小
    close_k = max(3, int(obj_size * 0.05))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    # --- 修复 4: 动态平滑 (关键) ---
    # 模糊核不能太大，否则 mask 会消失
    blur_k = int(obj_size * 0.15)  # 减小模糊核大小
    if blur_k % 2 == 0: blur_k += 1  # 必须是奇数
    blur_k = max(3, min(blur_k, 31))  # 限制在 3 到 31 之间

    mask = mask.astype(np.float32)
    mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)

    # 降低阈值，防止边缘因为模糊变暗而被切掉
    mask = (mask > 0.1).astype(np.uint8)

    return (mask * 255).astype(np.uint8)


def remove_object_with_lama_cleaner(image, mask):
    """
    使用lama-cleaner服务移除对象
    """
    # 新增：强制转换为RGB（避免BGR导致的色调问题）
    if image.ndim == 3 and image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 确保是RGB格式
    return local_lama.remove_object(image, mask)


def batch_iterative_removal_with_u2net_sam(input_dir, output_dir, u2net_detector, model, sp_processor,
                                           feature_extractor,
                                           max_objects):
    """
    批量处理目录中的图像（使用U2Net+SAM+Lama）
    """
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
        image_files.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))

    # 去重文件列表
    image_files = list(set(image_files))

    print(f"找到 {len(image_files)} 张图像进行处理")

    for i, image_path in enumerate(image_files):
        print(f"\n处理第 {i + 1}/{len(image_files)} 张图像: {os.path.basename(image_path)}")

        try:
            # 对单张图像执行迭代移除（使用U2Net+SAM+Lama）
            final_image, history = iterative_object_removal_with_u2net_sam(
                image_path, u2net_detector, model, sp_processor, feature_extractor, max_objects
            )

            # 保存最终结果
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            final_output_path = os.path.join(output_dir, f"{base_name}_cleaned_u2net_sam.jpg")
            io.imsave(final_output_path, final_image)
            print(f"最终结果已保存到: {final_output_path}")

            # 保存移除历史记录
            history_path = os.path.join(output_dir, f"{base_name}_removal_history_u2net_sam.txt")
            with open(history_path, 'w') as f:
                f.write(f"移除历史记录 - {base_name}\n")
                f.write("=" * 50 + "\n")
                for record in history:
                    f.write(f"迭代 {record['iteration']}: 显著性总分 {record['saliency_sum']:.3f}, "
                            f"平均显著性 {record['saliency_avg']:.3f}, "
                            f"边界框 {record['bbox']}\n")

        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {str(e)}")
            continue

    print(f"\n批量处理完成！结果保存在: {output_dir}")


# --------------------------
# 主函数
# --------------------------
if __name__ == "__main__":
    # 使用 spawn，避免 fork+CUDA 冲突（跨平台更安全）
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    saliency_avg_stop_thresh = 20.0  # 平均显著性阈值，低于此值停止迭代
    # U2Net模型路径
    U2NET_MODEL_PATH = "./checkpoints/u2net.pth"

    # 检查U2Net文件是否存在
    if not os.path.exists(U2NET_MODEL_PATH):
        print(f"错误：找不到U2Net检查点文件: {U2NET_MODEL_PATH}")
        print("请确保U2Net检查点文件存在后再运行此脚本。")
        exit(1)

    # 初始化组件
    sp_processor = SuperpixelProcessor(num_superpixels=300, compactness=10, sigma=1.0)
    feature_extractor = VGGFeatureExtractor()
    model = MLPClassifier().to(device)

    # 模型路径 (请替换为实际路径)
    model_path = "./pths/saliency_model4.pth"  # 请替换为你的模型路径

    if os.path.exists(model_path):
        print(f"找到模型 `{model_path}`，加载模型。")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    else:
        print(f"错误：找不到模型文件 `{model_path}`")
        print("请确保模型文件存在后再运行此脚本。")
        exit(1)

    # 初始化U2Net检测器
    try:
        u2net_detector = U2NetDetector(U2NET_MODEL_PATH)
    except Exception as e:
        print(f"初始化U2Net失败: {e}")
        exit(1)

    # 批量处理示例
    input_directory = "./test_images_for_remove"  # 输入图像目录
    output_directory = "./batch_output_u2net_sam"  # 输出目录

    if os.path.exists(input_directory):
        print(f"\n开始批量处理目录: {input_directory} (使用U2Net+SAM+Lama)")
        batch_iterative_removal_with_u2net_sam(
            input_directory, output_directory, u2net_detector, model, sp_processor, feature_extractor, max_objects=6
        )
    else:
        print(f"输入目录不存在: {input_directory}")