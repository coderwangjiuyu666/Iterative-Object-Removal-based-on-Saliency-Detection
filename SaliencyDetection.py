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
# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
torch.backends.cudnn.benchmark = True  # 根据输入尺寸自动选择更快算法
# --------------------------
# 1. 超像素处理工具类
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
            # 返回一个空的无效 bbox，调用端应处理（或返回零尺寸）
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
# 2. 特征提取器 (VGG16)
# --------------------------
class VGGFeatureExtractor:
    def __init__(self):
        # 加载预训练VGG16并提取fc6层特征
        super().__init__()
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
# 3. MLP分类器
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
# 4. 数据集类（支持训练/验证集划分）
# --------------------------
class CachedSaliencyDataset(Dataset):
    def __init__(self, cache_dir, mode="train", val_ratio=0.1, seed=42):
        """
        支持训练集/验证集划分的缓存数据集
        Args:
            cache_dir: 缓存文件（.npz）所在目录
            mode: 数据集模式，"train" 或 "val"
            val_ratio: 验证集占比（0-1之间），默认10%
            seed: 随机种子（确保划分结果可重复）
        """
        super().__init__()
        # 加载所有缓存文件路径
        self.cache_files = sorted(glob.glob(os.path.join(cache_dir, "*.npz")))
        if not self.cache_files:
            raise ValueError(f"缓存目录 {cache_dir} 中未找到 .npz 文件，请先运行 pretrain.py 预计算数据")

        # 固定随机种子，确保划分结果可重复
        np.random.seed(seed)
        torch.manual_seed(seed)

        # 数据集总数量
        total_size = len(self.cache_files)
        # 验证集数量（向下取整）
        val_size = int(total_size * val_ratio)

        # 生成随机索引，划分训练/验证集
        shuffled_indices = np.random.permutation(total_size)
        if mode == "train":
            # 训练集：排除前 val_size 个样本
            self.selected_indices = shuffled_indices[val_size:]
        elif mode == "val":
            # 验证集：取前 val_size 个样本
            self.selected_indices = shuffled_indices[:val_size]
        else:
            raise ValueError(f"mode 必须是 'train' 或 'val'，当前传入：{mode}")

        # 筛选出当前模式对应的缓存文件
        self.selected_files = [self.cache_files[i] for i in self.selected_indices]
        print(
            f"[{mode.upper()}] 数据集加载完成：{len(self.selected_files)} 个样本（总样本数：{total_size}，验证集占比：{val_ratio}）")

    def __len__(self):
        # 返回当前模式下的样本数
        return len(self.selected_files)

    def __getitem__(self, idx):
        # 读取当前模式下的第 idx 个样本
        cache_file = self.selected_files[idx]
        # 内存映射加载，避免占用过多内存
        with np.load(cache_file, mmap_mode='r') as data:
            feats = torch.from_numpy(data["feats"]).float()  # 转换为float（匹配模型输入）
            labels = torch.from_numpy(data["labels"]).long()
        return feats, labels

    def _get_neighbors_bbox(self, neighbor_ids, sp_info):
        """原有的邻域边界框计算方法（保持不变）"""
        bboxes = [sp_info[i]['bbox'] for i in neighbor_ids if 0 <= i < len(sp_info)]
        if not bboxes:
            return (0, 0, 0, 0)
        x1 = min(bbox[0] for bbox in bboxes)
        y1 = min(bbox[1] for bbox in bboxes)
        x2 = max(bbox[2] for bbox in bboxes)
        y2 = max(bbox[3] for bbox in bboxes)
        return (x1, y1, x2, y2)

# --------------------------
# 5. 训练函数
# --------------------------
# 新增验证函数
def evaluate(model, val_loader):
    model.eval()
    tp = tn = fp = fn = 0
    with torch.no_grad():
        for feats, labels in val_loader:
            feats = feats.to(device).float()
            labels = labels.to(device).long()
            outputs = model(feats)
            preds = torch.argmax(outputs, dim=1)
            tp += ((preds == 1) & (labels == 1)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0
    print(f"Val Recall: {recall:.3f}, Precision: {precision:.3f}, F1: {f1:.3f}")
    return f1

def train_model(train_loader,val_loader, model, criterion, optimizer, scheduler, num_epochs=30,log_dir="./training_logs"):
    os.makedirs(log_dir, exist_ok=True)
    model.train()
    best_val_f1 = 0.0
    # 用于记录曲线
    train_losses = []
    val_f1s = []
    epochs_list = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}(Train)"):
            features = features.to(device).float()
            labels = labels.to(device).long()
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item() * features.size(0)
            del features, labels, outputs, loss

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100 * correct / total
        scheduler.step()
        # 每个epoch后评估验证集
        val_f1 = evaluate(model, val_loader)
        # 记录
        train_losses.append(epoch_loss)
        val_f1s.append(val_f1)
        epochs_list.append(epoch + 1)
        # 保存最优模型（基于验证集F1）
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "saliency_model_best.pth")
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%, Val F1: {val_f1:.3f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        # 每个 epoch 保存日志 CSV 和曲线（方便中断后查看）
        log_csv = os.path.join(log_dir, "training_log.csv")
        # 保存为两列: epoch, train_loss, val_f1
        header = "epoch,train_loss,val_f1"
        data = np.column_stack([np.array(epochs_list), np.array(train_losses), np.array(val_f1s)])
        np.savetxt(log_csv, data, delimiter=",", header=header, comments="", fmt="%.6f")

        # 绘制曲线：loss 与 val_f1（两个子图）
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_list, train_losses, marker="o", label="Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs_list, val_f1s, marker="o", color="orange", label="Val F1")
        plt.xlabel("Epoch")
        plt.ylabel("F1")
        plt.title("Validation F1")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "loss_f1_curve.png"), dpi=150)
        plt.close()
    # 保存最终模型
    torch.save(model.state_dict(), "saliency_model_final.pth")
    print(f"\n训练完成！最终模型保存到 `saliency_model_final.pth`，最优验证集F1：{best_val_f1:.3f}")
    return model


# --------------------------
# 6. 推理函数
# --------------------------
def infer_saliency(image_path, model, sp_processor, feature_extractor):
    # 加载图像
    image = io.imread(image_path)
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
            if i % 50 == 0:  # 每50个超像素打印一次
                print(f"超像素{i}，前景概率：{prob:.3f}")
            # 赋值给超像素区域
            saliency_map[sp['mask']] = prob

    # 替换后处理逻辑（关键修改：去掉二值化，保留渐变）
    saliency_map = (saliency_map * 255).astype(np.uint8)  # 概率映射到0-255灰度
    # 1. 高斯模糊平滑噪声（保留渐变，让概率过渡更自然）
    saliency_map = cv2.GaussianBlur(saliency_map, (5, 5), 0)
    # 2. 轻微形态学开运算（仅去除小噪点，不破坏渐变）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    saliency_map = cv2.morphologyEx(saliency_map, cv2.MORPH_OPEN, kernel)
    # 3. 可选：归一化到0-255（确保渐变范围完整）
    saliency_map = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX)

    return image, saliency_map


def concat_superpixels_collate(batch):
    # batch: List[Tuple[np.ndarray(features_i), np.ndarray(labels_i)]]
    feats_list, labels_list = [], []
    for feats, labels in batch:
        if isinstance(feats, np.ndarray):
            feats = torch.from_numpy(feats)
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)
        feats_list.append(feats.float())
        labels_list.append(labels.long())
    feats = torch.cat(feats_list, dim=0)    # [sum(N_i), D]
    labels = torch.cat(labels_list, dim=0)  # [sum(N_i)]
    return feats, labels

def batch_infer_saliency(input_dir, output_dir, model, sp_processor, feature_extractor):
    # 确保输出文件夹存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入文件夹中的所有图片
    for image_name in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_name)
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue  # 跳过非图像文件

        print(f"正在处理: {image_name}")
        image, saliency_map = infer_saliency(image_path, model, sp_processor, feature_extractor)

        # 保存结果
        output_path = os.path.join(output_dir, f"saliency_{image_name}")
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(122)
        im = plt.imshow(saliency_map, cmap="viridis")  # 使用彩色热力图
        plt.title("Saliency Map")
        plt.axis("off")
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label("Foreground Probability (0-255)")  # 颜色条标签

        plt.tight_layout()
        plt.savefig(output_path,bbox_inches='tight')
        plt.close()
        print(f"结果图已保存到 `{output_path}`")

# 替换交叉熵损失为Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2,class_weight=None):
        super().__init__()
        self.alpha = alpha  # 前景类的权重因子
        self.gamma = gamma  # 难样本聚焦因子
        self.class_weight= class_weight

    def forward(self, outputs, labels):
        # 传入class_weight，平衡样本分布
        ce_loss = nn.functional.cross_entropy(
            outputs, labels, weight=self.class_weight, reduction="none"
        )
        pt = torch.exp(-ce_loss)  # 预测概率
        # 处理alpha（平衡前景/背景）
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                # 单个alpha：前景权重=alpha，背景=1-alpha
                alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            else:
                # 二元组alpha：分别对应背景、前景
                alpha_t = self.alpha[labels]
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# --------------------------
# 主函数
# --------------------------
if __name__ == "__main__":
    # 使用 spawn，避免 fork+CUDA 冲突（跨平台更安全）
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 初始化组件
    sp_processor = SuperpixelProcessor(num_superpixels=300, compactness=10, sigma=1.0)
    feature_extractor = VGGFeatureExtractor()
    model = MLPClassifier().to(device)

    # 数据集路径 (请替换为实际路径)
    train_image_dir = "./DUTS-TR/DUTS-TR-Image"
    train_gt_dir = "./DUTS-TR/DUTS-TR-Mask"

    model_path = "./pths/saliency_model4.pth"
    cache_dir = "./DUTS-TR/precomputed_data"

    if os.path.exists(model_path):
        print(f"找到模型 `{model_path}`，跳过训练并加载模型。")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("未找到模型，开始训练。")
        # 1. 创建训练集和验证集（核心修改）
        train_dataset = CachedSaliencyDataset(
            cache_dir=cache_dir,
            mode="train",
            val_ratio=0.1,  # 验证集占10%，可调整
            seed=42  # 固定种子，确保划分一致
        )
        val_dataset = CachedSaliencyDataset(
            cache_dir=cache_dir,
            mode="val",
            val_ratio=0.1,  # 必须与训练集的 val_ratio 一致
            seed=42  # 必须与训练集的 seed 一致
        )

        # 2. 创建训练集和验证集的 DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=True,
            num_workers=4,
            pin_memory=False,
            collate_fn=concat_superpixels_collate,
            prefetch_factor=2
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=64,
            shuffle=False,  # 验证集不打乱
            num_workers=4,
            pin_memory=False,
            collate_fn=concat_superpixels_collate,
            prefetch_factor=2
        )
        pos_ratio=0.256
        neg_ratio=1-pos_ratio
        class_weights = torch.tensor([1/neg_ratio, 1/pos_ratio], device=device)
        criterion = FocalLoss(alpha=pos_ratio, gamma=2,class_weight=class_weights)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.6)

        # 传入 val_loader 到训练函数
        model = train_model(
            train_loader=train_loader,
            val_loader=val_loader,  # 新增验证集加载器
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=50
        )
        torch.save(model.state_dict(), model_path)
        print(f"训练完成并保存到 `{model_path}`。")

    # 批量推理
    input_dir = "./test_images"  # 输入文件夹路径
    output_dir = "./output_results"  # 输出文件夹路径

    batch_infer_saliency(input_dir, output_dir, model, sp_processor, feature_extractor)
