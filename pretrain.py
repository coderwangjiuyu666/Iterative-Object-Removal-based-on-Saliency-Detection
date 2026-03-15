# 编写预计算脚本（首次运行一次）
import os
import torch
import numpy as np
import glob
from skimage import io
from SaliencyDetection import SuperpixelProcessor
from SaliencyDetection import VGGFeatureExtractor

def precompute_data(image_dir, gt_dir, save_dir, sp_processor, feature_extractor):
    os.makedirs(save_dir, exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    gt_paths = sorted(glob.glob(os.path.join(gt_dir, "*.png")))

    total_sp = 0  # 总超像素数
    total_pos = 0  # 前景超像素数

    # 正确解包 zip，避免把索引传给 imread
    for idx, (img_path, gt_path) in enumerate(zip(image_paths, gt_paths)):
        try:
            image = io.imread(img_path)
        except Exception as e:
            print(f"跳过无法读取的图像: {img_path}，错误: {e}")
            continue

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        try:
            gt = io.imread(gt_path, as_gray=True)
        except Exception as e:
            print(f"跳过无法读取的GT: {gt_path}，错误: {e}")
            continue

        gt = (gt > 127).astype(np.bool_)

        segments, sp_info = sp_processor.get_superpixels(image)

        global_feat = feature_extractor.extract(image)
        sp_feats = []
        sp_labels = []

        for sp in sp_info:
            sp_feat = feature_extractor.extract_patch(image, sp['bbox'])
            if sp['neighbors']:
                neighbor_bbox = sp_processor._get_neighbors_bbox(sp['neighbors'], sp_info)
                neighbor_feat = feature_extractor.extract_patch(image, neighbor_bbox)
            else:
                neighbor_feat = np.zeros_like(sp_feat)
            full_feat = np.concatenate([sp_feat, neighbor_feat, global_feat]).astype(np.float32)
            label = sp_processor.get_superpixel_label(sp['mask'], gt)

            sp_feats.append(full_feat)
            sp_labels.append(label)
        # 新增：累加统计
        total_sp += len(sp_labels)
        total_pos += sum(sp_labels)


        img_name = os.path.splitext(os.path.basename(img_path))[0] + ".npz"
        np.savez_compressed(
            os.path.join(save_dir, img_name),
            feats=np.array(sp_feats),
            labels=np.array(sp_labels)
        )
    # 新增：打印统计结果
    pos_ratio = total_pos / total_sp if total_sp > 0 else 0.0
    neg_ratio = 1 - pos_ratio
    print(f"===== 标签分布统计 =====")
    print(f"总超像素数：{total_sp}")
    print(f"前景超像素数：{total_pos}，占比：{pos_ratio:.3f}")
    print(f"背景超像素数：{total_sp - total_pos}，占比：{neg_ratio:.3f}")
        # print(f"预计算完成：{img_name}（{len(sp_feats)}个超像素）")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp_processor = SuperpixelProcessor(num_superpixels=300, compactness=10, sigma=1.0)
    feature_extractor = VGGFeatureExtractor()

    image_dir = "./DUTS-TR/DUTS-TR-Image"
    gt_dir = "./DUTS-TR/DUTS-TR-Mask"
    save_dir = "./DUTS-TR/precomputed_data"

    print("开始预计算数据...")
    precompute_data(image_dir, gt_dir, save_dir, sp_processor, feature_extractor)
