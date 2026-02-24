import os
import sys

print(">>> 脚本已启动，正在加载依赖库 (这可能需要几秒钟)...", flush=True)

import torch
import numpy as np
from PIL import Image

# MedSAM2 Root Directory
MEDSAM2_ROOT = "/home/alex/Project/MRI/MedSAM2"
# Add MedSAM2 to sys.path
sys.path.append(MEDSAM2_ROOT)

print(f">>> 正在导入 MedSAM2 模块...", flush=True)
from sam2.build_sam import build_sam2_video_predictor

# --- Configuration ---
# Checkpoint path
CHECKPOINT_PATH = os.path.join(MEDSAM2_ROOT, "checkpoints/MedSAM2_latest.pt")
# Config file (relative to sam2 module)
CONFIG_FILE = "configs/sam2.1_hiera_t512.yaml"
# Data path
NPZ_FILE = os.path.join(MEDSAM2_ROOT, "data/validation_public_npz/CT_Lesion_FLARE23Ts_0007.npz")
OUTPUT_FILE = "extracted_features.npz"

def resize_grayscale_to_rgb_and_resize(array, image_size):
    """
    预处理函数：将灰度 3D 图像转换为 RGB 并调整大小以适配模型输入
    """
    d, h, w = array.shape
    resized_array = np.zeros((d, 3, image_size, image_size))
    
    for i in range(d):
        # 简单的 min-max 归一化到 0-255 用于 PIL
        slice_img = array[i]
        if slice_img.max() > slice_img.min():
            slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min()) * 255.0
        
        img_pil = Image.fromarray(slice_img.astype(np.uint8))
        img_rgb = img_pil.convert("RGB")
        img_resized = img_rgb.resize((image_size, image_size))
        img_array = np.array(img_resized).transpose(2, 0, 1)
        resized_array[i] = img_array
    
    return resized_array

def main():
    print("=== 初始化 MedSAM2 特征提取器 ===")
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"错误: 未找到模型权重 {CHECKPOINT_PATH}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    try:
        # 加载模型 (仅加载预测器，不需要训练部分)
        predictor = build_sam2_video_predictor(CONFIG_FILE, CHECKPOINT_PATH, device=device)
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        print("请检查配置文件路径是否正确。")
        return

    if not os.path.exists(NPZ_FILE):
        print(f"数据文件未找到: {NPZ_FILE}")
        return
        
    print(f"正在加载数据: {NPZ_FILE}...")
    npz_data = np.load(NPZ_FILE, allow_pickle=True)
    imgs = npz_data['imgs'] # (Depth, Height, Width)
    print(f"原始图像尺寸: {imgs.shape}")
    
    # --- 预处理 ---
    print("正在预处理图像...")
    # 调整大小到 512x512 (MedSAM2 标准输入)
    img_resized = resize_grayscale_to_rgb_and_resize(imgs, 512)
        
    # 转为 Tensor 并进行 ImageNet 标准化
    img_tensor = torch.from_numpy(img_resized).float()
    img_tensor = img_tensor / 255.0
    img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    img_tensor -= img_mean
    img_tensor /= img_std
    
    img_tensor = img_tensor.to(device)
    
    # --- 特征提取核心逻辑 ---
    print("开始提取高维特征...")
    features_list = []
    batch_size = 4 # 根据显存大小调整
    
    # 获取底层的模型实例
    # SAM2VideoPredictor 继承自 SAM2Base，它本身就是模型
    model = predictor
    
    with torch.no_grad():
        for i in range(0, len(img_tensor), batch_size):
            batch = img_tensor[i : i + batch_size]
            
            # 直接调用 image_encoder 的前向传播
            backbone_out = model.forward_image(batch)
            
            # backbone_out["backbone_fpn"] 是一个列表，包含不同分辨率的特征图
            # [-1] 取最后一层，通常是语义信息最丰富、分辨率最低的层
            # 形状通常是 (Batch, Channels, H/32, W/32)
            deepest_feat = backbone_out["backbone_fpn"][-1] 
            
            # 【关键步骤】向量化
            # 使用全局平均池化 (Global Average Pooling) 将空间特征图压缩为一维向量
            # 这样每张切片就变成了一个长度为 C (例如 256 或 768) 的向量
            gap_feat = torch.mean(deepest_feat, dim=(2, 3))
            
            features_list.append(gap_feat.cpu().numpy())
            
            if i == 0:
                print(f"  特征图原始形状: {deepest_feat.shape}")
                print(f"  向量化后形状: {gap_feat.shape}")
            
    all_features = np.concatenate(features_list, axis=0)
    print(f"=== 提取完成 ===")
    print(f"最终特征矩阵形状: {all_features.shape} (切片数 x 特征维度)")
    
    # 保存为向量文件
    np.savez_compressed(OUTPUT_FILE, features=all_features)
    print(f"特征已保存至: {os.path.abspath(OUTPUT_FILE)}")
    print("您可以使用此文件进行下游分类、检索或聚类任务。")

if __name__ == "__main__":
    main()