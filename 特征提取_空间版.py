#!/usr/bin/env python3
"""
MedSAM2 空间特征提取脚本
从MedSAM2编码器提取空间特征图，用于nnUNet多通道输入

与原版特征提取的区别：
- 原版：提取256维全局向量（全局平均池化后）
- 本版：保留空间特征图，多层融合后保存为NIfTI格式

输出：
- 每个3D体数据对应一个空间特征体积，保存为 .nii.gz 文件
- 特征通道数：可选择256（完整）或128（压缩）
"""

import sys
sys.path.append('/home/alex/Project/MRI/MedSAM2')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import blosc2
from typing import List, Dict, Tuple, Optional
import os
from skimage.transform import resize
from tqdm import tqdm
import json

# 手动初始化Hydra配置
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate


def build_sam2_model(config_file: str, ckpt_path: str, device: str = 'cuda'):
    """
    构建SAM2模型
    """
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    config_dir = '/home/alex/Project/MRI/MedSAM2/sam2/configs'
    initialize_config_dir(config_dir=config_dir, version_base='1.2')
    cfg = compose(config_name=config_file)

    model = instantiate(cfg.model, _recursive_=True)

    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)['model']
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
        print(f"Loaded checkpoint from {ckpt_path}")

    model = model.to(device)
    model.eval()
    return model


class ChannelReducer(nn.Module):
    """
    1x1卷积通道压缩器
    将256通道压缩到指定通道数
    """
    def __init__(self, in_channels: int = 256, out_channels: int = 128):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class MedSAM2SpatialFeatureExtractor:
    """
    MedSAM2 空间特征提取器
    提取多层特征图并进行融合，保留空间信息
    """

    def __init__(self,
                 checkpoint_path: str,
                 device: str = 'cuda',
                 image_size: Tuple[int, int] = (512, 512),
                 output_channels: int = 128,
                 use_multilevel: bool = True):
        """
        初始化空间特征提取器

        Args:
            checkpoint_path: MedSAM2权重路径
            device: 计算设备
            image_size: 输入图像尺寸
            output_channels: 输出特征通道数 (128 或 256)
            use_multilevel: 是否使用多层特征融合
        """
        self.device = device
        self.image_size = image_size
        self.output_channels = output_channels
        self.use_multilevel = use_multilevel

        # 加载模型
        print(f"Loading MedSAM2 from {checkpoint_path}...")
        config_file = "sam2.1_hiera_t512.yaml"
        self.model = build_sam2_model(config_file=config_file, ckpt_path=checkpoint_path, device=device)

        # 冻结权重
        for param in self.model.parameters():
            param.requires_grad = False

        # 通道压缩器 (如果需要)
        self.channel_reducer = None
        if output_channels != 256:
            self.channel_reducer = ChannelReducer(256, output_channels).to(device)
            # 初始化为恒等映射的近似
            print(f"Channel reducer initialized: 256 -> {output_channels}")

        print("MedSAM2 Spatial Feature Extractor loaded!")

    def load_preprocessed_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载预处理后的图像数据和仿射矩阵

        Args:
            data_path: 数据文件路径 (.b2nd 或 .nii.gz)

        Returns:
            data: 3D numpy数组 (D, H, W)
            affine: 4x4仿射矩阵
        """
        if data_path.endswith('.b2nd'):
            arr = blosc2.open(data_path)
            data = arr[()]
            if data.ndim == 4 and data.shape[0] == 1:
                data = data.squeeze(0)
            # b2nd格式没有仿射矩阵，使用单位矩阵
            affine = np.eye(4)
        elif data_path.endswith('.nii.gz') or data_path.endswith('.nii'):
            nii = nib.load(data_path)
            data = nii.get_fdata()
            affine = nii.affine
        else:
            raise ValueError(f"Unsupported file format: {data_path}")

        return data, affine

    def load_segmentation(self, seg_path: str) -> np.ndarray:
        """加载分割标注"""
        data, _ = self.load_preprocessed_data(seg_path)
        return data

    def preprocess_slice(self, slice_data: np.ndarray) -> torch.Tensor:
        """
        预处理单个切片

        Args:
            slice_data: 2D切片数据 (H, W)

        Returns:
            预处理后的张量 (1, 3, 512, 512)
        """
        # 归一化到 0-1
        data_min = slice_data.min()
        data_max = slice_data.max()
        if data_max - data_min > 1e-6:
            slice_normalized = (slice_data - data_min) / (data_max - data_min)
        else:
            slice_normalized = np.zeros_like(slice_data)

        # 重采样到模型输入尺寸
        slice_resized = resize(slice_normalized,
                               self.image_size,
                               mode='reflect',
                               anti_aliasing=True)

        # 转为张量 (1, 1, H, W) -> (1, 3, H, W)
        tensor = torch.from_numpy(slice_resized).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        tensor = tensor.repeat(1, 3, 1, 1)

        return tensor.to(self.device)

    def extract_spatial_features_from_slice(self,
                                             slice_data: np.ndarray,
                                             original_size: Tuple[int, int]) -> np.ndarray:
        """
        从单个2D切片提取空间特征图

        Args:
            slice_data: 2D切片数据 (H, W)
            original_size: 原始图像尺寸 (H, W)

        Returns:
            空间特征图 (C, H, W) - 已上采样到原始尺寸
        """
        input_tensor = self.preprocess_slice(slice_data)

        with torch.no_grad():
            encoder_output = self.model.image_encoder(input_tensor)

        # 获取多层特征
        if isinstance(encoder_output, dict):
            backbone_fpn = encoder_output['backbone_fpn']
        else:
            backbone_fpn = [encoder_output]

        if self.use_multilevel and len(backbone_fpn) >= 2:
            # 多层特征融合：使用最高分辨率层(Level 0)和次高分辨率层(Level 1)
            # Level 0: (1, 256, 128, 128) stride=4
            # Level 1: (1, 256, 64, 64) stride=8

            feat_high = backbone_fpn[0]  # 高分辨率特征
            feat_mid = backbone_fpn[1]   # 中分辨率特征

            # 上采样中分辨率特征到高分辨率尺寸
            feat_mid_up = F.interpolate(feat_mid, size=feat_high.shape[2:],
                                        mode='bilinear', align_corners=False)

            # 特征融合 (相加)
            fused_feature = feat_high + feat_mid_up
        else:
            # 只使用最高分辨率层
            fused_feature = backbone_fpn[0]

        # 通道压缩
        if self.channel_reducer is not None:
            with torch.no_grad():
                fused_feature = self.channel_reducer(fused_feature)

        # 上采样到原始图像尺寸
        feature_map = F.interpolate(fused_feature, size=original_size,
                                    mode='bilinear', align_corners=False)

        # 转换为numpy: (1, C, H, W) -> (C, H, W)
        return feature_map.squeeze(0).cpu().numpy()

    def extract_spatial_features_from_volume(self,
                                              volume: np.ndarray,
                                              affine: np.ndarray = None,
                                              max_slices: int = 200,
                                              verbose: bool = True) -> nib.Nifti1Image:
        """
        从3D体数据提取空间特征体积

        Args:
            volume: 3D体数据 (D, H, W)
            affine: 仿射矩阵
            max_slices: 最大处理切片数
            verbose: 是否显示进度

        Returns:
            NIfTI图像对象，包含空间特征体积
        """
        D, H, W = volume.shape

        # 确定通道数
        C = self.output_channels

        # 初始化特征体积
        feature_volume = np.zeros((D, C, H, W), dtype=np.float32)

        # 确定要处理的切片
        slice_indices = list(range(D))

        # 处理每个切片
        iterator = tqdm(slice_indices, desc="Extracting spatial features", ascii=True) if verbose else slice_indices

        for d in iterator:
            slice_data = volume[d]

            # 跳过空白切片
            if np.abs(slice_data).max() < 1e-6:
                continue

            # 提取空间特征
            feature_map = self.extract_spatial_features_from_slice(
                slice_data, original_size=(H, W)
            )

            feature_volume[d] = feature_map

        # 保存为NIfTI
        if affine is None:
            affine = np.eye(4)

        # 注意：NIfTI的维度顺序是 (x, y, z, c)，我们需要转置
        # 当前: (D, C, H, W) -> NIfTI: (H, W, D, C)
        feature_volume_nifti = np.transpose(feature_volume, (2, 3, 0, 1))

        nii_img = nib.Nifti1Image(feature_volume_nifti, affine)

        return nii_img

    def process_dataset(self,
                        source_dataset: str,
                        target_dataset: str,
                        use_preprocessed: bool = True):
        """
        处理整个数据集，生成特征增强数据

        Args:
            source_dataset: 源数据集路径 (Dataset001_MyCenter)
            target_dataset: 目标数据集路径 (Dataset002_MedSAM_Enhanced)
            use_preprocessed: 是否使用预处理后的数据
        """
        # 创建目标目录
        imagesTr_dir = os.path.join(target_dataset, 'imagesTr')
        labelsTr_dir = os.path.join(target_dataset, 'labelsTr')
        os.makedirs(imagesTr_dir, exist_ok=True)
        os.makedirs(labelsTr_dir, exist_ok=True)

        if use_preprocessed:
            preproc_dir = os.path.join(source_dataset, 'nnUNetPlans_3d_fullres')
            gt_dir = os.path.join(source_dataset, 'gt_segmentations')

            # 获取所有病例
            case_files = [f for f in os.listdir(preproc_dir)
                         if f.endswith('.b2nd') and not f.endswith('_seg.b2nd')]
            case_ids = sorted(list(set([f.replace('.b2nd', '') for f in case_files])))

            print(f"\nFound {len(case_ids)} cases to process")
            print(f"Source: {source_dataset}")
            print(f"Target: {target_dataset}")
            print(f"Output channels: {self.output_channels}")
            print("="*60)

            results = []

            for i, case_id in enumerate(case_ids):
                print(f"\n[{i+1}/{len(case_ids)}] Processing {case_id}...")

                try:
                    # 加载图像
                    img_path = os.path.join(preproc_dir, f'{case_id}.b2nd')
                    seg_path = os.path.join(preproc_dir, f'{case_id}_seg.b2nd')

                    # 如果预处理目录没有分割，尝试原始分割目录
                    if not os.path.exists(seg_path):
                        seg_path = os.path.join(gt_dir, f'{case_id}.nii.gz')

                    if not os.path.exists(img_path):
                        print(f"  Image not found: {img_path}")
                        continue

                    # 加载数据
                    volume, affine = self.load_preprocessed_data(img_path)
                    print(f"  Volume shape: {volume.shape}")

                    # 提取空间特征
                    feature_nii = self.extract_spatial_features_from_volume(volume, affine)

                    # 保存特征为第二个通道 (_0001.nii.gz)
                    feature_path = os.path.join(imagesTr_dir, f'{case_id}_0001.nii.gz')
                    nib.save(feature_nii, feature_path)
                    print(f"  Saved feature: {feature_path}")

                    # 复制原始图像作为第一个通道 (_0000.nii.gz)
                    # 由于预处理数据是b2nd格式，需要转换
                    original_img_path = os.path.join(imagesTr_dir, f'{case_id}_0000.nii.gz')
                    original_nii = nib.Nifti1Image(volume, affine)
                    nib.save(original_nii, original_img_path)
                    print(f"  Saved original: {original_img_path}")

                    # 复制标签
                    label_src = seg_path
                    label_dst = os.path.join(labelsTr_dir, f'{case_id}.nii.gz')

                    if label_src.endswith('.b2nd'):
                        seg_data, seg_affine = self.load_preprocessed_data(label_src)
                        seg_nii = nib.Nifti1Image(seg_data, seg_affine)
                        nib.save(seg_nii, label_dst)
                    else:
                        import shutil
                        shutil.copy(label_src, label_dst)
                    print(f"  Saved label: {label_dst}")

                    results.append({
                        'case_id': case_id,
                        'status': 'success',
                        'feature_shape': feature_nii.shape
                    })

                except Exception as e:
                    print(f"  Error: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    results.append({
                        'case_id': case_id,
                        'status': 'error',
                        'error': str(e)
                    })

            return results

        return []


def create_dataset_json(target_dataset: str, num_training: int):
    """
    创建dataset.json配置文件
    """
    dataset_json = {
        "channel_names": {
            "0": "MRI",
            "1": "MedSAM_feature"
        },
        "labels": {
            "background": 0,
            "lesion": 1,
            "structure_2": 2
        },
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        "description": "MedSAM2 feature-enhanced MRI dataset for nnUNet segmentation"
    }

    json_path = os.path.join(target_dataset, 'dataset.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_json, f, indent=4, ensure_ascii=False)

    print(f"Created dataset.json: {json_path}")
    return json_path


if __name__ == '__main__':
    # 路径配置
    checkpoint_path = "/home/alex/Project/MRI/weights/MedSAM2_MRI_LiverLesion.pt"
    source_dataset = "/home/alex/Project/MRI/Data/Dataset001_MyCenter"
    target_dataset = "/home/alex/Project/MRI/Data/Dataset002_MedSAM_Enhanced"

    # 创建目标目录
    os.makedirs(target_dataset, exist_ok=True)

    # 初始化空间特征提取器
    # output_channels=128 适合24GB显存
    extractor = MedSAM2SpatialFeatureExtractor(
        checkpoint_path=checkpoint_path,
        device='cuda',
        image_size=(512, 512),
        output_channels=128,  # 可选 128 或 256
        use_multilevel=True
    )

    print("\n" + "="*60)
    print("MedSAM2 Spatial Feature Extraction")
    print("="*60)

    # 处理数据集
    results = extractor.process_dataset(
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        use_preprocessed=True
    )

    # 创建dataset.json
    success_count = sum(1 for r in results if r.get('status') == 'success')
    create_dataset_json(target_dataset, success_count)

    # 打印摘要
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Successfully processed: {success_count} cases")
    print(f"Output dataset: {target_dataset}")
    print(f"Feature channels: 128")

    # 保存处理日志
    log_path = os.path.join(target_dataset, 'processing_log.json')
    with open(log_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Processing log saved to: {log_path}")
