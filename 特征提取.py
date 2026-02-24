import sys
sys.path.append('/home/alex/Project/MRI/MedSAM2')

import torch
import numpy as np
import blosc2
import nibabel as nib
from typing import List, Dict, Tuple, Optional
import os
from skimage.transform import resize

# 手动初始化Hydra配置
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate


def build_sam2_model(config_file: str, ckpt_path: str, device: str = 'cuda'):
    """
    构建SAM2模型（手动处理hydra初始化）
    """
    # 清除之前的初始化
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    # 使用绝对路径初始化
    config_dir = '/home/alex/Project/MRI/MedSAM2/sam2/configs'

    initialize_config_dir(config_dir=config_dir, version_base='1.2')
    cfg = compose(config_name=config_file)

    # 实例化模型
    model = instantiate(cfg.model, _recursive_=True)

    # 加载权重
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


class MedSAM2FeatureExtractor:
    """
    MedSAM2 特征提取器
    用于从预处理后的 MRI 数据中提取高维特征向量
    支持基于分割mask的区域特征提取
    """

    def __init__(self,
                 checkpoint_path: str,
                 device: str = 'cuda',
                 image_size: Tuple[int, int] = (512, 512)):
        """
        初始化特征提取器

        Args:
            checkpoint_path: MedSAM2 预训练权重路径
            device: 计算设备 ('cuda' 或 'cpu')
            image_size: 输入图像尺寸
        """
        self.device = device
        self.image_size = image_size

        # 加载模型
        print(f"Loading MedSAM2 from {checkpoint_path}...")
        config_file = "sam2.1_hiera_t512.yaml"
        self.model = build_sam2_model(config_file=config_file, ckpt_path=checkpoint_path, device=device)

        # 冻结权重
        for param in self.model.parameters():
            param.requires_grad = False

        print("MedSAM2 loaded successfully!")

    def load_preprocessed_data(self, data_path: str) -> np.ndarray:
        """
        加载预处理后的图像数据 (.b2nd 或 .nii.gz)

        Args:
            data_path: 数据文件路径

        Returns:
            3D numpy数组 (D, H, W) 或 (H, W, D)
        """
        if data_path.endswith('.b2nd'):
            arr = blosc2.open(data_path)
            data = arr[()]  # 转换为numpy数组
            # b2nd格式: (1, D, H, W) -> (D, H, W)
            if data.ndim == 4 and data.shape[0] == 1:
                data = data.squeeze(0)
        elif data_path.endswith('.nii.gz') or data_path.endswith('.nii'):
            nii = nib.load(data_path)
            data = nii.get_fdata()
        else:
            raise ValueError(f"Unsupported file format: {data_path}")

        return data

    def load_segmentation(self, seg_path: str) -> np.ndarray:
        """
        加载分割标注

        Args:
            seg_path: 分割文件路径

        Returns:
            3D numpy数组，标签值: 0=背景, 1=病变, 2=其他结构
        """
        return self.load_preprocessed_data(seg_path)

    def preprocess_slice(self,
                         slice_data: np.ndarray,
                         mask: Optional[np.ndarray] = None) -> torch.Tensor:
        """
        预处理单个切片

        Args:
            slice_data: 2D切片数据
            mask: 可选的分割mask，用于区域提取

        Returns:
            预处理后的张量 (1, 3, H, W)
        """
        # 如果提供了mask，可以提取区域特征（这里先做全局特征提取）
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

    def extract_feature_from_slice(self,
                                   slice_data: np.ndarray,
                                   mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        从单个2D切片提取特征

        Args:
            slice_data: 2D切片数据 (H, W)
            mask: 可选的分割mask

        Returns:
            特征向量
        """
        input_tensor = self.preprocess_slice(slice_data, mask)

        with torch.no_grad():
            encoder_output = self.model.image_encoder(input_tensor)

        # image_encoder返回字典，取vision_features
        if isinstance(encoder_output, dict):
            image_embedding = encoder_output['vision_features']
        else:
            image_embedding = encoder_output

        # 全局平均池化: (1, C, H, W) -> (1, C)
        feature_vector = torch.mean(image_embedding, dim=[2, 3])

        return feature_vector.cpu().numpy().squeeze(0)

    def extract_feature_from_volume(self,
                                    volume: np.ndarray,
                                    segmentation: Optional[np.ndarray] = None,
                                    label_of_interest: int = 1,
                                    strategy: str = 'masked_mean',
                                    max_slices: int = 50) -> np.ndarray:
        """
        从3D体数据提取特征

        Args:
            volume: 3D体数据 (D, H, W) 或 (H, W, D)
            segmentation: 分割标注
            label_of_interest: 感兴趣的标签 (默认1=病变)
            strategy: 特征融合策略
                - 'mean': 所有切片平均
                - 'masked_mean': 只平均包含目标区域的切片
                - 'max': 取最大特征
            max_slices: 最大处理切片数（防止内存溢出）

        Returns:
            特征向量
        """
        # 统一维度顺序为 (D, H, W)
        if volume.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape {volume.shape}")

        # 确定切片轴（假设最后一维是深度或第一维是深度）
        # 根据nnUNet预处理数据，格式为 (D, H, W)
        slice_features = []
        valid_slice_indices = []

        # 找到包含目标区域的切片
        if segmentation is not None and strategy == 'masked_mean':
            for d in range(volume.shape[0]):
                mask_slice = segmentation[d] if segmentation.shape[0] == volume.shape[0] else segmentation[:, :, d]
                if np.any(mask_slice == label_of_interest):
                    valid_slice_indices.append(d)
        else:
            valid_slice_indices = list(range(volume.shape[0]))

        # 限制切片数量
        if len(valid_slice_indices) > max_slices:
            # 均匀采样
            step = len(valid_slice_indices) // max_slices
            valid_slice_indices = valid_slice_indices[::step][:max_slices]

        # 提取特征
        for d in valid_slice_indices:
            slice_data = volume[d] if volume.shape[0] == segmentation.shape[0] else volume[:, :, d]

            # 跳过空白切片
            if np.abs(slice_data).max() < 1e-6:
                continue

            # 获取对应的mask
            mask = None
            if segmentation is not None:
                mask = segmentation[d] if segmentation.shape[0] == volume.shape[0] else segmentation[:, :, d]

            feature = self.extract_feature_from_slice(slice_data, mask)
            slice_features.append(feature)

        if len(slice_features) == 0:
            # 如果没有有效切片，返回零向量
            print("  Warning: No valid slices found, returning zero vector")
            return np.zeros(256)  # SAM2 tiny的embedding维度

        slice_features = np.array(slice_features)

        # 特征融合
        if strategy in ['mean', 'masked_mean']:
            volume_feature = np.mean(slice_features, axis=0)
        elif strategy == 'max':
            volume_feature = np.max(slice_features, axis=0)
        else:
            volume_feature = np.mean(slice_features, axis=0)

        return volume_feature

    def save_features(self,
                      features: np.ndarray,
                      save_path: str,
                      metadata: Dict = None):
        """保存提取的特征"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        save_dict = {
            'features': features,
            'metadata': metadata
        }

        np.save(save_path, save_dict)
        print(f"Features saved to {save_path}")

    def batch_extract_from_nnunet_dataset(self,
                                          dataset_root: str,
                                          save_root: str,
                                          use_preprocessed: bool = True):
        """
        从nnUNet格式数据集批量提取特征

        Args:
            dataset_root: nnUNet数据集根目录
            save_root: 特征保存目录
            use_preprocessed: 是否使用预处理后的数据
        """
        results = []

        if use_preprocessed:
            # 使用预处理后的数据
            preproc_dir = os.path.join(dataset_root, 'nnUNetPlans_3d_fullres')
            seg_dir = os.path.join(dataset_root, 'gt_segmentations')

            if not os.path.exists(preproc_dir):
                raise FileNotFoundError(f"Preprocessed data not found: {preproc_dir}")

            # 获取所有病例ID
            case_files = [f for f in os.listdir(preproc_dir) if f.endswith('.b2nd') and not f.endswith('_seg.b2nd')]
            case_ids = sorted(list(set([f.replace('.b2nd', '') for f in case_files])))

            print(f"Found {len(case_ids)} cases in {preproc_dir}")

            for i, case_id in enumerate(case_ids):
                print(f"\n[{i+1}/{len(case_ids)}] Processing {case_id}...")

                try:
                    # 加载图像
                    img_path = os.path.join(preproc_dir, f'{case_id}.b2nd')
                    seg_path = os.path.join(preproc_dir, f'{case_id}_seg.b2nd')

                    # 如果预处理目录没有分割，尝试原始分割目录
                    if not os.path.exists(seg_path):
                        seg_path = os.path.join(seg_dir, f'{case_id}.nii.gz')

                    if not os.path.exists(img_path):
                        print(f"  Image not found: {img_path}")
                        continue

                    # 加载数据
                    volume = self.load_preprocessed_data(img_path)
                    segmentation = None
                    if os.path.exists(seg_path):
                        segmentation = self.load_segmentation(seg_path)

                    print(f"  Volume shape: {volume.shape}")
                    if segmentation is not None:
                        print(f"  Segmentation shape: {segmentation.shape}")

                    # 提取特征
                    feature = self.extract_feature_from_volume(
                        volume,
                        segmentation,
                        label_of_interest=1,  # 病变区域
                        strategy='masked_mean'
                    )

                    # 保存
                    save_path = os.path.join(save_root, f'{case_id}_features.npy')
                    self.save_features(feature, save_path,
                                       metadata={'case_id': case_id,
                                                'feature_dim': feature.shape[0],
                                                'volume_shape': volume.shape})

                    results.append({
                        'case_id': case_id,
                        'feature_path': save_path,
                        'feature_dim': feature.shape[0]
                    })

                except Exception as e:
                    print(f"  Error processing {case_id}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue

        return results


if __name__ == '__main__':
    # 路径配置
    checkpoint_path = "/home/alex/Project/MRI/weights/MedSAM2_MRI_LiverLesion.pt"
    dataset_root = "/home/alex/Project/MRI/Data/Dataset001_MyCenter"
    save_root = "/home/alex/Project/MRI/Data/features/"

    # 创建保存目录
    os.makedirs(save_root, exist_ok=True)

    # 初始化提取器
    extractor = MedSAM2FeatureExtractor(
        checkpoint_path=checkpoint_path,
        device='cuda'
    )

    # 批量提取特征
    print("\n" + "="*50)
    print("Starting Batch Feature Extraction")
    print("="*50)

    results = extractor.batch_extract_from_nnunet_dataset(
        dataset_root=dataset_root,
        save_root=save_root,
        use_preprocessed=True
    )

    print("\n" + "="*50)
    print("Batch Feature Extraction Summary")
    print("="*50)
    print(f"Successfully processed: {len(results)} cases")
    if results:
        print(f"Features saved in: {save_root}")
        print(f"Example: {results[0]}")
