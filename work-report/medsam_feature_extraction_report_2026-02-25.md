# MedSAM2空间特征提取任务报告

**日期**: 2026-02-25
**任务状态**: 部分完成（预处理中断）

---

## 一、任务概述

### 目标
从MedSAM2编码器提取空间特征图（非全局向量），作为额外输入通道送入nnUNet进行医学图像分割。

### 技术方案
- 原方案：提取256维全局向量（经全局平均池化）→ 丢失空间信息
- 新方案：保留空间特征图，多层特征融合，作为nnUNet第二输入通道

---

## 二、执行流程

### 2.1 文件创建

| 文件路径 | 说明 | 状态 |
|---------|------|------|
| `/home/alex/Project/MRI/特征提取_空间版.py` | 空间特征提取脚本 | 已创建 |
| `/home/alex/Project/MRI/Data/Dataset002_MedSAM_Enhanced/` | 特征增强数据集目录 | 已创建 |
| `/home/alex/Project/MRI/Data/Dataset002_MedSAM_Enhanced/dataset.json` | 数据集配置文件 | 已创建 |

### 2.2 特征提取流程

```
MedSAM2编码器
     ↓
backbone_fpn (多层特征金字塔)
  - Level 0: (1, 256, 128, 128) stride=4  高分辨率
  - Level 1: (1, 256, 64, 64)   stride=8  中分辨率
     ↓
特征融合 (Level 0 + Level 1 上采样后相加)
     ↓
通道压缩 (256 → 128 通道, 1x1卷积)
     ↓
上采样到原始图像分辨率
     ↓
通道平均 (128 → 1 通道)
     ↓
保存为NIfTI格式
```

---

## 三、遇到的问题及解决方案

### 3.1 问题一：nnUNet多通道格式不兼容

**错误描述**：
```
ERROR! Not all input images have the same shape!
Shapes: [(1, 502, 467, 85), (128, 502, 467, 85)]
```

**原因分析**：
- 初始方案输出4D文件 `(D, H, W, 128)` 包含128个特征通道
- nnUNet将4D文件解读为多个独立的3D体积，而非一个带特征的体积
- nnUNet要求每个输入通道必须是独立的3D体积文件

**解决方案**：
修改 `特征提取_空间版.py` 第296-300行，将128通道压缩为单通道：

```python
# 对每个体素位置，取128通道的平均作为单通道输出
feature_single_channel = np.mean(feature_volume_nifti, axis=-1)  # (D, H, W)

nii_img = nib.Nifti1Image(feature_single_channel, affine)
```

### 3.2 问题二：Unicode显示乱码

**错误描述**：tqdm进度条显示乱码字符

**解决方案**：
在tqdm调用中添加 `ascii=True` 参数：
```python
tqdm(slice_indices, desc="Extracting spatial features", ascii=True)
```

### 3.3 问题三：RAM不足

**解决方案**：
在nnUNet命令中添加 `-np 1` 参数，限制进程数为1

---

## 四、执行结果

### 4.1 特征提取结果

| 指标 | 数值 |
|------|------|
| 处理病例数 | 125 |
| 成功处理 | 125 |
| 失败数 | 0 |
| 输出通道数 | 128 → 1 (压缩后) |

### 4.2 输出文件格式验证

```
MRI文件形状:     (85, 467, 502) - 3D体积 ✓
特征文件形状:    (85, 467, 502) - 3D体积 ✓
形状匹配:        是 ✓
```

### 4.3 nnUNet预处理进度

| 步骤 | 状态 |
|------|------|
| 数据集指纹提取 | 完成 |
| 实验计划生成 | 完成 |
| 3d_fullres预处理 | 中断 (5/125) |

---

## 五、目录结构

```
/home/alex/Project/MRI/Data/Dataset002_MedSAM_Enhanced/
├── dataset.json                           # 数据集配置
├── processing_log.json                    # 处理日志
├── imagesTr/                              # 训练图像
│   ├── case000_0000.nii.gz               # 原始MRI
│   ├── case000_0001.nii.gz               # MedSAM特征
│   ├── case001_0000.nii.gz
│   ├── case001_0001.nii.gz
│   └── ... (共250个文件: 125×2)
└── labelsTr/                              # 训练标签
    ├── case000.nii.gz
    ├── case001.nii.gz
    └── ... (共125个文件)
```

---

## 六、后续步骤

### 继续预处理
```bash
source /home/alex/miniconda3/etc/profile.d/conda.sh && conda activate env1
export nnUNet_raw="/home/alex/Project/MRI/Data"
export nnUNet_preprocessed="/home/alex/Project/MRI/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/home/alex/Project/MRI/nnUNet/nnUNet_results"

nnUNetv2_preprocess -d 002 -c 3d_fullres -np 1
```

### 训练模型
```bash
nnUNetv2_train 002 3d_fullres 0
```

---

## 七、技术要点总结

1. **特征保留策略**：使用多层特征融合（Level 0 + Level 1），保留更多空间细节
2. **通道压缩**：128通道通过平均压缩为单通道，满足nnUNet多输入要求
3. **文件格式**：每个病例生成两个文件（_0000.nii.gz原始，_0001.nii.gz特征）
4. **显存优化**：冻结MedSAM2权重，仅前向传播，显存占用约4GB

---

## 八、相关文件

| 文件 | 路径 |
|------|------|
| 特征提取脚本 | `/home/alex/Project/MRI/特征提取_空间版.py` |
| 数据集配置 | `/home/alex/Project/MRI/Data/Dataset002_MedSAM_Enhanced/dataset.json` |
| 处理日志 | `/home/alex/Project/MRI/Data/Dataset002_MedSAM_Enhanced/processing_log.json` |
| nnUNet计划 | `/home/alex/Project/MRI/nnUNet/nnUNet_preprocessed/Dataset002_MedSAM_Enhanced/nnUNetPlans.json` |
