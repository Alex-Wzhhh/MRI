# MedSAM特征融合nnUNet分割方案 - 工作报告

## 任务目标

将MedSAM2提取的空间特征图作为额外输入通道送入nnUNet，实现特征增强的医学图像分割。

## 背景

- 原有方案：提取256维全局向量（经全局平均池化），丢失空间信息
- 新方案：保留空间特征图，多层特征融合，作为nnUNet的第二输入通道

## 文件路径清单

### 新建文件

| 文件路径 | 说明 |
|---------|------|
| `/home/alex/Project/MRI/特征提取_空间版.py` | 空间特征提取脚本 |
| `/home/alex/Project/MRI/Data/Dataset002_MedSAM_Enhanced/dataset.json` | 数据集配置文件 |
| `/home/alex/Project/MRI/work-report/medsam_feature_integration.md` | 本工作报告 |

### 新建目录

| 目录路径 | 说明 |
|---------|------|
| `/home/alex/Project/MRI/Data/Dataset002_MedSAM_Enhanced/` | 特征增强数据集根目录 |
| `/home/alex/Project/MRI/Data/Dataset002_MedSAM_Enhanced/imagesTr/` | 训练图像目录 |
| `/home/alex/Project/MRI/Data/Dataset002_MedSAM_Enhanced/labelsTr/` | 训练标签目录 |

### 源文件（保持不变）

| 文件路径 | 说明 |
|---------|------|
| `/home/alex/Project/MRI/Data/Dataset001_MyCenter/` | 原始数据集 |
| `/home/alex/Project/MRI/weights/MedSAM2_MRI_LiverLesion.pt` | MedSAM2预训练权重 |

## 技术方案

### MedSAM2编码器输出结构

```
输入图像: (1, 3, 512, 512)
        ↓
image_encoder
        ↓
backbone_fpn: 多层特征金字塔
  - Level 0: (1, 256, 128, 128) stride=4  高分辨率
  - Level 1: (1, 256, 64, 64)   stride=8  中分辨率
  - Level 2: (1, 256, 32, 32)   stride=16 低分辨率
```

### 特征融合策略

1. **多层融合**：Level 0 + Level 1（上采样后相加）
2. **通道压缩**：256通道 → 128通道（1×1卷积）
3. **上采样**：恢复到原始图像分辨率

### 输出格式

```
Dataset002_MedSAM_Enhanced/
├── imagesTr/
│   ├── case000_0000.nii.gz   # 原始MRI (D, H, W)
│   ├── case000_0001.nii.gz   # MedSAM特征 (H, W, D, 128)
│   └── ...
├── labelsTr/
│   ├── case000.nii.gz        # 分割标签
│   └── ...
└── dataset.json
```

## 执行步骤

### 步骤1: 运行空间特征提取

```bash
cd /home/alex/Project/MRI
python 特征提取_空间版.py
```

预计输出：
- 处理125个病例
- 每个病例生成两个文件：_0000.nii.gz（原始）和 _0001.nii.gz（特征）

### 步骤2: nnUNet预处理

```bash
# 设置环境变量
export nnUNet_raw="/home/alex/Project/MRI/Data"
export nnUNet_preprocessed="/home/alex/Project/MRI/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/home/alex/Project/MRI/nnUNet/nnUNet_results"

# 生成数据集指纹
nnUNetv2_plan_experiment -d 002

# 预处理
nnUNetv2_preprocess -d 002 -c 3d_fullres
```

### 步骤3: 训练模型

```bash
nnUNetv2_train 002 3d_fullres 0
```

## 验证方法

1. **特征图形状检查**：
   ```python
   import nibabel as nib
   feat = nib.load('case000_0001.nii.gz')
   print(feat.shape)  # 应为 (H, W, D, 128)
   ```

2. **可视化特征图**：
   ```python
   import matplotlib.pyplot as plt
   data = feat.get_fdata()
   plt.imshow(data[:, :, 50, 0])  # 显示第50层第0通道
   ```

3. **nnUNet通道识别**：
   - 预处理日志应显示2个输入通道
   - dataset.json中channel_names应包含"MedSAM_feature"

## 显存使用估算

| 组件 | 显存占用 |
|------|---------|
| MedSAM2特征提取（冻结权重） | ~4GB |
| nnUNet 3D训练 (128通道特征) | ~10-12GB |
| nnUNet 3D训练 (256通道特征) | ~16-18GB |

**推荐配置**：128通道特征，适配24GB显存

## 对比实验设计

| 实验组 | 输入通道 | 说明 |
|-------|---------|------|
| Baseline | 仅MRI | 原始nnUNet |
| Enhanced | MRI + MedSAM特征 | 特征增强nnUNet |

评估指标：
- Dice系数
- Hausdorff距离
- 分割边界质量

## 状态记录

| 日期 | 操作 | 状态 |
|------|------|------|
| 2026-02-24 | 创建空间特征提取脚本 | ✅ 完成 |
| 2026-02-24 | 创建Dataset002目录结构 | ✅ 完成 |
| 2026-02-24 | 创建dataset.json配置 | ✅ 完成 |
| - | 运行特征提取 | ⏳ 待执行 |
| - | nnUNet预处理 | ⏳ 待执行 |
| - | 模型训练 | ⏳ 待执行 |
| - | 对比实验 | ⏳ 待执行 |

## 注意事项

1. 特征提取前确保CUDA可用
2. 特征文件较大（每个case约500MB-1GB），确保磁盘空间充足
3. 首次运行建议先用少量case测试流程
4. 如显存不足，可将output_channels改为64
