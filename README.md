# MRI: 基于 MedSAM2 的肝癌 MVI 特征工程仓库

本仓库用于肝癌多序列 MRI 的阶段 1 实验，主线是：

`MedSAM2 编码器 -> ROI 构建 -> 区域池化特征 -> 轻量 probe 分类`

当前目标不是直接做最终多模态大模型，而是先验证 `MedSAM2` 视觉特征对 `MVI` 二分类是否具备基础可分性，并把整条实验链路整理成可复用工程。

本 README 主要依据本地 `.ai_context/` 中的项目上下文整理而成，但 `.ai_context/` 仅用于本地协作辅助，不属于 GitHub 共享内容，上传时不要包含。

## 当前状态

截至 `2026-04-01`，阶段 1 主线已经跑通。

- 数据基础：`center1` 审计后，当前可直接用于 `MVI` 二分类的病例数为 `220`
- 默认 6 序列映射：
  - `0000 -> T1`
  - `0001 -> T2`
  - `0002 -> DWI`
  - `0003 -> AP`
  - `0004 -> PVP`
  - `0005 -> HBP`
- 当前实现已确认 NIfTI 切片维为最后一维，统一按 `volume[:, :, z]` 和 `mask[:, :, z]` 处理
- 当前阶段 1 最优结果：
  - `SVM + l2 + tumor + concat_sequences`
  - `split_v1` 上 `test AUC = 0.7595`
  - `val AUC = 0.5927`

详细实验结论见 `work-report/stage1_mvi_feature_probe_report_2026-04-01.md`。

## 仓库结构

```text
MRI/
├── README.md
├── configs/
│   ├── data/                  # 数据入口、split、ROI 配置
│   ├── encoder/               # MedSAM2 编码器配置
│   ├── feature/               # 特征存储配置
│   └── probe/                 # Logistic / SVM / MLP 配置
├── src/
│   ├── data/                  # manifest 构建、数据划分、ROI 生成
│   ├── encoders/              # 编码器抽象与 MedSAM2 集成
│   ├── features/              # 特征抽取、池化、索引管理
│   ├── probes/                # 轻量 probe 训练入口
│   ├── eval/                  # 指标计算与结果写出
│   └── utils/                 # 配置、日志、IO、随机种子
├── MedSAM2/                   # 集成的上游 MedSAM2 代码
├── work-report/               # 阶段性工作报告
├── outputs/                   # 实验结果汇总
├── Data/                      # 本地数据与中间产物，默认不上传
├── weights/                   # 本地模型权重，默认不上传
├── nnUNet/                    # 本地 nnUNet 相关资源，默认不上传
├── .ai_context/               # 本地 AI 协作上下文，默认不上传
├── run_extract_v2.py          # 早期探索脚本，非当前主线
└── visualize_features.py      # 早期特征可视化脚本，非当前主线
```

### `src/` 子模块说明

- `src/data/build_manifest.py`
  - 从审计结果生成统一训练入口
  - 输出 `canonical_manifest.csv`、`binary_ready.csv`、`excluded_cases.csv`
- `src/data/make_splits.py`
  - 基于 `binary_ready.csv` 生成固定训练/验证/测试划分和 5-fold 划分
- `src/data/roi_builder.py`
  - 从分割掩膜构建 `tumor`、`peritumor_3mm`、`peritumor_5mm`
- `src/encoders/medsam2_encoder.py`
  - 统一封装 MedSAM2 中间层特征提取接口
- `src/features/extract_region_features.py`
  - 按病例、序列、层级、ROI 抽取区域池化特征并建立索引
- `src/features/rebuild_feature_index.py`
  - 从磁盘特征文件重建 `feature_index.csv`
- `src/probes/train_probe.py`
  - 用统一入口训练 `Logistic Regression`、`Linear SVM`、`MLP`
- `src/eval/`
  - 负责指标计算和 `results.csv` / `results.md` 写出

## 当前主线流程

阶段 1 的标准顺序如下：

1. 数据审计结果作为唯一可信入口
2. 生成标准化 manifest
3. 生成固定 split
4. 基于 segmentation 构建 ROI
5. 用 MedSAM2 提取中间层特征
6. 在 ROI 内做 pooling，生成区域级特征
7. 用 probe 训练并输出结果表

对应脚本如下：

```bash
python src/data/build_manifest.py --config configs/data/center1.yaml
python src/data/make_splits.py --data-config configs/data/center1.yaml --split-config configs/data/splits.yaml
python src/data/roi_builder.py --manifest Data/manifests/binary_ready.csv --roi-config configs/data/roi.yaml
python src/features/extract_region_features.py \
  --manifest Data/manifests/binary_ready.csv \
  --roi-metadata Data/processed/roi_metadata.jsonl \
  --encoder-config configs/encoder/medsam2.yaml \
  --feature-config configs/feature/feature_store.yaml \
  --max-slices 8
python src/probes/train_probe.py \
  --probe-config configs/probe/svm.yaml \
  --feature-index Data/features/stage1/feature_index.csv \
  --manifest Data/manifests/binary_ready.csv \
  --split-file Data/splits/split_v1.json \
  --layer l2 \
  --roi-type tumor \
  --sequence-mode concat_sequences
```

如果要切换 probe，只需要替换 `configs/probe/*.yaml`。如果要切换 ROI、层级或多序列聚合方式，直接改命令参数即可。

## 配置与路径约定

当前 YAML 配置大量使用绝对路径，例如：

- `configs/data/center1.yaml`
- `configs/encoder/medsam2.yaml`
- `configs/feature/feature_store.yaml`

如果组员将仓库克隆到不同路径，首先需要同步修改这些配置文件中的本地路径。

当前开发环境记录如下：

- `Ubuntu 22.04`
- `Python 3.12`
- `PyTorch + CUDA`
- `MedSAM2` 作为冻结视觉编码器

主线代码额外依赖主要包括：

- `torch`
- `numpy`
- `pandas`
- `scikit-learn`
- `nibabel`
- `scipy`
- `PyYAML`

`MedSAM2/requirements.txt` 中还列出了其自身依赖。如果组员要重新配置环境，建议先安装该文件中的依赖，再补充本项目主线使用的包。

## 关键输入与输出

### 本地数据入口

当前阶段 1 以 `Data/center1_audit/` 为标准入口，核心文件包括：

- `case_manifest.csv`
- `labels_clean.csv`
- `center1_data_summary.md`

### 关键中间产物

- `Data/manifests/canonical_manifest.csv`
- `Data/manifests/binary_ready.csv`
- `Data/manifests/excluded_cases.csv`
- `Data/splits/split_v1.json`
- `Data/splits/cv5_v1.json`
- `Data/processed/roi_metadata.jsonl`
- `Data/processed/roi_masks/`
- `Data/features/stage1/feature_index.csv`

### 结果文件

- `outputs/reports/results.csv`
- `outputs/reports/results.md`
- `work-report/stage1_mvi_feature_probe_report_2026-04-01.md`

## 共享给组员时的建议

建议上传到 GitHub 的内容：

- `README.md`
- `configs/`
- `src/`
- `MedSAM2/`
- `work-report/`
- `outputs/reports/`（如果希望保留当前实验记录）

不要上传的内容：

- `.ai_context/`
- `Data/`
- `weights/`
- `nnUNet/`
- 原始影像、分割、临床 CSV、模型权重等大文件或敏感数据

当前 `.gitignore` 已忽略 `Data/`、`weights/`、`nnUNet/` 和 `.ai_context/`，因此正常提交代码时，这些内容默认不会进入 Git。

## 备注

- `MedSAM2/` 主要作为上游分割与编码器依赖集成，阶段 1 默认不在其中做大规模结构改造
- 根目录的 `run_extract_v2.py` 与 `visualize_features.py` 属于早期探索脚本，不是当前阶段 1 的正式主线
- 如果后续进入阶段 2，预计会在当前视觉基线之上继续扩展多序列融合、临床变量联合建模和更稳的交叉验证
