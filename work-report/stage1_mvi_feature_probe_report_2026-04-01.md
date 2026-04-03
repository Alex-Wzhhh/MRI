# MRI 项目阶段 1 正式工作报告

**报告日期**: 2026-04-01  
**项目名称**: 肝癌 MVI 多模态诊断系统  
**阶段名称**: 阶段 1 `MedSAM2 -> ROI -> region pooled feature -> probe` 验证闭环  
**报告状态**: 已完成

## 一、报告摘要

本阶段工作的核心目标，是在不直接进入复杂多模态融合与大模型改造之前，先验证 `MedSAM2` 提取的视觉特征是否对 `MVI` 二分类具有基础可分性，并建立一套可以复用的、结构清晰的阶段 1 工程闭环。

本轮工作已经完成以下事项：

- 完成新版 `center1` 数据的审计、清洗与标准化入口构建。
- 建立阶段 1 工程骨架，包括配置、数据、编码器、特征、probe、评估等模块。
- 构建 `manifest`、`split`、`ROI`、`feature index`、实验结果汇总等关键数据资产。
- 基于 `MedSAM2` 冻结编码器，完成 `tumor / peritumor` 区域的 `region pooled feature` 提取。
- 跑通 `Logistic Regression`、`Linear SVM`、`MLP` 三类 probe，并完成 layer、ROI、sequence aggregation 的首轮实验矩阵。
- 发现并修正了 NIfTI 切片维处理错误，按真实轴顺序重跑关键实验。

轴顺序修正后的当前最佳结果为：

- `SVM + l2 + tumor + concat_sequences`
- `test AUC = 0.7595`
- `val AUC = 0.5927`

这一结果说明，多序列聚合后的 `MedSAM2` 区域级特征已经具备一定的 MVI 区分能力，可以作为下一阶段视觉主基线继续扩展。

## 二、工作背景与目标

项目当前的中长期目标，是构建面向肝癌 `MVI` 的多模态智能诊断系统，输入包括多序列 MRI、临床信息以及后续可能引入的随访信息，输出包括 `MVI` 风险预测和可解释性结论。

但在正式进入复杂融合之前，必须先回答一个更基础的问题：`MedSAM2` 编码器提取出的视觉特征，是否本身就携带了足以支持 `MVI` 二分类的有效信息。

因此，阶段 1 的具体目标被收缩为两项：

- 建立稳定的 `MedSAM2 -> ROI -> region pooled feature` 特征管线。
- 用轻量 probe 快速验证视觉特征的可分性，并比较 layer、ROI、序列聚合方式之间的差异。

这一阶段不追求最终最强性能，重点是验证路径是否可行、工程结构是否可复用、实验结论是否能支撑下一阶段设计。

## 三、数据基础与数据口径

本阶段仅围绕新版 `center1` 数据展开，不混入其他来源，以避免样本口径和标签口径进一步混乱。

本轮工作确认的数据口径如下：

- 原始影像目录：`/mnt/d/BaiduNetdiskDownload/center1/center1`
- 分割标签目录：`/mnt/d/BaiduNetdiskDownload/center1/center1_label/center1_label`
- 临床与标签文件：`/mnt/d/BaiduNetdiskDownload/center1/labels.csv`

数据审计后的主要统计结论如下：

- 完整 6 序列 MRI 影像病例：267 例
- 分割标签文件：272 个
- `CSV` 标签病例：267 例
- 可直接用于 `MVI` 二分类的病例：220 例
- 可作为三分类候选的病例：242 例

已确认的序列映射如下：

- `0000 -> T1`
- `0001 -> T2`
- `0002 -> DWI`
- `0003 -> AP`
- `0004 -> PVP`
- `0005 -> HBP`

已确认的 NIfTI 轴顺序如下：

- 对于典型 shape 如 `[512, 512, 72]`，最后一维 `72` 为切片维，即 `z / D`
- 因此切片抽取必须使用 `volume[:, :, z]`

## 四、主要工作内容

### 4.1 数据审计与标准化入口建设

本轮首先完成了新版 `center1` 数据的对齐、清洗与统一入口建设，主要包括：

- 生成逐病例可用性清单，明确每个病例是否同时具备影像、分割、标签。
- 清洗 `MVI` 标签，区分二分类可用样本与三分类候选样本。
- 输出统一训练入口，避免后续脚本各自扫目录、临时拼数据。

本部分产出的关键文件包括：

- `Data/center1_audit/center1_data_summary.md`
- `Data/center1_audit/case_manifest.csv`
- `Data/center1_audit/labels_clean.csv`
- `Data/manifests/canonical_manifest.csv`
- `Data/manifests/binary_ready.csv`
- `Data/manifests/excluded_cases.csv`

### 4.2 阶段 1 工程骨架搭建

为了避免继续堆积一次性脚本，本轮围绕阶段 1 需求，建立了贴当前仓库的最小工程结构，包含：

- `configs/`
- `src/data/`
- `src/encoders/`
- `src/features/`
- `src/probes/`
- `src/eval/`
- `outputs/reports/`

这样做的目的，是把“数据入口”“编码器抽象”“特征提取”“probe 训练”“结果记录”明确拆开，保证后续可以继续加 ROI、加序列策略、加临床融合，而不需要重写整个流程。

### 4.3 ROI 构建与区域特征提取

本轮采用 segmentation mask 构建阶段 1 所需的三个 ROI：

- `tumor`
- `peritumor_3mm`
- `peritumor_5mm`

随后基于 `MedSAM2` 冻结编码器抽取中间层特征，并在 ROI 内执行 masked average pooling，生成固定长度的区域级特征向量。

本阶段没有直接囤积大规模 dense feature 资产，而是优先产出更适合 probe 消费的 `region pooled feature`。这一取舍是有意的，目的是用更低成本尽快验证视觉特征的可分性。

### 4.4 轻量 probe 训练与实验矩阵

本轮跑通了三类 probe：

- `Logistic Regression`
- `Linear SVM`
- `MLP`

在此基础上，完成了三类实验比较：

- ROI 对比：`tumor`、`peritumor_3mm`、`peritumor_5mm`
- layer 对比：`l1`、`l2`、`l3`
- sequence aggregation 对比：`single_sequence`、`mean_over_sequences`、`concat_sequences`

实验结果统一写入：

- `outputs/reports/results.csv`
- `outputs/reports/results.md`

### 4.5 关键问题排查与修复

本轮最重要的技术修复，是发现并修正了 NIfTI 切片维处理错误。

最初实现中，切片曾被误按第 0 维提取。这会导致所谓“肿瘤切片”与 ROI pooling 实际上发生在错误平面上，从而使实验结果失真。后续经你确认，真实切片维为最后一维，因此实现被统一修正为：

- `volume[:, :, z]`
- `mask[:, :, z]`

在此基础上，重新生成 ROI、重新提取特征，并重跑关键实验。修正后，多序列聚合结果明显优于之前的单序列结果，说明这一修复是必要且有效的。

## 五、主要产出文件

本轮工作的主要代码与数据产物如下。

代码部分：

- `src/data/build_manifest.py`
- `src/data/make_splits.py`
- `src/data/roi_builder.py`
- `src/encoders/medsam2_encoder.py`
- `src/features/extract_region_features.py`
- `src/features/rebuild_feature_index.py`
- `src/probes/train_probe.py`

数据与结果部分：

- `Data/manifests/canonical_manifest.csv`
- `Data/manifests/binary_ready.csv`
- `Data/splits/split_v1.json`
- `Data/splits/cv5_v1.json`
- `Data/processed/roi_metadata.jsonl`
- `Data/processed/roi_masks/`
- `Data/features/stage1/feature_index.csv`
- `outputs/reports/results.csv`
- `outputs/reports/results.md`

上下文与记录部分：

- `.ai_context/03_ENVIRONMENT.md`
- `.ai_context/05_CURRENT_STATUS.md`
- `.ai_context/02_DYNAMIC_MAP.md`

## 六、实验结果总结

### 6.1 单序列基线结果

以 `T1` 单序列为基础，当前较主要的结果如下：

- `SVM + l2 + tumor + single_sequence(T1)`，`test AUC = 0.5405`
- `Logistic + l2 + tumor + single_sequence(T1)`，`test AUC = 0.5643`
- `MLP + l2 + tumor + single_sequence(T1)`，表现较差，并出现偏向多数类的问题

这说明如果只用单序列、单 ROI、轻量特征压缩，视觉信息的可分性存在，但强度有限。

### 6.2 ROI 对比结果

ROI 对比中，`peritumor_3mm` 在测试集上给出了更高分数：

- `SVM + l2 + peritumor_3mm + single_sequence(T1)`，`test AUC = 0.6238`
- `SVM + l2 + peritumor_5mm + single_sequence(T1)`，`test AUC = 0.5476`

但需要强调的是，`peritumor_3mm` 在验证集上表现明显不稳定，因此目前不能直接下结论说它一定优于 `tumor` ROI，只能说明其值得在后续阶段继续验证。

### 6.3 Layer 对比结果

layer 对比显示：

- `l1` 的 test AUC 为 `0.4833`
- `l3` 的 test AUC 为 `0.4762`
- `l2` 在当前实验中整体更平衡

其中 `l3` 在训练集上表现很高，但验证和测试显著回落，呈现出明显过拟合特征。因此当前默认层应保持为 `l2`。

### 6.4 多序列聚合结果

这是本轮最重要的实验结论。

在轴顺序修正后，多序列聚合明显优于单序列输入：

- `SVM + l2 + tumor + mean_over_sequences`，`test AUC = 0.6929`
- `SVM + l2 + tumor + concat_sequences`，`test AUC = 0.7595`

其中最优组合为：

- `SVM + l2 + tumor + concat_sequences`
- `val AUC = 0.5927`
- `test AUC = 0.7595`
- `accuracy = 0.6818`
- `sensitivity = 0.6429`
- `specificity = 0.7000`

这一结果说明，多序列信息在当前任务中明显有增益，且直接拼接不同序列的区域特征，比简单平均更有效。

## 七、结论与判断

基于本轮工作，可以得出以下结论。

第一，`MedSAM2` 冻结编码器提取的区域级特征，对 `MVI` 二分类已经表现出可用的基础可分性，不是完全无效信号。

第二，当前阶段最合理的视觉主基线，不再是单序列单区域，而应当是：

- `concat_sequences + l2 + tumor + SVM`

第三，`peritumor_3mm` 在测试集上表现值得关注，但由于验证集不稳定，暂时不宜把它直接提升为默认主路线。

第四，`l3` 特征在当前小样本设置下过拟合明显，不应作为下一阶段默认特征层。

第五，当前结果已经足以支撑下一阶段工作，即在稳定视觉基线之上进一步叠加：

- 更稳的 split 或交叉验证
- 临床变量联合建模
- 更系统的多序列融合策略

## 八、当前局限与风险

虽然本轮结果已经明显好于初始实现，但当前阶段仍存在以下限制。

- 当前评估仍以单次 `split_v1` 为主，稳定性还不够，需要补充 `5-fold` 或重复实验。
- 验证集与测试集之间仍存在波动，说明样本规模和当前 split 对结果影响较大。
- `M2` 样本暂未纳入当前主路线，三分类方向仍待后续明确。
- 少量病例仍存在 `image / seg / csv` 错位问题，需要在进入下一阶段前保持口径清晰。
- 当前尚未将临床变量纳入模型，故结论仍是“视觉特征基线”层面的结论，不等于最终多模态系统性能。

## 九、下一步建议

建议下一阶段按以下顺序推进。

第一步，以当前最佳视觉基线 `concat_sequences + l2 + tumor + SVM` 为默认对照组，补充 `5-fold` 或重复随机划分，验证结果稳定性。

第二步，在保持视觉基线不变的前提下，加入临床变量，构建视觉加临床的联合基线。

第三步，若联合基线得到稳定增益，再进一步考虑更复杂的多序列融合模块，而不是过早引入大规模结构改造。

第四步，在验证主路线稳定后，再讨论 `M2` 的三分类去留问题，以及解释性分析与报告化输出。

## 十、报告对应文件

本报告对应的关键文件位置如下：

- 正式报告：`/home/alex/Project/MRI/work-report/stage1_mvi_feature_probe_report_2026-04-01.md`
- 实验汇总：`/home/alex/Project/MRI/outputs/reports/results.md`
- 结果明细：`/home/alex/Project/MRI/outputs/reports/results.csv`
- 当前状态：`/home/alex/Project/MRI/.ai_context/05_CURRENT_STATUS.md`

本报告作为阶段 1 的正式文字记录，可直接作为后续阶段讨论和继续开发的依据。
