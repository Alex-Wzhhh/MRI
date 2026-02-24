# 📋 代码修改与调整需求清单（AI 代理执行版）

以下是完整的代码修改需求，可直接交给 AI 代理执行。所有要求已按优先级和模块分类。

---

## 一、🔴 高优先级修改（必须完成）

### 1.1 可视化颜色编码修正

| 修改项 | 当前状态 | 目标状态 | 代码位置 |
| :--- | :--- | :--- | :--- |
| **颜色映射变量** | `color=sample_index` | `color=mvi_label` | `visualize_features()` |
| **颜色方案** | 连续渐变色 (viridis) | 离散分类色 (蓝/红) | `visualize_features()` |
| **图例说明** | Sample Index (0-125) | MVI Label (Negative/Positive) | `plt.legend()` |
| **Colorbar** | 显示 | 隐藏（分类任务不需要） | `plt.colorbar()` |

**具体代码修改要求：**
```python
# 删除或注释掉以下代码
# c=sample_indices, cmap='viridis'
# plt.colorbar(label='Sample Index')

# 替换为以下代码
colors = {0: '#2E86AB', 1: '#A23B72'}  # 0=蓝色=MVI 阴性，1=红色=MVI 阳性
labels = {0: 'MVI Negative', 1: 'MVI Positive'}

for label in [0, 1]:
    mask = mvi_labels == label
    plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
               c=[colors[label]], label=labels[label],
               alpha=0.6, s=80, edgecolors='white', linewidth=0.5)

plt.legend(loc='best')
# 不添加 colorbar
```

---

### 1.2 添加统计信息输出

| 统计项 | 输出格式 | 输出位置 |
| :--- | :--- | :--- |
| **PCA 解释方差** | `PC1: XX.X%, PC2: XX.X%, Total: XX.X%` | 控制台 print |
| **样本数量统计** | `MVI Negative: N 例，MVI Positive: N 例` | 控制台 print |
| **类别比例** | `Ratio: X:X` | 控制台 print |

**具体代码修改要求：**
```python
# 在可视化函数末尾添加以下代码
print("=" * 60)
print("PCA 解释方差统计:")
print(f"  PC1: {pca.explained_variance_ratio_[0]:.2%}")
print(f"  PC2: {pca.explained_variance_ratio_[1]:.2%}")
print(f"  PC1+PC2: {sum(pca.explained_variance_ratio_[:2]):.2%}")
print()
print("样本分布统计:")
print(f"  MVI Negative (0): {np.sum(mvi_labels == 0)} 例")
print(f"  MVI Positive (1): {np.sum(mvi_labels == 1)} 例")
print(f"  类别比例：{np.sum(mvi_labels == 0)}:{np.sum(mvi_labels == 1)}")
print("=" * 60)
```

---

### 1.3 数据加载模块修正

| 修改项 | 当前状态 | 目标状态 | 代码位置 |
| :--- | :--- | :--- | :--- |
| **标签加载** | 可能缺失或格式错误 | 从 CSV/JSON 正确加载 MVI 标签 | `load_labels()` |
| **标签验证** | 无 | 添加标签格式和范围检查 | `load_labels()` |
| **特征 - 标签对齐** | 可能未验证 | 添加样本 ID 匹配验证 | `load_data()` |

**具体代码修改要求：**
```python
def load_labels(label_file: str) -> dict:
    """
    加载 MVI 标签并验证
    
    Returns:
        dict: {patient_id: mvi_label}  mvi_label ∈ {0, 1}
    """
    import pandas as pd
    df = pd.read_csv(label_file)
    
    # 验证必要列存在
    assert 'patient_id' in df.columns, "缺少 patient_id 列"
    assert 'MVI_label' in df.columns, "缺少 MVI_label 列"
    
    # 验证标签取值
    unique_labels = df['MVI_label'].unique()
    assert set(unique_labels).issubset({0, 1}), f"标签必须为 0 或 1, 当前发现：{unique_labels}"
    
    return dict(zip(df['patient_id'], df['MVI_label']))

def load_and_align_data(feature_dir: str, labels: dict) -> tuple:
    """
    加载特征并与标签对齐
    """
    features = []
    mvi_labels = []
    patient_ids = []
    
    for fname in sorted(os.listdir(feature_dir)):
        if fname.endswith('.npy'):
            data = np.load(os.path.join(feature_dir, fname), allow_pickle=True).item()
            patient_id = data['metadata']['patient_id']
            
            if patient_id in labels:
                features.append(data['features'])
                mvi_labels.append(labels[patient_id])
                patient_ids.append(patient_id)
            else:
                print(f"⚠️ 警告：{patient_id} 无标签，跳过")
    
    print(f"✅ 成功加载 {len(features)} 例对齐样本")
    return np.array(features), np.array(mvi_labels), patient_ids
```

---

## 二、🟡 中优先级修改（建议完成）

### 2.1 可视化样式优化

| 修改项 | 当前设置 | 目标设置 | 目的 |
| :--- | :--- | :--- | :--- |
| **图片尺寸** | 默认 | `figsize=(14, 6)` 并排显示 | 论文出版质量 |
| **字体大小** | 默认 | `fontsize=12-14` | 清晰可读 |
| **点的大小** | 默认 | `s=80` | 适中可见 |
| **透明度** | 默认 | `alpha=0.6` | 重叠可见 |
| **边框** | 无 | `edgecolors='white', linewidth=0.5` | 点间区分 |
| **网格** | 无 | `grid(True, alpha=0.3)` | 便于读图 |
| **DPI** | 默认 | `dpi=300` | 出版质量 |

**具体代码修改要求：**
```python
# 在绘图代码中添加/修改以下参数
plt.style.use('seaborn-v0_8')  # 或'seaborn-whitegrid'

fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

# scatter 参数
plt.scatter(..., s=80, alpha=0.6, edgecolors='white', linewidth=0.5)

# 坐标轴标签
axes[i].set_xlabel('...', fontsize=12)
axes[i].set_ylabel('...', fontsize=12)
axes[i].set_title('...', fontsize=14, fontweight='bold')
axes[i].legend(loc='best', fontsize=10)
axes[i].grid(True, alpha=0.3)

# 保存
plt.savefig('output.png', dpi=300, bbox_inches='tight')
```

---

### 2.2 t-SNE 参数优化

| 参数 | 当前值 | 建议值 | 说明 |
| :--- | :--- | :--- | :--- |
| **perplexity** | 30 | 20-40（可配置） | 125 样本建议 20-30 |
| **n_iter** | 1000 | 1500 | 更稳定收敛 |
| **random_state** | 42 | 42（保持固定） | 保证可重复 |
| **learning_rate** | 默认 | 200（默认即可） | 通常无需调整 |

**具体代码修改要求：**
```python
# 添加参数配置
tsne_params = {
    'n_components': 2,
    'perplexity': 30,  # 可调整为 20/25/30/35/40 测试稳定性
    'random_state': 42,
    'n_iter': 1500,
    'learning_rate': 200
}

tsne = TSNE(**tsne_params)
```

---

### 2.3 批次效应检查（如有需要）

| 检查项 | 实现方式 | 输出 |
| :--- | :--- | :--- |
| **样本索引聚集检验** | 计算相邻样本在特征空间的平均距离 | 统计值 + 判断 |
| **可视化辅助** | 添加按批次着色的可选视图 | 额外图片 |

**具体代码修改要求：**
```python
def check_batch_effect(features, sample_indices, n_bins=5):
    """
    检查批次效应（样本索引是否影响特征分布）
    """
    from sklearn.metrics import pairwise_distances
    
    # 将样本按索引分成若干组
    bins = np.digitize(sample_indices, np.linspace(0, len(sample_indices), n_bins+1))
    
    # 计算组内和组间距离
    within_distances = []
    between_distances = []
    
    for b in range(1, n_bins+1):
        mask = bins == b
        if np.sum(mask) > 1:
            dist = pairwise_distances(features[mask])
            within_distances.extend(dist[np.triu_indices(np.sum(mask), k=1)])
    
    # 简化版：只输出警告
    print(f"\n批次效应检查:")
    print(f"  样本索引范围：{sample_indices.min()} - {sample_indices.max()}")
    print(f"  建议：如怀疑批次效应，请按采集时间/设备重新着色检查")
```

---

## 三、🟢 低优先级修改（可选增强）

### 3.1 输出文件组织

| 文件类型 | 命名规范 | 保存位置 |
| :--- | :--- | :--- |
| **PCA 图** | `pca_visualization_mvi.png` | `./figures/` |
| **t-SNE 图** | `tsne_visualization_mvi.png` | `./figures/` |
| **合并图** | `combined_visualization.png` | `./figures/` |
| **统计报告** | `feature_statistics.txt` | `./results/` |

**具体代码修改要求：**
```python
# 添加目录创建和文件保存
os.makedirs('./figures', exist_ok=True)
os.makedirs('./results', exist_ok=True)

plt.savefig('./figures/combined_visualization.png', dpi=300, bbox_inches='tight')

# 保存统计报告
with open('./results/feature_statistics.txt', 'w') as f:
    f.write(f"PCA 解释方差：PC1={pca.explained_variance_ratio_[0]:.4f}\n")
    f.write(f"样本分布：MVI Negative={np.sum(mvi_labels==0)}, Positive={np.sum(mvi_labels==1)}\n")
```

---

### 3.2 交互式参数配置

| 配置项 | 类型 | 默认值 |
| :--- | :--- | :--- |
| **feature_dir** | 字符串 | `./features/` |
| **label_file** | 字符串 | `./data/labels.csv` |
| **output_dir** | 字符串 | `./figures/` |
| **perplexity** | 整数 | 30 |
| **random_seed** | 整数 | 42 |

**具体代码修改要求：**
```python
# 添加配置文件或命令行参数支持
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--feature_dir', type=str, default='./features/')
parser.add_argument('--label_file', type=str, default='./data/labels.csv')
parser.add_argument('--output_dir', type=str, default='./figures/')
parser.add_argument('--perplexity', type=int, default=30)
parser.add_argument('--random_seed', type=int, default=42)
args = parser.parse_args()
```

---

## 四、📁 完整文件结构要求

```
project/
├── src/
│   ├── feature_extractor.py      # 特征提取模块
│   ├── visualization.py          # 可视化模块（重点修改）
│   ├── data_loader.py            # 数据加载模块
│   └── utils.py                  # 工具函数
├── data/
│   ├── features/                 # 提取的特征文件
│   └── labels.csv                # MVI 标签文件
├── figures/                      # 输出可视化图
│   ├── pca_visualization_mvi.png
│   ├── tsne_visualization_mvi.png
│   └── combined_visualization.png
├── results/                      # 统计报告
│   └── feature_statistics.txt
├── config/
│   └── config.yaml               # 配置文件（可选）
└── run_visualization.py          # 主执行脚本
```

---

## 五、✅ 验收标准清单

AI 代理完成修改后，请逐项检查：

```
┌─────────────────────────────────────────────────────────────────────┐
│                      修改验收清单                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  🔴 必须通过：                                                      │
│  ├── [ ] 可视化图颜色代表 MVI 标签（非样本索引）                     │
│  ├── [ ] 图例显示 MVI Negative / MVI Positive                       │
│  ├── [ ] 无 colorbar（分类任务不需要）                              │
│  ├── [ ] 控制台输出 PCA 解释方差（PC1+PC2 百分比）                   │
│  ├── [ ] 控制台输出 MVI 阳/阴性样本数量                             │
│  └── [ ] 代码可正常运行无报错                                       │
│                                                                     │
│  🟡 建议通过：                                                      │
│  ├── [ ] 图片尺寸 14×6 英寸并排显示                                │
│  ├── [ ] 字体大小 12-14，标题加粗                                   │
│  ├── [ ] 点有白色边框 (edgecolors='white')                          │
│  ├── [ ] 添加网格线 (grid=True)                                     │
│  ├── [ ] 保存 DPI=300                                              │
│  └── [ ] t-SNE perplexity 可配置 (20-40)                            │
│                                                                     │
│  🟢 可选增强：                                                      │
│  ├── [ ] 自动创建输出目录 (figures/, results/)                      │
│  ├── [ ] 保存统计报告到文件                                         │
│  ├── [ ] 支持命令行参数配置                                         │
│  └── [ ] 添加批次效应检查功能                                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 六、🚀 快速执行命令

```bash
# 1. 安装依赖
pip install numpy pandas matplotlib seaborn scikit-learn

# 2. 运行可视化
python run_visualization.py \
    --feature_dir ./data/features/ \
    --label_file ./data/labels.csv \
    --output_dir ./figures/ \
    --perplexity 30

# 3. 查看输出
ls -lh ./figures/
cat ./results/feature_statistics.txt
```

---

## 七、📧 给 AI 代理的提示词模板

```
请根据以下要求修改可视化代码：

1. 颜色编码：将散点图颜色从"样本索引"改为"MVI 标签"(0=蓝色=阴性，1=红色=阳性)
2. 统计输出：在控制台打印 PCA 解释方差和 MVI 样本分布统计
3. 样式优化：图片尺寸 14×6，DPI=300，点大小 80，添加白色边框和网格
4. 文件保存：自动创建 figures/目录，保存合并可视化图
5. 参数配置：t-SNE perplexity 可通过命令行参数配置

请确保代码可正常运行，输出符合验收标准。
```

---

## 💬 修改完成后请提供

1. **新生成的可视化图**（用 MVI 标签着色）
2. **控制台统计输出**（PCA 方差 + 样本分布）
3. **任何报错或警告信息**

拿到这些后，我可以给你准确的特征质量评估和后续训练建议！🎯