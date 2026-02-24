# ========== 特征可视化与质量检查 ==========
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import pandas as pd
import os
from typing import Tuple, List, Optional


class FeatureAnalyzer:
    """特征质量分析工具"""

    def __init__(self, feature_dir: str, label_file: Optional[str] = None):
        """
        Args:
            feature_dir: 特征文件目录
            label_file: 标签文件 (CSV，包含 case_id 和 MVI 标签)，可选
        """
        self.feature_dir = feature_dir
        self.labels = {}
        if label_file and os.path.exists(label_file):
            self.labels = self._load_labels(label_file)
            print(f"Loaded {len(self.labels)} labels from {label_file}")
        else:
            print("No label file provided or file not found. Running without labels.")

    def _load_labels(self, label_file: str) -> dict:
        """
        加载 MVI 标签并验证

        Returns:
            dict: {case_id: mvi_label}  mvi_label ∈ {0, 1}
        """
        df = pd.read_csv(label_file)

        # 验证必要列存在
        if 'case_id' not in df.columns:
            raise ValueError("标签文件缺少 'case_id' 列")
        if 'MVI_label' not in df.columns:
            raise ValueError("标签文件缺少 'MVI_label' 列")

        # 验证标签取值
        unique_labels = df['MVI_label'].unique()
        if not set(unique_labels).issubset({0, 1}):
            raise ValueError(f"标签必须为 0 或 1, 当前发现：{unique_labels}")

        print(f"✅ 成功加载 {len(df)} 个标签")
        print(f"   MVI Negative (0): {np.sum(df['MVI_label'] == 0)} 例")
        print(f"   MVI Positive (1): {np.sum(df['MVI_label'] == 1)} 例")

        return dict(zip(df['case_id'], df['MVI_label']))

    def load_all_features(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        加载所有特征并与标签对齐

        Returns:
            features: 特征数组 (N, 256)
            labels: 标签数组 (N,)，未匹配标签为 -1
            case_ids: 病例ID列表
        """
        features = []
        labels = []
        case_ids = []
        matched_count = 0
        unmatched_cases = []

        for fname in sorted(os.listdir(self.feature_dir)):
            if fname.endswith('.npy'):
                fpath = os.path.join(self.feature_dir, fname)
                data = np.load(fpath, allow_pickle=True).item()
                features.append(data['features'])
                case_id = data['metadata'].get('case_id', fname.replace('_features.npy', ''))
                case_ids.append(case_id)

                # 查找标签
                label = self.labels.get(case_id, -1)
                labels.append(label)

                if label >= 0:
                    matched_count += 1
                else:
                    unmatched_cases.append(case_id)

        # 输出对齐结果
        print(f"\n特征-标签对齐结果:")
        print(f"  总特征文件: {len(features)}")
        print(f"  成功匹配: {matched_count}")
        print(f"  未匹配: {len(unmatched_cases)}")

        if unmatched_cases:
            print(f"\n⚠️ 以下病例无标签（前5个）:")
            for case_id in unmatched_cases[:5]:
                print(f"    - {case_id}")
            if len(unmatched_cases) > 5:
                print(f"    ... 还有 {len(unmatched_cases) - 5} 个")

        return np.array(features), np.array(labels), case_ids

    def visualize_feature_distribution(self, features: np.ndarray,
                                       labels: np.ndarray,
                                       save_path: str = 'feature_distribution.png'):
        """特征分布可视化"""
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)

        # 1. 特征值分布直方图
        axes[0].hist(features.flatten(), bins=100, alpha=0.7, color='steelblue', edgecolor='white')
        axes[0].set_title('Feature Value Distribution', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Feature Value', fontsize=10)
        axes[0].set_ylabel('Frequency', fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # 2. 特征稀疏性
        sparsity = np.mean(np.abs(features) < 1e-6, axis=1)
        axes[1].hist(sparsity, bins=50, alpha=0.7, color='coral', edgecolor='white')
        axes[1].set_title('Feature Sparsity', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Zero Ratio', fontsize=10)
        axes[1].grid(True, alpha=0.3)

        # 3. 类别特征均值对比（如果有标签）
        valid_labels = labels[labels >= 0]
        if len(np.unique(valid_labels)) >= 2:
            mean_label0 = features[labels == 0].mean(axis=0)
            mean_label1 = features[labels == 1].mean(axis=0)
            axes[2].plot(mean_label0, label='MVI Negative', alpha=0.7, linewidth=1.5)
            axes[2].plot(mean_label1, label='MVI Positive', alpha=0.7, linewidth=1.5)
            axes[2].set_title('Mean Feature by Class', fontsize=12, fontweight='bold')
            axes[2].legend(fontsize=9)
            axes[2].grid(True, alpha=0.3)
        else:
            # 显示所有样本的平均特征
            mean_feature = features.mean(axis=0)
            std_feature = features.std(axis=0)
            x = range(len(mean_feature))
            axes[2].fill_between(x, mean_feature - std_feature, mean_feature + std_feature,
                                alpha=0.3, color='steelblue')
            axes[2].plot(mean_feature, color='steelblue', linewidth=1)
            axes[2].set_title('Mean Feature (with std)', fontsize=12, fontweight='bold')
            axes[2].set_xlabel('Feature Dimension', fontsize=10)
            axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Feature distribution plot saved to {save_path}")

    def visualize_dimensionality_reduction(self, features: np.ndarray,
                                           labels: np.ndarray,
                                           method: str = 't-SNE',
                                           save_path: str = None,
                                           perplexity: int = 30,
                                           n_iter: int = 1500):
        """降维可视化（支持MVI标签颜色编码）"""
        if save_path is None:
            save_path = f'{method.lower()}_visualization.png'

        if method == 'PCA':
            reducer = PCA(n_components=2)
        elif method == 't-SNE':
            reducer = TSNE(n_components=2, perplexity=min(perplexity, len(features)-1),
                          random_state=42, max_iter=n_iter, learning_rate=200)
        elif method == 'UMAP':
            try:
                import umap
                reducer = umap.UMAP(n_components=2, random_state=42)
            except ImportError:
                print("UMAP not installed, falling back to t-SNE")
                reducer = TSNE(n_components=2, perplexity=min(perplexity, len(features)-1),
                              random_state=42, max_iter=n_iter)
                method = 't-SNE'
        else:
            raise ValueError(f"Unknown method: {method}")

        print(f"Running {method}...")
        reduced = reducer.fit_transform(features)

        # 设置绘图样式
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

        # MVI标签颜色映射：0=蓝色=阴性，1=红色=阳性
        colors = {0: '#2E86AB', 1: '#A23B72'}
        labels_text = {0: 'MVI Negative', 1: 'MVI Positive'}

        valid_labels = labels[labels >= 0]
        if len(np.unique(valid_labels)) >= 2:
            # 有标签的情况：按MVI标签着色（分类任务）
            for label in [0, 1]:
                mask = labels == label
                if np.any(mask):
                    ax.scatter(reduced[mask, 0], reduced[mask, 1],
                              c=colors[label], label=labels_text[label],
                              alpha=0.6, s=80, edgecolors='white', linewidth=0.5)
            ax.legend(loc='best', fontsize=10)
            # 不添加colorbar（分类任务不需要）
        else:
            # 无标签的情况，用颜色表示样本索引
            scatter = ax.scatter(reduced[:, 0], reduced[:, 1],
                                 c=range(len(reduced)), cmap='viridis', alpha=0.6, s=50)
            plt.colorbar(scatter, label='Sample Index')

        ax.set_title(f'{method} Visualization ({len(features)} samples)',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Component 1', fontsize=12)
        ax.set_ylabel('Component 2', fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{method} visualization saved to {save_path}")

        # 返回reducer用于后续统计
        return reducer

    def analyze_feature_correlation(self, features: np.ndarray,
                                    save_path: str = 'feature_correlation.png'):
        """特征相关性分析"""
        # 计算特征间的相关系数矩阵
        corr_matrix = np.corrcoef(features.T)

        plt.figure(figsize=(12, 10))
        # 只显示部分特征（如果特征维度太高）
        if features.shape[1] > 50:
            # 显示前50个特征的相关性
            sns.heatmap(corr_matrix[:50, :50], cmap='RdBu_r', center=0,
                       xticklabels=5, yticklabels=5)
            plt.title('Feature Correlation Matrix (first 50 dimensions)')
        else:
            sns.heatmap(corr_matrix, cmap='RdBu_r', center=0)
            plt.title('Feature Correlation Matrix')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Correlation matrix saved to {save_path}")

    def analyze_feature_importance(self, features: np.ndarray,
                                   labels: np.ndarray,
                                   save_path: str = 'feature_importance.png'):
        """特征重要性分析（需要标签）"""
        valid_mask = labels >= 0
        valid_features = features[valid_mask]
        valid_labels = labels[valid_mask]

        if len(np.unique(valid_labels)) < 2:
            print("Warning: Need at least 2 classes for feature importance analysis")
            return None

        # 训练随机森林评估特征重要性
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(valid_features, valid_labels)

        importances = rf.feature_importances_

        # 绘制 Top 30 重要特征
        top_n = min(30, len(importances))
        top_indices = np.argsort(importances)[::-1][:top_n]

        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), importances[top_indices][::-1], color='steelblue')
        plt.yticks(range(top_n), [f'Dim {i}' for i in top_indices[::-1]])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Important Features')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Feature importance plot saved to {save_path}")

        return importances

    def print_statistics(self, features: np.ndarray, labels: np.ndarray,
                         pca_reducer: PCA = None):
        """打印统计信息到控制台"""
        print("=" * 60)
        print("样本分布统计:")
        valid_labels = labels[labels >= 0]
        n_negative = np.sum(valid_labels == 0)
        n_positive = np.sum(valid_labels == 1)
        print(f"  MVI Negative (0): {n_negative} 例")
        print(f"  MVI Positive (1): {n_positive} 例")
        if n_negative > 0 and n_positive > 0:
            ratio = n_negative / n_positive if n_positive > 0 else float('inf')
            print(f"  类别比例：{n_negative}:{n_positive} (约 {ratio:.2f}:1)")
        print()

        if pca_reducer is not None:
            print("PCA 解释方差统计:")
            print(f"  PC1: {pca_reducer.explained_variance_ratio_[0]:.2%}")
            print(f"  PC2: {pca_reducer.explained_variance_ratio_[1]:.2%}")
            print(f"  PC1+PC2: {sum(pca_reducer.explained_variance_ratio_[:2]):.2%}")
            print()
        print("=" * 60)

    def generate_summary_report(self, features: np.ndarray, case_ids: List[str],
                                save_path: str = 'feature_summary.txt'):
        """生成特征摘要报告"""
        with open(save_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("Feature Extraction Summary Report\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Total samples: {len(features)}\n")
            f.write(f"Feature dimension: {features.shape[1]}\n")
            f.write(f"Feature dtype: {features.dtype}\n\n")

            f.write("Feature Statistics:\n")
            f.write(f"  - Mean: {features.mean():.4f}\n")
            f.write(f"  - Std: {features.std():.4f}\n")
            f.write(f"  - Min: {features.min():.4f}\n")
            f.write(f"  - Max: {features.max():.4f}\n")
            f.write(f"  - Sparsity (|x| < 1e-6): {(np.abs(features) < 1e-6).mean()*100:.2f}%\n\n")

            f.write("Per-dimension Statistics:\n")
            dim_means = features.mean(axis=0)
            dim_stds = features.std(axis=0)
            f.write(f"  - Mean range: [{dim_means.min():.4f}, {dim_means.max():.4f}]\n")
            f.write(f"  - Std range: [{dim_stds.min():.4f}, {dim_stds.max():.4f}]\n\n")

            f.write("Case IDs:\n")
            for i, case_id in enumerate(case_ids[:10]):
                f.write(f"  {i+1}. {case_id}\n")
            if len(case_ids) > 10:
                f.write(f"  ... and {len(case_ids) - 10} more\n")

        print(f"Summary report saved to {save_path}")


# 使用示例
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Feature Analysis and Visualization')
    parser.add_argument('--feature_dir', type=str,
                       default='/home/alex/Project/MRI/Data/features/',
                       help='Directory containing feature files')
    parser.add_argument('--label_file', type=str,
                       default='/home/alex/Project/MRI/Data/labels.csv',
                       help='CSV file with case_id and MVI_label columns')
    parser.add_argument('--output_dir', type=str,
                       default='/home/alex/Project/MRI/Data/analysis/',
                       help='Output directory for plots and reports')
    parser.add_argument('--perplexity', type=int, default=30,
                       help='t-SNE perplexity parameter (default: 30)')
    parser.add_argument('--n_iter', type=int, default=1500,
                       help='t-SNE number of iterations (default: 1500)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    args = parser.parse_args()

    # 设置随机种子
    np.random.seed(args.random_seed)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化分析器
    analyzer = FeatureAnalyzer(
        feature_dir=args.feature_dir,
        label_file=args.label_file if os.path.exists(args.label_file) else None
    )

    # 加载特征
    print("\n" + "="*50)
    print("Loading features...")
    features, labels, case_ids = analyzer.load_all_features()
    print(f"Total features: {features.shape}, Labels: {labels.shape}")

    # 打印样本分布统计
    print("\n" + "="*50)
    print("Sample Statistics:")
    analyzer.print_statistics(features, labels)

    # 生成摘要报告
    print("\n" + "="*50)
    print("Generating summary report...")
    analyzer.generate_summary_report(features, case_ids,
                                     os.path.join(args.output_dir, 'feature_summary.txt'))

    # 特征分布可视化
    print("\n" + "="*50)
    print("Visualizing feature distribution...")
    analyzer.visualize_feature_distribution(features, labels,
                                           os.path.join(args.output_dir, 'feature_distribution.png'))

    # 降维可视化
    print("\n" + "="*50)
    print("Running dimensionality reduction...")

    # PCA可视化并获取reducer用于统计
    pca_reducer = analyzer.visualize_dimensionality_reduction(
        features, labels,
        method='PCA',
        save_path=os.path.join(args.output_dir, 'pca_visualization.png'),
        perplexity=args.perplexity,
        n_iter=args.n_iter
    )

    # t-SNE可视化
    analyzer.visualize_dimensionality_reduction(
        features, labels,
        method='t-SNE',
        save_path=os.path.join(args.output_dir, 'tsne_visualization.png'),
        perplexity=args.perplexity,
        n_iter=args.n_iter
    )

    # 打印PCA统计信息
    print("\n" + "="*50)
    print("PCA Statistics:")
    analyzer.print_statistics(features, labels, pca_reducer=pca_reducer)

    # 特征相关性分析
    print("\n" + "="*50)
    print("Analyzing feature correlation...")
    analyzer.analyze_feature_correlation(features,
                                        os.path.join(args.output_dir, 'feature_correlation.png'))

    # 特征重要性分析（如果有标签）
    if len(analyzer.labels) > 0:
        print("\n" + "="*50)
        print("Analyzing feature importance...")
        analyzer.analyze_feature_importance(features, labels,
                                           os.path.join(args.output_dir, 'feature_importance.png'))

    print("\n" + "="*50)
    print("Analysis complete!")
    print(f"Results saved to: {args.output_dir}")
