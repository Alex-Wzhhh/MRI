import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def main():
    # 0. 检查结果目录
    result_dir = "/home/alex/Project/MRI/result"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print(f"已创建结果目录: {result_dir}")

    # 1. 加载特征
    feature_file = "/home/alex/Project/MRI/extracted_features.npz"
    print(f"正在加载特征文件: {feature_file}...")
    try:
        data = np.load(feature_file)
        features = data['features'] # Shape: (N_slices, Feature_Dim)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {feature_file}。请先运行提取脚本。")
        return

    print(f"特征矩阵形状: {features.shape}")
    num_samples, num_features = features.shape

    # 2. 降维 (PCA + t-SNE)
    print("正在进行降维处理...")
    
    # 如果特征维度很高，先用 PCA 降到 50 维以加速 t-SNE
    if num_features > 50:
        print("  步骤 1/2: PCA 降维 (Dim -> 50)...")
        pca = PCA(n_components=50)
        features_pca = pca.fit_transform(features)
    else:
        features_pca = features

    # t-SNE 降到 2 维
    print("  步骤 2/2: t-SNE 降维 (50 -> 2)...")
    # perplexity 建议在 5-50 之间，样本少时设小一点
    perplexity = min(30, num_samples - 1) 
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
    features_2d = tsne.fit_transform(features_pca)

    # 3. 可视化
    print("正在生成可视化图表...")
    plt.figure(figsize=(12, 10))
    
    # 使用切片索引作为颜色 (模拟解剖位置的变化)
    # 颜色从浅到深代表从头部到脚部（或反之）
    indices = np.arange(num_samples)
    
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                          c=indices, cmap='viridis', alpha=0.7, s=60)
    
    plt.colorbar(scatter, label='Slice Depth (Z-Index)')
    plt.title(f'MedSAM2 Feature Space Visualization (t-SNE)\nInput: {num_samples} slices, Original Dim: {num_features}')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 保存图片到 result 文件夹
    output_img = os.path.join(result_dir, "feature_visualization.png")
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"可视化结果已保存为: {output_img}")
    print("解释: 图中颜色渐变的轨迹代表了人体解剖结构的连续变化。如果点聚在一起，说明这些切片的语义内容相似。")

if __name__ == "__main__":
    main()
