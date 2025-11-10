"""
层次聚类算法 (Hierarchical Clustering)
===========================================

层次聚类是一种聚类分析方法，通过计算不同类别数据点间的相似度来创建一棵有层次的嵌套聚类树。
在聚类树中，不同类别的原始数据点是树的最底层，树的顶层是一个聚类的根节点。

层次聚类分为两类：
1. 凝聚层次聚类（Agglomerative）：自底向上的合并策略
2. 分裂层次聚类（Divisive）：自顶向下的分裂策略

本实现专注于凝聚层次聚类，支持多种链接方法：
- 单链接（Single Linkage）：最近邻距离
- 全链接（Complete Linkage）：最远邻距离
- 平均链接（Average Linkage）：平均距离
- Ward方法：最小化类内方差

算法步骤：
1. 初始化：将每个样本作为一个簇
2. 计算所有簇对之间的距离
3. 合并距离最近的两个簇
4. 更新距离矩阵
5. 重复步骤3-4直到达到期望的簇数量或所有样本合并为一个簇

作者：Kshqsz
日期：2025-11-10
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union
from scipy.cluster.hierarchy import dendrogram
import warnings

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class HierarchicalClustering:
    """
    层次聚类算法实现类
    
    参数:
    -----------
    n_clusters : int, default=2
        最终聚类数量
    linkage : str, default='average'
        链接方法，可选: 'single', 'complete', 'average', 'ward'
    metric : str, default='euclidean'
        距离度量方法，可选: 'euclidean', 'manhattan', 'cosine'
    """
    
    def __init__(self, n_clusters: int = 2, linkage: str = 'average', metric: str = 'euclidean'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.metric = metric
        
        # 存储训练结果
        self.labels_ = None
        self.linkage_matrix_ = None  # 存储层次树结构
        self.distances_ = []  # 存储每次合并时的距离
        self.n_samples_ = 0
        self.n_features_ = 0
        
        # 验证参数
        if linkage not in ['single', 'complete', 'average', 'ward']:
            raise ValueError(f"Unknown linkage type: {linkage}")
        if metric not in ['euclidean', 'manhattan', 'cosine']:
            raise ValueError(f"Unknown metric: {metric}")
        if linkage == 'ward' and metric != 'euclidean':
            warnings.warn("Ward linkage only supports euclidean distance, switching to euclidean")
            self.metric = 'euclidean'
    
    def _compute_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        计算两个向量之间的距离
        
        参数:
        -----------
        x1, x2 : array-like
            输入向量
            
        返回:
        -----------
        distance : float
            距离值
        """
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.metric == 'cosine':
            # 余弦距离 = 1 - 余弦相似度
            dot_product = np.dot(x1, x2)
            norm_product = np.linalg.norm(x1) * np.linalg.norm(x2)
            if norm_product == 0:
                return 1.0
            return 1.0 - dot_product / norm_product
    
    def _compute_cluster_distance(self, cluster1_indices: List[int], 
                                  cluster2_indices: List[int], 
                                  X: np.ndarray) -> float:
        """
        计算两个簇之间的距离
        
        参数:
        -----------
        cluster1_indices, cluster2_indices : list
            两个簇包含的样本索引
        X : array-like, shape (n_samples, n_features)
            训练数据
            
        返回:
        -----------
        distance : float
            簇间距离
        """
        if self.linkage == 'single':
            # 单链接：最近邻距离（最小距离）
            min_dist = float('inf')
            for i in cluster1_indices:
                for j in cluster2_indices:
                    dist = self._compute_distance(X[i], X[j])
                    if dist < min_dist:
                        min_dist = dist
            return min_dist
        
        elif self.linkage == 'complete':
            # 全链接：最远邻距离（最大距离）
            max_dist = 0.0
            for i in cluster1_indices:
                for j in cluster2_indices:
                    dist = self._compute_distance(X[i], X[j])
                    if dist > max_dist:
                        max_dist = dist
            return max_dist
        
        elif self.linkage == 'average':
            # 平均链接：平均距离
            total_dist = 0.0
            count = 0
            for i in cluster1_indices:
                for j in cluster2_indices:
                    total_dist += self._compute_distance(X[i], X[j])
                    count += 1
            return total_dist / count if count > 0 else 0.0
        
        elif self.linkage == 'ward':
            # Ward方法：最小化类内方差增量
            # 计算两个簇的中心
            center1 = np.mean(X[cluster1_indices], axis=0)
            center2 = np.mean(X[cluster2_indices], axis=0)
            
            # 合并后的中心
            n1, n2 = len(cluster1_indices), len(cluster2_indices)
            merged_center = (n1 * center1 + n2 * center2) / (n1 + n2)
            
            # 计算方差增量
            # Δ = n1*n2/(n1+n2) * ||center1 - center2||²
            dist_squared = np.sum((center1 - center2) ** 2)
            return np.sqrt((n1 * n2) / (n1 + n2) * dist_squared)
    
    def fit(self, X: np.ndarray) -> 'HierarchicalClustering':
        """
        训练层次聚类模型
        
        参数:
        -----------
        X : array-like, shape (n_samples, n_features)
            训练数据
            
        返回:
        -----------
        self : object
            返回自身实例
        """
        X = np.array(X)
        self.n_samples_, self.n_features_ = X.shape
        
        print(f"\n{'='*60}")
        print(f"层次聚类训练")
        print(f"{'='*60}")
        print(f"样本数量: {self.n_samples_}")
        print(f"特征维度: {self.n_features_}")
        print(f"目标簇数: {self.n_clusters}")
        print(f"链接方法: {self.linkage}")
        print(f"距离度量: {self.metric}")
        print(f"{'='*60}\n")
        
        # 初始化：每个样本是一个簇
        clusters = [[i] for i in range(self.n_samples_)]
        
        # 用于构建层次树的矩阵 (n_samples-1, 4)
        # 每行: [cluster1_id, cluster2_id, distance, n_samples_in_cluster]
        linkage_matrix = []
        next_cluster_id = self.n_samples_  # 新簇的ID从n_samples开始
        
        # 为每个当前簇分配一个ID（初始时就是样本索引）
        cluster_ids = list(range(self.n_samples_))
        
        # 迭代合并直到达到目标簇数
        step = 0
        while len(clusters) > self.n_clusters:
            step += 1
            
            # 找到距离最近的两个簇
            min_dist = float('inf')
            merge_i, merge_j = -1, -1
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dist = self._compute_cluster_distance(clusters[i], clusters[j], X)
                    if dist < min_dist:
                        min_dist = dist
                        merge_i, merge_j = i, j
            
            # 获取要合并的簇的ID
            cluster_i_id = cluster_ids[merge_i]
            cluster_j_id = cluster_ids[merge_j]
            
            merged_cluster = clusters[merge_i] + clusters[merge_j]
            
            # 记录到linkage矩阵（确保较小的ID在前）
            id1, id2 = min(cluster_i_id, cluster_j_id), max(cluster_i_id, cluster_j_id)
            linkage_matrix.append([
                id1,
                id2,
                min_dist,
                len(merged_cluster)
            ])
            self.distances_.append(min_dist)
            
            if step <= 5 or len(clusters) <= self.n_clusters + 1:
                print(f"步骤 {step}: 合并簇 {cluster_i_id} 和簇 {cluster_j_id}")
                print(f"  距离: {min_dist:.4f}")
                print(f"  新簇大小: {len(merged_cluster)}")
                print(f"  剩余簇数: {len(clusters) - 1}")
            
            # 更新簇列表和ID列表
            # 先删除索引较大的，避免索引错乱
            clusters.pop(merge_j)
            cluster_ids.pop(merge_j)
            
            clusters.pop(merge_i)
            cluster_ids.pop(merge_i)
            
            # 添加合并后的新簇
            clusters.append(merged_cluster)
            cluster_ids.append(next_cluster_id)
            next_cluster_id += 1
        
        print(f"\n聚类完成！最终簇数: {len(clusters)}")
        print(f"{'='*60}\n")
        
        # 生成标签
        self.labels_ = np.zeros(self.n_samples_, dtype=int)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                self.labels_[sample_idx] = cluster_idx
        
        # 保存linkage矩阵
        self.linkage_matrix_ = np.array(linkage_matrix)
        
        return self
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        训练模型并返回聚类标签
        
        参数:
        -----------
        X : array-like, shape (n_samples, n_features)
            训练数据
            
        返回:
        -----------
        labels : array, shape (n_samples,)
            聚类标签
        """
        self.fit(X)
        return self.labels_
    
    def get_cluster_info(self) -> dict:
        """
        获取聚类信息
        
        返回:
        -----------
        info : dict
            包含簇大小、距离统计等信息
        """
        if self.labels_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        unique_labels = np.unique(self.labels_)
        cluster_sizes = [np.sum(self.labels_ == label) for label in unique_labels]
        
        info = {
            'n_clusters': len(unique_labels),
            'cluster_sizes': cluster_sizes,
            'min_cluster_size': min(cluster_sizes),
            'max_cluster_size': max(cluster_sizes),
            'merge_distances': self.distances_,
            'min_merge_distance': min(self.distances_) if self.distances_ else 0,
            'max_merge_distance': max(self.distances_) if self.distances_ else 0,
        }
        
        return info
    
    def plot_dendrogram(self, X: np.ndarray, figsize: Tuple[int, int] = (12, 6), 
                        title: str = None) -> plt.Figure:
        """
        绘制树状图（dendrogram）
        
        参数:
        -----------
        X : array-like, shape (n_samples, n_features)
            原始数据（用于重新计算linkage矩阵）
        figsize : tuple, default=(12, 6)
            图形大小
        title : str, optional
            图形标题
            
        返回:
        -----------
        fig : matplotlib.figure.Figure
            图形对象
        """
        from scipy.cluster.hierarchy import linkage as scipy_linkage
        
        # 使用scipy重新计算linkage矩阵以确保格式正确
        Z = scipy_linkage(X, method=self.linkage, metric=self.metric)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制树状图
        dendrogram(Z, ax=ax, color_threshold=None)
        
        if title is None:
            title = f'层次聚类树状图 ({self.linkage.capitalize()} Linkage)'
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('样本索引', fontsize=12)
        ax.set_ylabel('距离', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig


def demo_basic_hierarchical_clustering():
    """演示基础层次聚类"""
    print("\n" + "="*60)
    print("示例1: 基础层次聚类")
    print("="*60)
    
    # 生成简单的2D数据集
    np.random.seed(42)
    
    # 三个明显分离的簇
    cluster1 = np.random.randn(20, 2) * 0.5 + np.array([0, 0])
    cluster2 = np.random.randn(20, 2) * 0.5 + np.array([5, 5])
    cluster3 = np.random.randn(20, 2) * 0.5 + np.array([5, 0])
    
    X = np.vstack([cluster1, cluster2, cluster3])
    true_labels = np.array([0]*20 + [1]*20 + [2]*20)
    
    print(f"\n数据集信息:")
    print(f"样本数量: {len(X)}")
    print(f"特征维度: {X.shape[1]}")
    print(f"真实簇数: 3")
    
    # 创建图形
    fig = plt.figure(figsize=(16, 10))
    
    # 测试不同的链接方法
    linkage_methods = ['single', 'complete', 'average', 'ward']
    
    for idx, method in enumerate(linkage_methods):
        print(f"\n{'='*60}")
        print(f"测试链接方法: {method}")
        print(f"{'='*60}")
        
        # 训练模型
        model = HierarchicalClustering(n_clusters=3, linkage=method)
        labels = model.fit_predict(X)
        
        # 获取聚类信息
        info = model.get_cluster_info()
        print(f"\n聚类结果:")
        print(f"簇大小: {info['cluster_sizes']}")
        print(f"最小合并距离: {info['min_merge_distance']:.4f}")
        print(f"最大合并距离: {info['max_merge_distance']:.4f}")
        
        # 绘制聚类结果
        ax1 = plt.subplot(4, 2, idx*2 + 1)
        scatter = ax1.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', 
                             s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax1.set_title(f'{method.capitalize()} Linkage - 聚类结果', 
                     fontsize=12, fontweight='bold')
        ax1.set_xlabel('Feature 1')
        ax1.set_ylabel('Feature 2')
        plt.colorbar(scatter, ax=ax1, label='Cluster')
        ax1.grid(True, alpha=0.3)
        
        # 绘制树状图
        ax2 = plt.subplot(4, 2, idx*2 + 2)
        from scipy.cluster.hierarchy import linkage as scipy_linkage
        Z = scipy_linkage(X, method=method, metric='euclidean')
        dendrogram(Z, ax=ax2, color_threshold=None, no_labels=True)
        ax2.set_title(f'{method.capitalize()} Linkage - 树状图', 
                     fontsize=12, fontweight='bold')
        ax2.set_xlabel('样本索引')
        ax2.set_ylabel('距离')
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('clustering/hierarchical_clustering_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n图形已保存: clustering/hierarchical_clustering_comparison.png")
    plt.show()


def demo_distance_metrics():
    """演示不同距离度量方法"""
    print("\n" + "="*60)
    print("示例2: 不同距离度量方法比较")
    print("="*60)
    
    # 生成数据
    np.random.seed(42)
    cluster1 = np.random.randn(15, 2) * 0.8 + np.array([0, 0])
    cluster2 = np.random.randn(15, 2) * 0.8 + np.array([4, 4])
    X = np.vstack([cluster1, cluster2])
    
    # 创建图形
    fig = plt.figure(figsize=(15, 5))
    
    metrics = ['euclidean', 'manhattan', 'cosine']
    
    for idx, metric in enumerate(metrics):
        print(f"\n{'='*60}")
        print(f"测试距离度量: {metric}")
        print(f"{'='*60}")
        
        # 训练模型（使用average linkage）
        model = HierarchicalClustering(n_clusters=2, linkage='average', metric=metric)
        labels = model.fit_predict(X)
        
        # 绘制结果
        ax = plt.subplot(1, 3, idx + 1)
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis',
                           s=80, alpha=0.6, edgecolors='black', linewidth=1)
        ax.set_title(f'{metric.capitalize()} Distance', fontsize=12, fontweight='bold')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        plt.colorbar(scatter, ax=ax, label='Cluster')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('clustering/hierarchical_distance_metrics.png', dpi=150, bbox_inches='tight')
    print(f"\n图形已保存: clustering/hierarchical_distance_metrics.png")
    plt.show()


def demo_elbow_method():
    """演示肘部法则选择最佳簇数"""
    print("\n" + "="*60)
    print("示例3: 使用肘部法则选择最佳簇数")
    print("="*60)
    
    # 生成数据
    np.random.seed(42)
    cluster1 = np.random.randn(20, 2) * 0.5 + np.array([0, 0])
    cluster2 = np.random.randn(20, 2) * 0.5 + np.array([5, 5])
    cluster3 = np.random.randn(20, 2) * 0.5 + np.array([5, 0])
    X = np.vstack([cluster1, cluster2, cluster3])
    
    # 测试不同的簇数
    n_clusters_range = range(2, 11)
    max_distances = []
    
    print("\n测试不同簇数的效果:")
    for n in n_clusters_range:
        model = HierarchicalClustering(n_clusters=n, linkage='ward')
        model.fit(X)
        info = model.get_cluster_info()
        max_distances.append(info['max_merge_distance'])
        print(f"簇数 = {n}: 最大合并距离 = {info['max_merge_distance']:.4f}")
    
    # 绘制肘部曲线
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：肘部曲线
    ax1.plot(n_clusters_range, max_distances, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('簇数量', fontsize=12)
    ax1.set_ylabel('最大合并距离', fontsize=12)
    ax1.set_title('肘部法则 - 选择最佳簇数', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=3, color='r', linestyle='--', linewidth=2, alpha=0.7, label='最佳簇数=3')
    ax1.legend()
    
    # 右图：最佳簇数的聚类结果
    model = HierarchicalClustering(n_clusters=3, linkage='ward')
    labels = model.fit_predict(X)
    scatter = ax2.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis',
                         s=80, alpha=0.6, edgecolors='black', linewidth=1)
    ax2.set_title('最佳聚类结果 (n_clusters=3)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Feature 1', fontsize=12)
    ax2.set_ylabel('Feature 2', fontsize=12)
    plt.colorbar(scatter, ax=ax2, label='Cluster')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('clustering/hierarchical_elbow_method.png', dpi=150, bbox_inches='tight')
    print(f"\n图形已保存: clustering/hierarchical_elbow_method.png")
    plt.show()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("层次聚类算法 (Hierarchical Clustering)")
    print("="*60)
    
    # 运行所有演示
    demo_basic_hierarchical_clustering()
    demo_distance_metrics()
    demo_elbow_method()
    
    print("\n" + "="*60)
    print("所有演示完成！")
    print("="*60)
