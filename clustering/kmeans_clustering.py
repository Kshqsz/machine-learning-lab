"""
K-Means聚类算法 (K-Means Clustering)
==========================================

K-Means是一种经典的基于划分的聚类算法，通过迭代优化将数据点分配到K个簇中。
算法的目标是最小化簇内平方误差（SSE），使得每个簇内的数据点尽可能相似，
而不同簇之间的数据点尽可能不同。

算法核心思想：
1. 随机初始化K个聚类中心
2. 将每个数据点分配到最近的聚类中心
3. 重新计算每个簇的中心（均值）
4. 重复步骤2-3直到收敛（中心不再变化或变化很小）

目标函数（最小化簇内平方误差SSE）：
J = Σ(k=1 to K) Σ(x∈Ck) ||x - μk||²

本实现包括：
- 标准K-Means算法
- K-Means++初始化（改进初始中心选择）
- 轮廓系数评估
- 肘部法则可视化
- 多种初始化方法对比

作者：Kshqsz
日期：2025-11-10
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
import warnings

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class KMeans:
    """
    K-Means聚类算法实现类
    
    参数:
    -----------
    n_clusters : int, default=3
        聚类数量K
    init : str, default='k-means++'
        初始化方法，可选: 'random', 'k-means++', 'manual'
    max_iter : int, default=300
        最大迭代次数
    tol : float, default=1e-4
        收敛阈值，中心变化小于此值则停止
    random_state : int, optional
        随机种子
    """
    
    def __init__(self, n_clusters: int = 3, init: str = 'k-means++', 
                 max_iter: int = 300, tol: float = 1e-4, 
                 random_state: Optional[int] = None):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        # 存储训练结果
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None  # 簇内平方误差SSE
        self.n_iter_ = 0  # 实际迭代次数
        self.n_samples_ = 0
        self.n_features_ = 0
        
        # 验证参数
        if init not in ['random', 'k-means++', 'manual']:
            raise ValueError(f"Unknown init method: {init}")
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _init_centers_random(self, X: np.ndarray) -> np.ndarray:
        """
        随机初始化聚类中心
        
        参数:
        -----------
        X : array-like, shape (n_samples, n_features)
            训练数据
            
        返回:
        -----------
        centers : array, shape (n_clusters, n_features)
            初始聚类中心
        """
        # 从数据集中随机选择K个点作为初始中心
        indices = np.random.choice(self.n_samples_, self.n_clusters, replace=False)
        return X[indices].copy()
    
    def _init_centers_kmeans_plus_plus(self, X: np.ndarray) -> np.ndarray:
        """
        K-Means++初始化聚类中心
        
        算法步骤：
        1. 随机选择第一个中心
        2. 对每个数据点，计算其到最近中心的距离D(x)
        3. 以概率 D(x)²/Σ D(x)² 选择下一个中心
        4. 重复步骤2-3直到选出K个中心
        
        优势：选出的初始中心彼此距离较远，提高收敛速度和结果质量
        
        参数:
        -----------
        X : array-like, shape (n_samples, n_features)
            训练数据
            
        返回:
        -----------
        centers : array, shape (n_clusters, n_features)
            初始聚类中心
        """
        centers = np.zeros((self.n_clusters, self.n_features_))
        
        # 随机选择第一个中心
        first_idx = np.random.randint(0, self.n_samples_)
        centers[0] = X[first_idx]
        
        # 选择剩余的中心
        for k in range(1, self.n_clusters):
            # 计算每个点到最近中心的距离
            distances = np.array([
                min([np.linalg.norm(x - center)**2 for center in centers[:k]])
                for x in X
            ])
            
            # 根据距离平方的概率分布选择下一个中心
            probabilities = distances / distances.sum()
            cumulative_probs = np.cumsum(probabilities)
            r = np.random.rand()
            
            for idx, prob in enumerate(cumulative_probs):
                if r < prob:
                    centers[k] = X[idx]
                    break
        
        return centers
    
    def _compute_distances(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """
        计算每个样本到每个中心的距离
        
        参数:
        -----------
        X : array-like, shape (n_samples, n_features)
            数据点
        centers : array-like, shape (n_clusters, n_features)
            聚类中心
            
        返回:
        -----------
        distances : array, shape (n_samples, n_clusters)
            距离矩阵
        """
        distances = np.zeros((X.shape[0], centers.shape[0]))
        for k in range(centers.shape[0]):
            # 计算欧几里得距离
            distances[:, k] = np.linalg.norm(X - centers[k], axis=1)
        return distances
    
    def _assign_clusters(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """
        将每个样本分配到最近的聚类中心
        
        参数:
        -----------
        X : array-like, shape (n_samples, n_features)
            数据点
        centers : array-like, shape (n_clusters, n_features)
            聚类中心
            
        返回:
        -----------
        labels : array, shape (n_samples,)
            每个样本的簇标签
        """
        distances = self._compute_distances(X, centers)
        return np.argmin(distances, axis=1)
    
    def _update_centers(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        重新计算每个簇的中心（均值）
        
        参数:
        -----------
        X : array-like, shape (n_samples, n_features)
            数据点
        labels : array, shape (n_samples,)
            每个样本的簇标签
            
        返回:
        -----------
        new_centers : array, shape (n_clusters, n_features)
            更新后的聚类中心
        """
        new_centers = np.zeros((self.n_clusters, self.n_features_))
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                new_centers[k] = cluster_points.mean(axis=0)
            else:
                # 如果某个簇为空，随机选择一个点作为新中心
                warnings.warn(f"Cluster {k} is empty. Reinitializing...")
                new_centers[k] = X[np.random.randint(0, self.n_samples_)]
        return new_centers
    
    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray, 
                         centers: np.ndarray) -> float:
        """
        计算簇内平方误差SSE（惯性）
        
        参数:
        -----------
        X : array-like, shape (n_samples, n_features)
            数据点
        labels : array, shape (n_samples,)
            每个样本的簇标签
        centers : array, shape (n_clusters, n_features)
            聚类中心
            
        返回:
        -----------
        inertia : float
            簇内平方误差
        """
        inertia = 0.0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - centers[k])**2)
        return inertia
    
    def fit(self, X: np.ndarray, initial_centers: Optional[np.ndarray] = None) -> 'KMeans':
        """
        训练K-Means模型
        
        参数:
        -----------
        X : array-like, shape (n_samples, n_features)
            训练数据
        initial_centers : array-like, shape (n_clusters, n_features), optional
            手动指定的初始中心（当init='manual'时使用）
            
        返回:
        -----------
        self : object
            返回自身实例
        """
        X = np.array(X)
        self.n_samples_, self.n_features_ = X.shape
        
        print(f"\n{'='*60}")
        print(f"K-Means聚类训练")
        print(f"{'='*60}")
        print(f"样本数量: {self.n_samples_}")
        print(f"特征维度: {self.n_features_}")
        print(f"聚类数量K: {self.n_clusters}")
        print(f"初始化方法: {self.init}")
        print(f"最大迭代次数: {self.max_iter}")
        print(f"{'='*60}\n")
        
        # 初始化聚类中心
        if self.init == 'random':
            centers = self._init_centers_random(X)
            print("使用随机初始化方法")
        elif self.init == 'k-means++':
            centers = self._init_centers_kmeans_plus_plus(X)
            print("使用K-Means++初始化方法")
        elif self.init == 'manual':
            if initial_centers is None:
                raise ValueError("Manual init requires initial_centers parameter")
            centers = np.array(initial_centers)
            print("使用手动指定的初始中心")
        
        # 迭代优化
        for iteration in range(self.max_iter):
            # 分配样本到最近的中心
            labels = self._assign_clusters(X, centers)
            
            # 更新中心
            new_centers = self._update_centers(X, labels)
            
            # 计算中心的变化
            center_shift = np.linalg.norm(new_centers - centers)
            
            # 计算当前惯性
            inertia = self._compute_inertia(X, labels, new_centers)
            
            if iteration % 10 == 0 or center_shift < self.tol:
                print(f"迭代 {iteration}: 中心变化={center_shift:.6f}, SSE={inertia:.4f}")
            
            # 检查收敛
            if center_shift < self.tol:
                print(f"\n收敛！迭代 {iteration} 次后中心变化小于阈值 {self.tol}")
                self.n_iter_ = iteration + 1
                break
            
            centers = new_centers
            self.n_iter_ = iteration + 1
        else:
            print(f"\n达到最大迭代次数 {self.max_iter}")
        
        # 保存结果
        self.cluster_centers_ = centers
        self.labels_ = labels
        self.inertia_ = self._compute_inertia(X, labels, centers)
        
        print(f"\n最终结果:")
        print(f"迭代次数: {self.n_iter_}")
        print(f"最终SSE: {self.inertia_:.4f}")
        print(f"{'='*60}\n")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测新数据的簇标签
        
        参数:
        -----------
        X : array-like, shape (n_samples, n_features)
            新数据
            
        返回:
        -----------
        labels : array, shape (n_samples,)
            预测的簇标签
        """
        if self.cluster_centers_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        X = np.array(X)
        return self._assign_clusters(X, self.cluster_centers_)
    
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
            包含簇大小、中心、SSE等信息
        """
        if self.labels_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        unique_labels = np.unique(self.labels_)
        cluster_sizes = [np.sum(self.labels_ == label) for label in unique_labels]
        
        info = {
            'n_clusters': len(unique_labels),
            'cluster_sizes': cluster_sizes,
            'cluster_centers': self.cluster_centers_,
            'inertia': self.inertia_,
            'n_iter': self.n_iter_,
        }
        
        return info


def compute_silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    计算轮廓系数（Silhouette Score）
    
    轮廓系数衡量聚类质量：
    - s(i) = (b(i) - a(i)) / max(a(i), b(i))
    - a(i): 样本i到同簇其他点的平均距离
    - b(i): 样本i到最近其他簇所有点的平均距离
    - 取值范围[-1, 1]，越接近1表示聚类越好
    
    参数:
    -----------
    X : array-like, shape (n_samples, n_features)
        数据
    labels : array, shape (n_samples,)
        聚类标签
        
    返回:
    -----------
    score : float
        平均轮廓系数
    """
    n_samples = X.shape[0]
    n_clusters = len(np.unique(labels))
    
    if n_clusters == 1 or n_clusters == n_samples:
        return 0.0
    
    silhouette_scores = np.zeros(n_samples)
    
    for i in range(n_samples):
        # 当前样本的簇
        cluster_i = labels[i]
        cluster_i_points = X[labels == cluster_i]
        
        # a(i): 同簇内其他点的平均距离
        if len(cluster_i_points) > 1:
            a_i = np.mean([np.linalg.norm(X[i] - p) for p in cluster_i_points if not np.array_equal(p, X[i])])
        else:
            a_i = 0
        
        # b(i): 到最近其他簇的平均距离
        b_i = float('inf')
        for cluster_j in range(n_clusters):
            if cluster_j != cluster_i:
                cluster_j_points = X[labels == cluster_j]
                if len(cluster_j_points) > 0:
                    avg_dist = np.mean([np.linalg.norm(X[i] - p) for p in cluster_j_points])
                    b_i = min(b_i, avg_dist)
        
        # 计算轮廓系数
        if max(a_i, b_i) > 0:
            silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i)
        else:
            silhouette_scores[i] = 0
    
    return np.mean(silhouette_scores)


def demo_basic_kmeans():
    """演示基础K-Means聚类"""
    print("\n" + "="*60)
    print("示例1: 基础K-Means聚类")
    print("="*60)
    
    # 生成3个明显分离的簇
    np.random.seed(42)
    cluster1 = np.random.randn(50, 2) * 0.5 + np.array([0, 0])
    cluster2 = np.random.randn(50, 2) * 0.5 + np.array([5, 5])
    cluster3 = np.random.randn(50, 2) * 0.5 + np.array([5, 0])
    X = np.vstack([cluster1, cluster2, cluster3])
    
    print(f"\n数据集信息:")
    print(f"样本数量: {len(X)}")
    print(f"特征维度: {X.shape[1]}")
    print(f"真实簇数: 3")
    
    # 创建图形
    fig = plt.figure(figsize=(16, 5))
    
    # 测试不同的初始化方法
    init_methods = ['random', 'k-means++']
    
    for idx, method in enumerate(init_methods):
        print(f"\n{'='*60}")
        print(f"测试初始化方法: {method}")
        print(f"{'='*60}")
        
        # 训练模型
        model = KMeans(n_clusters=3, init=method, random_state=42)
        labels = model.fit_predict(X)
        
        # 获取聚类信息
        info = model.get_cluster_info()
        silhouette = compute_silhouette_score(X, labels)
        
        print(f"\n聚类结果:")
        print(f"簇大小: {info['cluster_sizes']}")
        print(f"SSE: {info['inertia']:.4f}")
        print(f"轮廓系数: {silhouette:.4f}")
        
        # 绘制结果
        ax = plt.subplot(1, 3, idx + 1)
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis',
                           s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
        # 绘制聚类中心
        centers = info['cluster_centers']
        ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='X',
                  s=200, edgecolors='black', linewidth=2, label='Centers')
        ax.set_title(f'{method.upper()}\nSSE={info["inertia"]:.2f}, Silhouette={silhouette:.3f}',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Cluster')
    
    # 绘制原始数据
    ax = plt.subplot(1, 3, 3)
    true_labels = np.array([0]*50 + [1]*50 + [2]*50)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=true_labels, cmap='viridis',
                        s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.set_title('True Labels', fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Cluster')
    
    plt.tight_layout()
    plt.savefig('clustering/kmeans_basic_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n图形已保存: clustering/kmeans_basic_comparison.png")
    plt.close()


def demo_elbow_method():
    """演示肘部法则选择最佳K值"""
    print("\n" + "="*60)
    print("示例2: 使用肘部法则选择最佳K值")
    print("="*60)
    
    # 生成数据
    np.random.seed(42)
    cluster1 = np.random.randn(50, 2) * 0.5 + np.array([0, 0])
    cluster2 = np.random.randn(50, 2) * 0.5 + np.array([5, 5])
    cluster3 = np.random.randn(50, 2) * 0.5 + np.array([5, 0])
    X = np.vstack([cluster1, cluster2, cluster3])
    
    # 测试不同的K值
    k_range = range(2, 11)
    sse_values = []
    silhouette_values = []
    
    print("\n测试不同K值的效果:")
    for k in k_range:
        model = KMeans(n_clusters=k, init='k-means++', random_state=42)
        labels = model.fit_predict(X)
        sse_values.append(model.inertia_)
        silhouette = compute_silhouette_score(X, labels)
        silhouette_values.append(silhouette)
        print(f"K={k}: SSE={model.inertia_:.4f}, Silhouette={silhouette:.4f}")
    
    # 绘制肘部曲线和轮廓系数
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # 左图：SSE肘部曲线
    ax1.plot(k_range, sse_values, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('K值（聚类数量）', fontsize=12)
    ax1.set_ylabel('SSE（簇内平方误差）', fontsize=12)
    ax1.set_title('肘部法则 - SSE曲线', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=3, color='r', linestyle='--', linewidth=2, alpha=0.7, label='最佳K=3')
    ax1.legend()
    
    # 中图：轮廓系数
    ax2.plot(k_range, silhouette_values, 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('K值（聚类数量）', fontsize=12)
    ax2.set_ylabel('轮廓系数', fontsize=12)
    ax2.set_title('轮廓系数曲线', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    best_k = k_range[np.argmax(silhouette_values)]
    ax2.axvline(x=best_k, color='r', linestyle='--', linewidth=2, alpha=0.7, 
                label=f'最佳K={best_k}')
    ax2.legend()
    
    # 右图：最佳K的聚类结果
    model = KMeans(n_clusters=3, init='k-means++', random_state=42)
    labels = model.fit_predict(X)
    scatter = ax3.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis',
                         s=80, alpha=0.6, edgecolors='black', linewidth=1)
    centers = model.cluster_centers_
    ax3.scatter(centers[:, 0], centers[:, 1], c='red', marker='X',
               s=300, edgecolors='black', linewidth=2, label='Centers')
    ax3.set_title(f'最佳聚类结果 (K=3)\nSSE={model.inertia_:.2f}',
                 fontsize=14, fontweight='bold')
    ax3.set_xlabel('Feature 1', fontsize=12)
    ax3.set_ylabel('Feature 2', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Cluster')
    
    plt.tight_layout()
    plt.savefig('clustering/kmeans_elbow_method.png', dpi=150, bbox_inches='tight')
    print(f"\n图形已保存: clustering/kmeans_elbow_method.png")
    plt.close()


def demo_convergence_visualization():
    """演示K-Means收敛过程"""
    print("\n" + "="*60)
    print("示例3: K-Means收敛过程可视化")
    print("="*60)
    
    # 生成简单数据
    np.random.seed(42)
    cluster1 = np.random.randn(30, 2) * 0.5 + np.array([0, 0])
    cluster2 = np.random.randn(30, 2) * 0.5 + np.array([4, 4])
    cluster3 = np.random.randn(30, 2) * 0.5 + np.array([4, 0])
    X = np.vstack([cluster1, cluster2, cluster3])
    
    # 手动执行K-Means以记录每次迭代
    model = KMeans(n_clusters=3, init='random', max_iter=20, random_state=42)
    
    # 设置样本数和特征数
    model.n_samples_, model.n_features_ = X.shape
    
    # 初始化中心
    centers = model._init_centers_random(X)
    
    # 记录迭代过程
    iterations_to_plot = [0, 1, 2, 5, 10]
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    iteration = 0
    sse_history = []
    
    for step in range(20):
        # 分配簇
        labels = model._assign_clusters(X, centers)
        
        # 计算SSE
        sse = model._compute_inertia(X, labels, centers)
        sse_history.append(sse)
        
        # 绘制特定迭代
        if iteration in iterations_to_plot:
            idx = iterations_to_plot.index(iteration)
            ax = axes[idx]
            
            scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis',
                               s=60, alpha=0.6, edgecolors='black', linewidth=0.5)
            ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='X',
                      s=300, edgecolors='black', linewidth=2)
            ax.set_title(f'迭代 {iteration}\nSSE = {sse:.2f}',
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax, label='Cluster')
        
        # 更新中心
        new_centers = model._update_centers(X, labels)
        center_shift = np.linalg.norm(new_centers - centers)
        
        if center_shift < model.tol:
            print(f"收敛于迭代 {iteration}")
            break
        
        centers = new_centers
        iteration += 1
    
    # 绘制SSE收敛曲线
    ax = axes[-1]
    ax.plot(range(len(sse_history)), sse_history, 'b-o', linewidth=2, markersize=6)
    ax.set_xlabel('迭代次数', fontsize=12)
    ax.set_ylabel('SSE', fontsize=12)
    ax.set_title('SSE收敛曲线', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('clustering/kmeans_convergence.png', dpi=150, bbox_inches='tight')
    print(f"\n图形已保存: clustering/kmeans_convergence.png")
    plt.close()


def demo_different_k_values():
    """演示不同K值的聚类结果"""
    print("\n" + "="*60)
    print("示例4: 不同K值的聚类效果对比")
    print("="*60)
    
    # 生成数据
    np.random.seed(42)
    cluster1 = np.random.randn(50, 2) * 0.5 + np.array([0, 0])
    cluster2 = np.random.randn(50, 2) * 0.5 + np.array([5, 5])
    cluster3 = np.random.randn(50, 2) * 0.5 + np.array([5, 0])
    X = np.vstack([cluster1, cluster2, cluster3])
    
    # 测试不同K值
    k_values = [2, 3, 4, 5, 6, 8]
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for idx, k in enumerate(k_values):
        print(f"\n测试 K={k}")
        model = KMeans(n_clusters=k, init='k-means++', random_state=42)
        labels = model.fit_predict(X)
        
        silhouette = compute_silhouette_score(X, labels)
        
        ax = axes[idx]
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10',
                           s=60, alpha=0.6, edgecolors='black', linewidth=0.5)
        centers = model.cluster_centers_
        ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='X',
                  s=200, edgecolors='black', linewidth=2)
        ax.set_title(f'K={k}\nSSE={model.inertia_:.2f}, Silhouette={silhouette:.3f}',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Cluster')
    
    plt.tight_layout()
    plt.savefig('clustering/kmeans_different_k.png', dpi=150, bbox_inches='tight')
    print(f"\n图形已保存: clustering/kmeans_different_k.png")
    plt.close()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("K-Means聚类算法 (K-Means Clustering)")
    print("="*60)
    
    # 运行所有演示
    demo_basic_kmeans()
    demo_elbow_method()
    demo_convergence_visualization()
    demo_different_k_values()
    
    print("\n" + "="*60)
    print("所有演示完成！")
    print("="*60)
