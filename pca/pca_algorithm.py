"""
主成分分析 (Principal Component Analysis, PCA)
==========================================

PCA是一种经典的无监督学习算法，用于数据降维和特征提取。
通过线性变换将原始数据投影到新的坐标系统中，使得数据在新坐标轴（主成分）上的方差最大化。

理论基础：
---------
1. 数据中心化: X_centered = X - mean(X)
2. 协方差矩阵: C = (1/n) * X_centered^T * X_centered
3. 特征分解: C = V * Λ * V^T
   - V: 特征向量矩阵（主成分方向）
   - Λ: 特征值矩阵（方差大小）
4. 降维投影: X_reduced = X_centered * V_k
   其中V_k是前k个最大特征值对应的特征向量

主要方法：
---------
1. 标准PCA (基于协方差矩阵的特征分解)
2. SVD-based PCA (基于奇异值分解，数值更稳定)
3. 增量PCA (Incremental PCA，适合大数据集)
4. 核PCA (Kernel PCA，处理非线性数据)

应用场景：
---------
- 数据可视化（降维到2D/3D）
- 特征提取和压缩
- 噪声过滤
- 图像处理（人脸识别等）
- 数据预处理
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Tuple, Union

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class PCA:
    """
    主成分分析 (Principal Component Analysis)
    
    Parameters:
    -----------
    n_components : int or float, optional (default=None)
        要保留的主成分数量
        - 如果是int: 保留前n_components个主成分
        - 如果是float (0-1): 保留解释方差比例达到该值的主成分数量
        - 如果是None: 保留所有主成分
    
    method : str, optional (default='svd')
        PCA实现方法
        - 'eigen': 基于协方差矩阵的特征分解
        - 'svd': 基于奇异值分解（推荐，数值更稳定）
    
    whiten : bool, optional (default=False)
        是否进行白化处理（使各主成分方差归一化）
    
    Attributes:
    -----------
    components_ : ndarray, shape (n_components, n_features)
        主成分（特征向量）
    
    explained_variance_ : ndarray, shape (n_components,)
        每个主成分解释的方差
    
    explained_variance_ratio_ : ndarray, shape (n_components,)
        每个主成分解释的方差比例
    
    singular_values_ : ndarray, shape (n_components,)
        奇异值
    
    mean_ : ndarray, shape (n_features,)
        训练数据的均值
    
    n_samples_ : int
        训练样本数量
    
    n_features_ : int
        特征数量
    """
    
    def __init__(self, n_components: Optional[Union[int, float]] = None, 
                 method: str = 'svd', whiten: bool = False):
        self.n_components = n_components
        self.method = method
        self.whiten = whiten
        
        # 模型参数
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.n_samples_ = None
        self.n_features_ = None
        self.n_components_ = None
    
    def fit(self, X: np.ndarray) -> 'PCA':
        """
        拟合PCA模型
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features)
            训练数据
        
        Returns:
        --------
        self : object
        """
        X = np.asarray(X, dtype=np.float64)
        self.n_samples_, self.n_features_ = X.shape
        
        # 数据中心化
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        if self.method == 'svd':
            # 使用SVD方法（推荐）
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            
            # 主成分 = 右奇异向量的行向量（V的列向量）
            self.components_ = Vt
            
            # 奇异值
            self.singular_values_ = S
            
            # 解释方差 = 奇异值的平方 / (n-1)
            self.explained_variance_ = (S ** 2) / (self.n_samples_ - 1)
            
        elif self.method == 'eigen':
            # 使用特征分解方法
            # 计算协方差矩阵
            cov_matrix = np.cov(X_centered.T)
            
            # 特征分解
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            
            # 按特征值降序排列
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # 主成分
            self.components_ = eigenvectors.T
            
            # 解释方差
            self.explained_variance_ = eigenvalues
            
            # 奇异值（从特征值计算）
            self.singular_values_ = np.sqrt(eigenvalues * (self.n_samples_ - 1))
        
        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'svd' or 'eigen'.")
        
        # 计算解释方差比例
        total_variance = np.sum(self.explained_variance_)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        
        # 确定要保留的主成分数量
        if self.n_components is None:
            self.n_components_ = min(self.n_samples_, self.n_features_)
        elif isinstance(self.n_components, float):
            # 根据方差比例确定
            cumsum = np.cumsum(self.explained_variance_ratio_)
            self.n_components_ = np.searchsorted(cumsum, self.n_components) + 1
        else:
            self.n_components_ = min(self.n_components, len(self.explained_variance_))
        
        # 截断到指定的主成分数量
        self.components_ = self.components_[:self.n_components_]
        self.explained_variance_ = self.explained_variance_[:self.n_components_]
        self.explained_variance_ratio_ = self.explained_variance_ratio_[:self.n_components_]
        self.singular_values_ = self.singular_values_[:self.n_components_]
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        将数据投影到主成分空间
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features)
            输入数据
        
        Returns:
        --------
        X_transformed : ndarray, shape (n_samples, n_components)
            投影后的数据
        """
        X = np.asarray(X, dtype=np.float64)
        
        # 中心化
        X_centered = X - self.mean_
        
        # 投影
        X_transformed = np.dot(X_centered, self.components_.T)
        
        # 白化处理
        if self.whiten:
            X_transformed /= np.sqrt(self.explained_variance_)
        
        return X_transformed
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        拟合模型并转换数据
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features)
            训练数据
        
        Returns:
        --------
        X_transformed : ndarray, shape (n_samples, n_components)
            投影后的数据
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        将降维后的数据重建回原始空间
        
        Parameters:
        -----------
        X_transformed : ndarray, shape (n_samples, n_components)
            降维后的数据
        
        Returns:
        --------
        X_reconstructed : ndarray, shape (n_samples, n_features)
            重建后的数据
        """
        if self.whiten:
            X_transformed = X_transformed * np.sqrt(self.explained_variance_)
        
        X_reconstructed = np.dot(X_transformed, self.components_) + self.mean_
        return X_reconstructed
    
    def get_covariance(self) -> np.ndarray:
        """
        获取数据的协方差矩阵（在主成分空间中重建）
        
        Returns:
        --------
        cov : ndarray, shape (n_features, n_features)
            协方差矩阵
        """
        # C = V * Λ * V^T
        components = self.components_.T
        variance_matrix = np.diag(self.explained_variance_)
        cov = components @ variance_matrix @ components.T
        return cov
    
    def reconstruction_error(self, X: np.ndarray) -> float:
        """
        计算重建误差
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features)
            原始数据
        
        Returns:
        --------
        error : float
            重建误差（Frobenius范数）
        """
        X_transformed = self.transform(X)
        X_reconstructed = self.inverse_transform(X_transformed)
        error = np.linalg.norm(X - X_reconstructed, 'fro')
        return error


class IncrementalPCA:
    """
    增量主成分分析 (Incremental PCA)
    
    适合大规模数据集的PCA实现，可以批量处理数据，不需要一次性加载所有数据到内存。
    
    Parameters:
    -----------
    n_components : int, optional (default=None)
        要保留的主成分数量
    
    batch_size : int, optional (default=None)
        每批数据的样本数量
    
    Attributes:
    -----------
    components_ : ndarray, shape (n_components, n_features)
        主成分
    
    explained_variance_ : ndarray, shape (n_components,)
        每个主成分解释的方差
    
    explained_variance_ratio_ : ndarray, shape (n_components,)
        每个主成分解释的方差比例
    
    singular_values_ : ndarray, shape (n_components,)
        奇异值
    
    mean_ : ndarray, shape (n_features,)
        训练数据的均值
    
    n_samples_seen_ : int
        已处理的样本总数
    """
    
    def __init__(self, n_components: Optional[int] = None, 
                 batch_size: Optional[int] = None):
        self.n_components = n_components
        self.batch_size = batch_size
        
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.var_ = None
        self.n_samples_seen_ = 0
        self.n_features_ = None
        self.n_components_ = None
    
    def partial_fit(self, X: np.ndarray) -> 'IncrementalPCA':
        """
        增量拟合PCA模型（处理一批数据）
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features)
            当前批次的数据
        
        Returns:
        --------
        self : object
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        
        if self.n_features_ is None:
            self.n_features_ = n_features
            if self.n_components is None:
                self.n_components_ = n_features
            else:
                self.n_components_ = min(self.n_components, n_features)
        
        # 更新均值和方差（Welford's online algorithm）
        if self.n_samples_seen_ == 0:
            self.mean_ = np.zeros(n_features)
            self.var_ = np.zeros(n_features)
        
        # 更新统计量
        for i in range(n_samples):
            self.n_samples_seen_ += 1
            delta = X[i] - self.mean_
            self.mean_ += delta / self.n_samples_seen_
            delta2 = X[i] - self.mean_
            self.var_ += delta * delta2
        
        # 中心化当前批次
        X_centered = X - self.mean_
        
        # 如果是第一批，直接SVD
        if self.components_ is None:
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            self.components_ = Vt[:self.n_components_]
            self.singular_values_ = S[:self.n_components_]
        else:
            # 增量更新（简化版本，完整实现需要更复杂的矩阵更新）
            # 这里使用重新计算的方式
            if self.n_samples_seen_ >= n_features:
                cov_matrix = self.var_ / (self.n_samples_seen_ - 1)
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                idx = np.argsort(eigenvalues)[::-1]
                eigenvalues = eigenvalues[idx][:self.n_components_]
                eigenvectors = eigenvectors[:, idx][:, :self.n_components_]
                
                self.components_ = eigenvectors.T
                self.explained_variance_ = eigenvalues
                self.singular_values_ = np.sqrt(eigenvalues * (self.n_samples_seen_ - 1))
        
        # 更新解释方差
        if self.n_samples_seen_ >= n_features:
            total_var = np.sum(self.var_) / (self.n_samples_seen_ - 1)
            if total_var > 0:
                self.explained_variance_ratio_ = self.explained_variance_ / total_var
        
        return self
    
    def fit(self, X: np.ndarray) -> 'IncrementalPCA':
        """
        批量拟合PCA模型
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features)
            训练数据
        
        Returns:
        --------
        self : object
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        
        if self.batch_size is None:
            batch_size = 5 * X.shape[1]
        else:
            batch_size = self.batch_size
        
        # 分批处理
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            self.partial_fit(X[start_idx:end_idx])
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """将数据投影到主成分空间"""
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)
    
    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """将降维后的数据重建回原始空间"""
        return np.dot(X_transformed, self.components_) + self.mean_


class KernelPCA:
    """
    核主成分分析 (Kernel PCA)
    
    通过核技巧将数据映射到高维空间，然后在该空间中进行PCA，
    可以提取非线性特征。
    
    Parameters:
    -----------
    n_components : int
        要保留的主成分数量
    
    kernel : str or callable, optional (default='rbf')
        核函数类型
        - 'linear': 线性核 k(x,y) = x·y
        - 'poly': 多项式核 k(x,y) = (γx·y + c)^d
        - 'rbf': 高斯核 k(x,y) = exp(-γ||x-y||²)
        - callable: 自定义核函数
    
    gamma : float, optional (default=None)
        RBF、poly核的参数，默认为1/n_features
    
    degree : int, optional (default=3)
        多项式核的次数
    
    coef0 : float, optional (default=1)
        多项式核的常数项
    
    Attributes:
    -----------
    lambdas_ : ndarray, shape (n_components,)
        特征值
    
    alphas_ : ndarray, shape (n_samples, n_components)
        核空间中的特征向量
    
    X_fit_ : ndarray, shape (n_samples, n_features)
        训练数据
    """
    
    def __init__(self, n_components: int, kernel: str = 'rbf',
                 gamma: Optional[float] = None, degree: int = 3, coef0: float = 1.0):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        
        self.lambdas_ = None
        self.alphas_ = None
        self.X_fit_ = None
        self.K_fit_ = None
    
    def _get_kernel(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算核矩阵
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples_X, n_features)
        Y : ndarray, shape (n_samples_Y, n_features), optional
        
        Returns:
        --------
        K : ndarray, shape (n_samples_X, n_samples_Y)
            核矩阵
        """
        if Y is None:
            Y = X
        
        if self.kernel == 'linear':
            return np.dot(X, Y.T)
        
        elif self.kernel == 'poly':
            gamma = self.gamma if self.gamma is not None else 1.0 / X.shape[1]
            return (gamma * np.dot(X, Y.T) + self.coef0) ** self.degree
        
        elif self.kernel == 'rbf':
            gamma = self.gamma if self.gamma is not None else 1.0 / X.shape[1]
            # ||x-y||² = ||x||² + ||y||² - 2x·y
            X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
            Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
            K = X_norm + Y_norm - 2 * np.dot(X, Y.T)
            K = np.exp(-gamma * K)
            return K
        
        elif callable(self.kernel):
            return self.kernel(X, Y)
        
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def fit(self, X: np.ndarray) -> 'KernelPCA':
        """
        拟合Kernel PCA模型
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features)
            训练数据
        
        Returns:
        --------
        self : object
        """
        X = np.asarray(X, dtype=np.float64)
        self.X_fit_ = X
        n_samples = X.shape[0]
        
        # 计算核矩阵
        K = self._get_kernel(X)
        
        # 中心化核矩阵
        # K_centered = K - 1_n*K - K*1_n + 1_n*K*1_n
        one_n = np.ones((n_samples, n_samples)) / n_samples
        K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
        
        self.K_fit_ = K_centered
        
        # 特征分解
        eigenvalues, eigenvectors = np.linalg.eigh(K_centered)
        
        # 按特征值降序排列
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 保留前n_components个
        self.lambdas_ = eigenvalues[:self.n_components]
        self.alphas_ = eigenvectors[:, :self.n_components]
        
        # 归一化特征向量: α_i / sqrt(λ_i)
        self.alphas_ = self.alphas_ / np.sqrt(self.lambdas_)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        将数据投影到核主成分空间
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features)
            输入数据
        
        Returns:
        --------
        X_transformed : ndarray, shape (n_samples, n_components)
            投影后的数据
        """
        # 计算测试数据与训练数据的核矩阵
        K = self._get_kernel(X, self.X_fit_)
        
        # 中心化
        n_train = self.X_fit_.shape[0]
        one_n = np.ones((n_train, n_train)) / n_train
        K_pred_centered = K - np.mean(K, axis=1, keepdims=True) - np.mean(self.K_fit_, axis=0, keepdims=True) + np.mean(self.K_fit_)
        
        # 投影: X_transformed = K_centered * alphas
        X_transformed = K_pred_centered @ self.alphas_
        
        return X_transformed
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """拟合模型并转换数据"""
        self.fit(X)
        return self.K_fit_ @ self.alphas_


def demo1_basic_pca():
    """
    示例1: 基础PCA降维
    展示标准PCA在2D数据上的效果，包括主成分方向和投影结果
    """
    print("=" * 70)
    print("示例1: 基础PCA降维")
    print("=" * 70)
    
    # 生成2D相关数据
    np.random.seed(42)
    mean = [0, 0]
    cov = [[3, 2], [2, 2]]
    n_samples = 300
    X = np.random.multivariate_normal(mean, cov, n_samples)
    
    # PCA降维到1D
    pca = PCA(n_components=1, method='svd')
    X_pca = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(X_pca)
    
    # 打印结果
    print(f"\n原始数据形状: {X.shape}")
    print(f"降维后数据形状: {X_pca.shape}")
    print(f"重建后数据形状: {X_reconstructed.shape}")
    print(f"\n主成分: \n{pca.components_}")
    print(f"\n解释方差: {pca.explained_variance_}")
    print(f"解释方差比例: {pca.explained_variance_ratio_}")
    print(f"累积解释方差: {np.sum(pca.explained_variance_ratio_):.4f}")
    print(f"\n重建误差: {pca.reconstruction_error(X):.6f}")
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 原始数据与主成分方向
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], alpha=0.5, s=20)
    
    # 绘制主成分方向
    mean_point = pca.mean_
    pc1 = pca.components_[0]
    scale = 3 * np.sqrt(pca.explained_variance_[0])
    ax.arrow(mean_point[0], mean_point[1], 
             pc1[0] * scale, pc1[1] * scale,
             head_width=0.3, head_length=0.3, fc='red', ec='red', linewidth=2,
             label=f'PC1 (解释{pca.explained_variance_ratio_[0]:.1%}方差)')
    
    ax.scatter(mean_point[0], mean_point[1], c='red', s=100, marker='x', linewidths=3)
    ax.set_xlabel('特征1')
    ax.set_ylabel('特征2')
    ax.set_title('原始数据与第一主成分')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # 投影到主成分空间
    ax = axes[1]
    ax.scatter(X_pca, np.zeros_like(X_pca), alpha=0.5, s=20)
    ax.set_xlabel('第一主成分')
    ax.set_ylabel('')
    ax.set_title('投影到一维主成分空间')
    ax.set_yticks([])
    ax.grid(True, alpha=0.3)
    
    # 重建后的数据
    ax = axes[2]
    ax.scatter(X[:, 0], X[:, 1], alpha=0.3, s=20, label='原始数据')
    ax.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], alpha=0.3, s=20, label='重建数据')
    
    # 绘制投影线
    for i in range(0, n_samples, 10):
        ax.plot([X[i, 0], X_reconstructed[i, 0]], 
               [X[i, 1], X_reconstructed[i, 1]], 
               'k-', alpha=0.2, linewidth=0.5)
    
    ax.set_xlabel('特征1')
    ax.set_ylabel('特征2')
    ax.set_title('数据重建（投影误差）')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig('pca/pca_basic_demo.png', dpi=300, bbox_inches='tight')
    print("\n图像已保存: pca/pca_basic_demo.png")
    plt.close()


def demo2_dimensionality_reduction():
    """
    示例2: 高维数据降维与可视化
    展示如何选择最佳主成分数量（肘部法则和累积方差）
    """
    print("\n" + "=" * 70)
    print("示例2: 高维数据降维与可视化")
    print("=" * 70)
    
    # 生成高维数据
    np.random.seed(42)
    n_samples = 200
    n_features = 50
    
    # 创建有内在低维结构的数据
    # 真实数据在3个隐藏因子上
    latent_factors = np.random.randn(n_samples, 3)
    # 通过随机投影矩阵生成高维数据
    projection = np.random.randn(3, n_features)
    X = latent_factors @ projection
    # 添加噪声
    X += 0.5 * np.random.randn(n_samples, n_features)
    
    # 生成分类标签（用于可视化）
    y = np.zeros(n_samples)
    y[latent_factors[:, 0] > 0] = 1
    y[(latent_factors[:, 0] <= 0) & (latent_factors[:, 1] > 0)] = 2
    
    print(f"\n原始数据形状: {X.shape}")
    
    # 完整PCA分析
    pca_full = PCA(n_components=None, method='svd')
    pca_full.fit(X)
    
    # 打印前10个主成分的信息
    print(f"\n前10个主成分的解释方差比例:")
    for i in range(min(10, len(pca_full.explained_variance_ratio_))):
        print(f"  PC{i+1}: {pca_full.explained_variance_ratio_[i]:.4f}")
    
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    print(f"\n前3个主成分累积解释方差: {cumsum[2]:.4f}")
    print(f"前5个主成分累积解释方差: {cumsum[4]:.4f}")
    print(f"前10个主成分累积解释方差: {cumsum[9]:.4f}")
    
    # 降维到3D用于可视化
    pca_3d = PCA(n_components=3, method='svd')
    X_pca_3d = pca_3d.fit_transform(X)
    
    # 可视化
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 方差解释比例
    ax1 = plt.subplot(2, 3, 1)
    n_components_plot = min(20, len(pca_full.explained_variance_ratio_))
    ax1.bar(range(1, n_components_plot + 1), 
            pca_full.explained_variance_ratio_[:n_components_plot])
    ax1.set_xlabel('主成分编号')
    ax1.set_ylabel('解释方差比例')
    ax1.set_title('各主成分的方差贡献')
    ax1.grid(True, alpha=0.3)
    
    # 2. 累积方差解释比例（肘部法则）
    ax2 = plt.subplot(2, 3, 2)
    cumsum_plot = np.cumsum(pca_full.explained_variance_ratio_[:n_components_plot])
    ax2.plot(range(1, n_components_plot + 1), cumsum_plot, 'o-', linewidth=2)
    ax2.axhline(y=0.95, color='r', linestyle='--', label='95%阈值')
    ax2.axhline(y=0.90, color='orange', linestyle='--', label='90%阈值')
    ax2.set_xlabel('主成分数量')
    ax2.set_ylabel('累积解释方差比例')
    ax2.set_title('累积方差贡献（肘部法则）')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 找到达到95%方差的主成分数量
    n_95 = np.argmax(cumsum >= 0.95) + 1
    ax2.axvline(x=n_95, color='r', linestyle=':', alpha=0.5)
    ax2.text(n_95, 0.5, f'n={n_95}', ha='center')
    
    # 3. 奇异值谱（对数尺度）
    ax3 = plt.subplot(2, 3, 3)
    singular_values_plot = pca_full.singular_values_[:n_components_plot]
    ax3.semilogy(range(1, len(singular_values_plot) + 1), singular_values_plot, 'o-')
    ax3.set_xlabel('主成分编号')
    ax3.set_ylabel('奇异值（对数尺度）')
    ax3.set_title('奇异值谱')
    ax3.grid(True, alpha=0.3)
    
    # 4. 2D投影（PC1 vs PC2）
    ax4 = plt.subplot(2, 3, 4)
    scatter = ax4.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], c=y, cmap='viridis', 
                         alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    ax4.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})')
    ax4.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})')
    ax4.set_title('2D投影 (PC1 vs PC2)')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='类别')
    
    # 5. 2D投影（PC1 vs PC3）
    ax5 = plt.subplot(2, 3, 5)
    scatter = ax5.scatter(X_pca_3d[:, 0], X_pca_3d[:, 2], c=y, cmap='viridis', 
                         alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    ax5.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})')
    ax5.set_ylabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})')
    ax5.set_title('2D投影 (PC1 vs PC3)')
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax5, label='类别')
    
    # 6. 3D投影
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    scatter = ax6.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], 
                         c=y, cmap='viridis', alpha=0.6, s=30, 
                         edgecolors='k', linewidth=0.5)
    ax6.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})')
    ax6.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})')
    ax6.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})')
    ax6.set_title('3D投影')
    plt.colorbar(scatter, ax=ax6, label='类别', shrink=0.8)
    
    plt.tight_layout()
    plt.savefig('pca/pca_dimensionality_reduction.png', dpi=300, bbox_inches='tight')
    print("\n图像已保存: pca/pca_dimensionality_reduction.png")
    plt.close()


def demo3_svd_vs_eigen():
    """
    示例3: SVD方法 vs 特征分解方法
    比较两种PCA实现方法的结果和性能
    """
    print("\n" + "=" * 70)
    print("示例3: SVD方法 vs 特征分解方法比较")
    print("=" * 70)
    
    # 生成测试数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 100
    X = np.random.randn(n_samples, n_features)
    
    # 添加一些结构
    X[:, :10] += np.random.randn(n_samples, 1) @ np.random.randn(1, 10)
    
    print(f"\n数据形状: {X.shape}")
    
    # SVD方法
    import time
    start_time = time.time()
    pca_svd = PCA(n_components=10, method='svd')
    X_svd = pca_svd.fit_transform(X)
    time_svd = time.time() - start_time
    
    # 特征分解方法
    start_time = time.time()
    pca_eigen = PCA(n_components=10, method='eigen')
    X_eigen = pca_eigen.fit_transform(X)
    time_eigen = time.time() - start_time
    
    # 比较结果
    print(f"\n计算时间:")
    print(f"  SVD方法: {time_svd:.4f}秒")
    print(f"  特征分解方法: {time_eigen:.4f}秒")
    print(f"  速度提升: {time_eigen/time_svd:.2f}x")
    
    print(f"\n解释方差比例 (SVD方法):")
    for i in range(5):
        print(f"  PC{i+1}: {pca_svd.explained_variance_ratio_[i]:.6f}")
    
    print(f"\n解释方差比例 (特征分解方法):")
    for i in range(5):
        print(f"  PC{i+1}: {pca_eigen.explained_variance_ratio_[i]:.6f}")
    
    # 检查主成分方向是否一致（允许符号差异）
    print(f"\n主成分一致性检查:")
    for i in range(5):
        # 计算相关系数的绝对值（忽略符号）
        corr = np.abs(np.corrcoef(pca_svd.components_[i], pca_eigen.components_[i])[0, 1])
        print(f"  PC{i+1}相关系数: {corr:.6f}")
    
    # 检查转换结果是否一致
    diff = np.abs(np.abs(X_svd) - np.abs(X_eigen))
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    print(f"\n转换结果差异:")
    print(f"  最大差异: {max_diff:.10f}")
    print(f"  平均差异: {mean_diff:.10f}")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 解释方差比例对比
    ax = axes[0, 0]
    x = np.arange(10)
    width = 0.35
    ax.bar(x - width/2, pca_svd.explained_variance_ratio_, width, 
           label='SVD方法', alpha=0.8)
    ax.bar(x + width/2, pca_eigen.explained_variance_ratio_, width, 
           label='特征分解方法', alpha=0.8)
    ax.set_xlabel('主成分编号')
    ax.set_ylabel('解释方差比例')
    ax.set_title('方差解释比例对比')
    ax.set_xticks(x)
    ax.set_xticklabels([f'PC{i+1}' for i in range(10)], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 主成分方向相似度
    ax = axes[0, 1]
    similarities = []
    for i in range(10):
        corr = np.abs(np.corrcoef(pca_svd.components_[i], pca_eigen.components_[i])[0, 1])
        similarities.append(corr)
    ax.bar(range(10), similarities, alpha=0.8, color='green')
    ax.axhline(y=0.999, color='r', linestyle='--', label='0.999阈值')
    ax.set_xlabel('主成分编号')
    ax.set_ylabel('相关系数')
    ax.set_title('主成分方向一致性')
    ax.set_xticks(range(10))
    ax.set_xticklabels([f'PC{i+1}' for i in range(10)], rotation=45)
    ax.set_ylim([0.99, 1.001])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 转换结果对比（前2个主成分）
    ax = axes[1, 0]
    sample_indices = np.random.choice(n_samples, 200, replace=False)
    ax.scatter(X_svd[sample_indices, 0], X_svd[sample_indices, 1], 
              alpha=0.5, s=20, label='SVD方法')
    ax.scatter(X_eigen[sample_indices, 0], X_eigen[sample_indices, 1], 
              alpha=0.5, s=20, marker='x', label='特征分解方法')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('转换结果对比（前2个主成分）')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 计算时间对比
    ax = axes[1, 1]
    methods = ['SVD\n方法', '特征分解\n方法']
    times = [time_svd, time_eigen]
    colors = ['#1f77b4', '#ff7f0e']
    bars = ax.bar(methods, times, color=colors, alpha=0.8)
    ax.set_ylabel('计算时间（秒）')
    ax.set_title('计算性能对比')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上标注数值
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.4f}s',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('pca/pca_svd_vs_eigen.png', dpi=300, bbox_inches='tight')
    print("\n图像已保存: pca/pca_svd_vs_eigen.png")
    plt.close()


def demo4_kernel_pca():
    """
    示例4: 核PCA处理非线性数据
    展示核PCA在非线性数据上的优势
    """
    print("\n" + "=" * 70)
    print("示例4: 核PCA处理非线性数据")
    print("=" * 70)
    
    # 生成同心圆数据（非线性可分）
    np.random.seed(42)
    n_samples = 400
    
    # 内圈
    r1 = 1.0
    theta1 = np.random.uniform(0, 2*np.pi, n_samples // 2)
    X1 = np.column_stack([
        r1 * np.cos(theta1) + 0.1 * np.random.randn(n_samples // 2),
        r1 * np.sin(theta1) + 0.1 * np.random.randn(n_samples // 2)
    ])
    
    # 外圈
    r2 = 3.0
    theta2 = np.random.uniform(0, 2*np.pi, n_samples // 2)
    X2 = np.column_stack([
        r2 * np.cos(theta2) + 0.1 * np.random.randn(n_samples // 2),
        r2 * np.sin(theta2) + 0.1 * np.random.randn(n_samples // 2)
    ])
    
    X = np.vstack([X1, X2])
    y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])
    
    print(f"\n数据形状: {X.shape}")
    print(f"类别分布: {np.bincount(y.astype(int))}")
    
    # 标准PCA
    pca = PCA(n_components=2, method='svd')
    X_pca = pca.fit_transform(X)
    
    # 核PCA (RBF核)
    kpca_rbf = KernelPCA(n_components=2, kernel='rbf', gamma=0.5)
    X_kpca_rbf = kpca_rbf.fit_transform(X)
    
    # 核PCA (多项式核)
    kpca_poly = KernelPCA(n_components=2, kernel='poly', degree=3, gamma=0.5)
    X_kpca_poly = kpca_poly.fit_transform(X)
    
    print(f"\n标准PCA前2个主成分解释方差: {pca.explained_variance_ratio_}")
    print(f"核PCA (RBF)特征值: {kpca_rbf.lambdas_}")
    print(f"核PCA (Poly)特征值: {kpca_poly.lambdas_}")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # 原始数据
    ax = axes[0, 0]
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', 
                        alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    ax.set_xlabel('特征1')
    ax.set_ylabel('特征2')
    ax.set_title('原始数据（同心圆）')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='类别')
    
    # 标准PCA
    ax = axes[0, 1]
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', 
                        alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title('标准PCA（线性投影）')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='类别')
    
    # 核PCA (RBF)
    ax = axes[1, 0]
    scatter = ax.scatter(X_kpca_rbf[:, 0], X_kpca_rbf[:, 1], c=y, cmap='coolwarm', 
                        alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    ax.set_xlabel('核PC1')
    ax.set_ylabel('核PC2')
    ax.set_title('核PCA - RBF核（γ=0.5）')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='类别')
    
    # 核PCA (Poly)
    ax = axes[1, 1]
    scatter = ax.scatter(X_kpca_poly[:, 0], X_kpca_poly[:, 1], c=y, cmap='coolwarm', 
                        alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    ax.set_xlabel('核PC1')
    ax.set_ylabel('核PC2')
    ax.set_title('核PCA - 多项式核（度=3）')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='类别')
    
    plt.tight_layout()
    plt.savefig('pca/pca_kernel_comparison.png', dpi=300, bbox_inches='tight')
    print("\n图像已保存: pca/pca_kernel_comparison.png")
    plt.close()


def demo5_image_compression():
    """
    示例5: 图像压缩应用
    使用PCA进行图像压缩，展示不同压缩比下的效果
    """
    print("\n" + "=" * 70)
    print("示例5: PCA图像压缩应用")
    print("=" * 70)
    
    # 生成合成图像（带有一些结构）
    np.random.seed(42)
    img_size = 64
    
    # 创建有结构的图像
    x = np.linspace(-3, 3, img_size)
    y = np.linspace(-3, 3, img_size)
    X_grid, Y_grid = np.meshgrid(x, y)
    
    # 多个高斯分布的组合
    image = (np.exp(-(X_grid**2 + Y_grid**2) / 2) * 255 + 
             np.exp(-((X_grid-1)**2 + (Y_grid-1)**2) / 1) * 128 +
             np.exp(-((X_grid+1)**2 + (Y_grid+1)**2) / 1) * 128)
    
    # 归一化到0-255
    image = (image - image.min()) / (image.max() - image.min()) * 255
    image = image.astype(np.uint8)
    
    print(f"\n原始图像形状: {image.shape}")
    print(f"原始图像大小: {image.size} 像素 = {image.nbytes} 字节")
    
    # 测试不同的压缩比
    n_components_list = [1, 2, 4, 8, 16, 32]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    # 原始图像
    ax = axes[0]
    im = ax.imshow(image, cmap='gray')
    ax.set_title('原始图像')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 不同压缩比的结果
    for idx, n_comp in enumerate(n_components_list):
        # PCA压缩
        pca = PCA(n_components=n_comp, method='svd')
        image_transformed = pca.fit_transform(image)
        image_reconstructed = pca.inverse_transform(image_transformed)
        
        # 确保在有效范围内
        image_reconstructed = np.clip(image_reconstructed, 0, 255)
        
        # 计算压缩比和误差
        original_size = image.size
        compressed_size = (image_transformed.size + pca.components_.size + 
                          pca.mean_.size)
        compression_ratio = original_size / compressed_size
        
        # MSE和PSNR
        mse = np.mean((image - image_reconstructed) ** 2)
        psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
        
        # 方差解释
        variance_explained = np.sum(pca.explained_variance_ratio_)
        
        print(f"\n主成分数={n_comp}:")
        print(f"  压缩比: {compression_ratio:.2f}x")
        print(f"  MSE: {mse:.2f}")
        print(f"  PSNR: {psnr:.2f} dB")
        print(f"  方差解释: {variance_explained:.2%}")
        
        # 绘制重建图像
        ax = axes[idx + 1]
        im = ax.imshow(image_reconstructed, cmap='gray', vmin=0, vmax=255)
        ax.set_title(f'k={n_comp}\n'
                    f'压缩比={compression_ratio:.1f}x\n'
                    f'PSNR={psnr:.1f}dB\n'
                    f'方差={variance_explained:.1%}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 隐藏最后一个子图
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig('pca/pca_image_compression.png', dpi=300, bbox_inches='tight')
    print("\n图像已保存: pca/pca_image_compression.png")
    plt.close()
    
    # 绘制压缩性能分析图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 测试更多压缩比
    n_components_range = range(1, min(img_size, 40))
    compression_ratios = []
    psnr_values = []
    variance_explained_values = []
    
    for n_comp in n_components_range:
        pca = PCA(n_components=n_comp, method='svd')
        image_transformed = pca.fit_transform(image)
        image_reconstructed = pca.inverse_transform(image_transformed)
        image_reconstructed = np.clip(image_reconstructed, 0, 255)
        
        # 压缩比
        original_size = image.size
        compressed_size = (image_transformed.size + pca.components_.size + 
                          pca.mean_.size)
        compression_ratio = original_size / compressed_size
        compression_ratios.append(compression_ratio)
        
        # PSNR
        mse = np.mean((image - image_reconstructed) ** 2)
        psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
        psnr_values.append(psnr)
        
        # 方差解释
        variance_explained_values.append(np.sum(pca.explained_variance_ratio_))
    
    # 压缩比 vs 主成分数
    ax = axes[0]
    ax.plot(n_components_range, compression_ratios, 'o-', linewidth=2)
    ax.set_xlabel('主成分数量')
    ax.set_ylabel('压缩比')
    ax.set_title('压缩比 vs 主成分数量')
    ax.grid(True, alpha=0.3)
    
    # PSNR vs 主成分数
    ax = axes[1]
    ax.plot(n_components_range, psnr_values, 'o-', linewidth=2, color='green')
    ax.set_xlabel('主成分数量')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('图像质量 vs 主成分数量')
    ax.grid(True, alpha=0.3)
    
    # 方差解释 vs 主成分数
    ax = axes[2]
    ax.plot(n_components_range, variance_explained_values, 'o-', linewidth=2, color='red')
    ax.axhline(y=0.95, color='gray', linestyle='--', label='95%阈值')
    ax.axhline(y=0.99, color='gray', linestyle=':', label='99%阈值')
    ax.set_xlabel('主成分数量')
    ax.set_ylabel('累积方差解释比例')
    ax.set_title('方差保留 vs 主成分数量')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pca/pca_compression_analysis.png', dpi=300, bbox_inches='tight')
    print("图像已保存: pca/pca_compression_analysis.png")
    plt.close()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("主成分分析 (PCA) - 完整演示")
    print("=" * 70)
    
    # 运行所有示例
    demo1_basic_pca()
    demo2_dimensionality_reduction()
    demo3_svd_vs_eigen()
    demo4_kernel_pca()
    demo5_image_compression()
    
    print("\n" + "=" * 70)
    print("所有演示完成！")
    print("=" * 70)
    print("\n生成的文件:")
    print("  1. pca/pca_basic_demo.png - 基础PCA降维演示")
    print("  2. pca/pca_dimensionality_reduction.png - 高维数据降维")
    print("  3. pca/pca_svd_vs_eigen.png - SVD vs 特征分解对比")
    print("  4. pca/pca_kernel_comparison.png - 核PCA演示")
    print("  5. pca/pca_image_compression.png - 图像压缩应用")
    print("  6. pca/pca_compression_analysis.png - 压缩性能分析")
    print("\n" + "=" * 70)
