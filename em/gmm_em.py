"""
高斯混合模型的EM算法 (Gaussian Mixture Model with EM)
=====================================================

高斯混合模型(GMM)是最常用的混合模型之一，使用EM算法进行参数估计。

理论基础：
---------
模型定义：
  P(x) = Σ_{k=1}^K π_k * N(x | μ_k, Σ_k)

其中：
- K: 混合成分数量
- π_k: 第k个成分的混合系数（权重），Σ_k π_k = 1
- μ_k: 第k个高斯分布的均值向量
- Σ_k: 第k个高斯分布的协方差矩阵
- N(x|μ,Σ): 多元高斯分布

EM算法步骤：
-----------
E步：计算responsibility（后验概率）
  γ_{ik} = P(z_i=k | x_i, θ) 
         = π_k * N(x_i | μ_k, Σ_k) / Σ_j π_j * N(x_i | μ_j, Σ_j)

M步：更新参数
  N_k = Σ_i γ_{ik}
  π_k = N_k / N
  μ_k = (1/N_k) * Σ_i γ_{ik} * x_i
  Σ_k = (1/N_k) * Σ_i γ_{ik} * (x_i - μ_k)(x_i - μ_k)^T

协方差类型：
-----------
1. full: 完整协方差矩阵（各向异性）
2. tied: 所有成分共享同一协方差矩阵
3. diag: 对角协方差矩阵（各维度独立）
4. spherical: 球形协方差（σ²I）

应用：
-----
- 聚类分析
- 密度估计
- 异常检测
- 数据生成
- 图像分割

本文件实现：
-----------
1. 完整的多维GMM-EM算法
2. 多种协方差类型
3. 初始化方法（K-means++）
4. 模型选择（BIC、AIC）
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Optional, Literal
from scipy import linalg

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class GaussianMixtureEM:
    """
    高斯混合模型的EM算法
    
    Parameters:
    -----------
    n_components : int
        混合成分数量
    
    covariance_type : str
        协方差类型
        - 'full': 完整协方差矩阵
        - 'tied': 共享协方差矩阵
        - 'diag': 对角协方差矩阵
        - 'spherical': 球形协方差
    
    max_iter : int
        最大迭代次数
    
    tol : float
        收敛阈值
    
    reg_covar : float
        协方差正则化参数（防止奇异）
    
    init_method : str
        初始化方法
        - 'kmeans': 使用K-means初始化
        - 'random': 随机初始化
    
    Attributes:
    -----------
    weights_ : ndarray, shape (n_components,)
        混合系数
    
    means_ : ndarray, shape (n_components, n_features)
        均值向量
    
    covariances_ : ndarray
        协方差矩阵（形状取决于covariance_type）
    
    converged_ : bool
        是否收敛
    
    n_iter_ : int
        实际迭代次数
    
    log_likelihoods_ : list
        每次迭代的对数似然
    """
    
    def __init__(self, n_components: int = 2, 
                 covariance_type: Literal['full', 'tied', 'diag', 'spherical'] = 'full',
                 max_iter: int = 100, tol: float = 1e-6, 
                 reg_covar: float = 1e-6, init_method: str = 'kmeans',
                 verbose: bool = True):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.init_method = init_method
        self.verbose = verbose
        
        # 模型参数
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.precisions_cholesky_ = None  # 用于加速计算
        
        # 拟合信息
        self.converged_ = False
        self.n_iter_ = 0
        self.log_likelihoods_ = []
    
    def _initialize_parameters(self, X: np.ndarray, random_state: Optional[int] = None):
        """
        初始化参数
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features)
            训练数据
        """
        n_samples, n_features = X.shape
        
        if random_state is not None:
            np.random.seed(random_state)
        
        if self.init_method == 'kmeans':
            # 使用简单的K-means++初始化
            self.means_ = self._kmeans_plusplus_init(X, self.n_components)
        else:
            # 随机选择样本作为初始均值
            indices = np.random.choice(n_samples, self.n_components, replace=False)
            self.means_ = X[indices].copy()
        
        # 初始化权重（均匀分布）
        self.weights_ = np.ones(self.n_components) / self.n_components
        
        # 初始化协方差
        self._initialize_covariances(X)
    
    def _kmeans_plusplus_init(self, X: np.ndarray, n_clusters: int) -> np.ndarray:
        """K-means++初始化"""
        n_samples = X.shape[0]
        centers = np.zeros((n_clusters, X.shape[1]))
        
        # 随机选择第一个中心
        centers[0] = X[np.random.randint(n_samples)]
        
        # 选择剩余的中心
        for k in range(1, n_clusters):
            # 计算每个点到最近中心的距离
            distances = np.min([np.sum((X - centers[j])**2, axis=1) 
                               for j in range(k)], axis=0)
            
            # 以距离的平方为概率选择下一个中心
            probs = distances / distances.sum()
            cumprobs = np.cumsum(probs)
            r = np.random.rand()
            
            for i, p in enumerate(cumprobs):
                if r < p:
                    centers[k] = X[i]
                    break
        
        return centers
    
    def _initialize_covariances(self, X: np.ndarray):
        """初始化协方差矩阵"""
        n_samples, n_features = X.shape
        
        if self.covariance_type == 'full':
            # 每个成分有自己的完整协方差矩阵
            self.covariances_ = np.array([np.cov(X.T) + self.reg_covar * np.eye(n_features)
                                         for _ in range(self.n_components)])
        
        elif self.covariance_type == 'tied':
            # 所有成分共享一个协方差矩阵
            self.covariances_ = np.cov(X.T) + self.reg_covar * np.eye(n_features)
        
        elif self.covariance_type == 'diag':
            # 对角协方差矩阵
            self.covariances_ = np.array([np.var(X, axis=0) + self.reg_covar
                                         for _ in range(self.n_components)])
        
        elif self.covariance_type == 'spherical':
            # 球形协方差（各向同性）
            self.covariances_ = np.array([np.mean(np.var(X, axis=0)) + self.reg_covar
                                         for _ in range(self.n_components)])
    
    def _compute_precision_cholesky(self):
        """
        计算精度矩阵的Cholesky分解
        用于加速高斯概率密度的计算
        """
        if self.covariance_type == 'full':
            self.precisions_cholesky_ = np.array([
                linalg.cholesky(linalg.pinv(cov), lower=True)
                for cov in self.covariances_
            ])
        elif self.covariance_type == 'tied':
            self.precisions_cholesky_ = linalg.cholesky(
                linalg.pinv(self.covariances_), lower=True)
        elif self.covariance_type == 'diag':
            self.precisions_cholesky_ = 1.0 / np.sqrt(self.covariances_)
        elif self.covariance_type == 'spherical':
            self.precisions_cholesky_ = 1.0 / np.sqrt(self.covariances_)
    
    def _estimate_log_gaussian_prob(self, X: np.ndarray) -> np.ndarray:
        """
        计算每个样本属于每个成分的对数概率
        
        Returns:
        --------
        log_prob : ndarray, shape (n_samples, n_components)
            log N(x_i | μ_k, Σ_k)
        """
        n_samples, n_features = X.shape
        log_prob = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            mean = self.means_[k]
            
            if self.covariance_type == 'full':
                cov = self.covariances_[k]
                # 使用scipy.stats.multivariate_normal会更简单，但这里手动实现
                diff = X - mean
                log_det = np.linalg.slogdet(cov)[1]
                precision = np.linalg.inv(cov)
                mahalanobis = np.sum(diff @ precision * diff, axis=1)
                log_prob[:, k] = -0.5 * (n_features * np.log(2 * np.pi) + 
                                        log_det + mahalanobis)
            
            elif self.covariance_type == 'tied':
                diff = X - mean
                log_det = np.linalg.slogdet(self.covariances_)[1]
                precision = np.linalg.inv(self.covariances_)
                mahalanobis = np.sum(diff @ precision * diff, axis=1)
                log_prob[:, k] = -0.5 * (n_features * np.log(2 * np.pi) + 
                                        log_det + mahalanobis)
            
            elif self.covariance_type == 'diag':
                diff = X - mean
                var = self.covariances_[k]
                log_det = np.sum(np.log(var))
                mahalanobis = np.sum((diff ** 2) / var, axis=1)
                log_prob[:, k] = -0.5 * (n_features * np.log(2 * np.pi) + 
                                        log_det + mahalanobis)
            
            elif self.covariance_type == 'spherical':
                diff = X - mean
                var = self.covariances_[k]
                log_det = n_features * np.log(var)
                mahalanobis = np.sum(diff ** 2, axis=1) / var
                log_prob[:, k] = -0.5 * (n_features * np.log(2 * np.pi) + 
                                        log_det + mahalanobis)
        
        return log_prob
    
    def _e_step(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        E步：计算responsibilities和对数似然
        
        Returns:
        --------
        responsibilities : ndarray, shape (n_samples, n_components)
            后验概率 γ_{ik}
        log_likelihood : float
            对数似然
        """
        # 计算对数概率
        log_prob = self._estimate_log_gaussian_prob(X)
        
        # 加上混合系数的对数
        weighted_log_prob = log_prob + np.log(self.weights_)
        
        # 计算对数似然：log P(X|θ) = Σ_i log Σ_k π_k N(x_i|μ_k,Σ_k)
        log_likelihood = np.sum(self._logsumexp(weighted_log_prob, axis=1))
        
        # 计算responsibilities
        log_resp = weighted_log_prob - self._logsumexp(weighted_log_prob, axis=1, keepdims=True)
        responsibilities = np.exp(log_resp)
        
        return responsibilities, log_likelihood
    
    def _logsumexp(self, arr: np.ndarray, axis: Optional[int] = None, 
                   keepdims: bool = False) -> np.ndarray:
        """数值稳定的logsumexp"""
        max_val = np.max(arr, axis=axis, keepdims=True)
        result = max_val + np.log(np.sum(np.exp(arr - max_val), 
                                         axis=axis, keepdims=True))
        if not keepdims and axis is not None:
            result = result.squeeze(axis=axis)
        return result
    
    def _m_step(self, X: np.ndarray, responsibilities: np.ndarray):
        """
        M步：更新参数
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features)
        responsibilities : ndarray, shape (n_samples, n_components)
        """
        n_samples, n_features = X.shape
        
        # 有效样本数
        n_k = responsibilities.sum(axis=0) + 10 * np.finfo(responsibilities.dtype).eps
        
        # 更新权重
        self.weights_ = n_k / n_samples
        
        # 更新均值
        self.means_ = (responsibilities.T @ X) / n_k[:, np.newaxis]
        
        # 更新协方差
        if self.covariance_type == 'full':
            self.covariances_ = np.zeros((self.n_components, n_features, n_features))
            for k in range(self.n_components):
                diff = X - self.means_[k]
                weighted_diff = responsibilities[:, k:k+1] * diff
                self.covariances_[k] = (weighted_diff.T @ diff) / n_k[k]
                self.covariances_[k] += self.reg_covar * np.eye(n_features)
        
        elif self.covariance_type == 'tied':
            self.covariances_ = np.zeros((n_features, n_features))
            for k in range(self.n_components):
                diff = X - self.means_[k]
                weighted_diff = responsibilities[:, k:k+1] * diff
                self.covariances_ += (weighted_diff.T @ diff)
            self.covariances_ /= n_samples
            self.covariances_ += self.reg_covar * np.eye(n_features)
        
        elif self.covariance_type == 'diag':
            self.covariances_ = np.zeros((self.n_components, n_features))
            for k in range(self.n_components):
                diff = X - self.means_[k]
                self.covariances_[k] = np.sum(responsibilities[:, k:k+1] * 
                                             diff ** 2, axis=0) / n_k[k]
                self.covariances_[k] += self.reg_covar
        
        elif self.covariance_type == 'spherical':
            self.covariances_ = np.zeros(self.n_components)
            for k in range(self.n_components):
                diff = X - self.means_[k]
                self.covariances_[k] = np.sum(responsibilities[:, k:k+1] * 
                                             np.sum(diff ** 2, axis=1, keepdims=True)) / \
                                      (n_k[k] * n_features)
                self.covariances_[k] += self.reg_covar
    
    def fit(self, X: np.ndarray, random_state: Optional[int] = None) -> 'GaussianMixtureEM':
        """
        拟合GMM模型
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features)
            训练数据
        
        Returns:
        --------
        self
        """
        # 初始化参数
        self._initialize_parameters(X, random_state)
        
        self.log_likelihoods_ = []
        self.converged_ = False
        
        if self.verbose:
            print(f"初始化完成，开始EM迭代...")
        
        for iteration in range(self.max_iter):
            # E步
            responsibilities, log_likelihood = self._e_step(X)
            self.log_likelihoods_.append(log_likelihood)
            
            # M步
            self._m_step(X, responsibilities)
            
            if self.verbose and (iteration % 10 == 0 or iteration < 5):
                print(f"Iteration {iteration}: Log-Likelihood = {log_likelihood:.6f}")
            
            # 检查收敛
            if iteration > 0:
                change = abs(self.log_likelihoods_[-1] - self.log_likelihoods_[-2])
                if change < self.tol:
                    self.converged_ = True
                    if self.verbose:
                        print(f"\n收敛于第 {iteration} 次迭代")
                        print(f"对数似然变化: {change:.8f} < {self.tol}")
                    break
        
        self.n_iter_ = len(self.log_likelihoods_)
        
        if not self.converged_ and self.verbose:
            print(f"\n警告: 未收敛（达到最大迭代次数 {self.max_iter}）")
        
        if self.verbose:
            print(f"最终对数似然: {self.log_likelihoods_[-1]:.6f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测聚类标签
        
        Returns:
        --------
        labels : ndarray, shape (n_samples,)
        """
        responsibilities, _ = self._e_step(X)
        return np.argmax(responsibilities, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测每个成分的概率
        
        Returns:
        --------
        probabilities : ndarray, shape (n_samples, n_components)
        """
        responsibilities, _ = self._e_step(X)
        return responsibilities
    
    def score(self, X: np.ndarray) -> float:
        """
        计算对数似然
        
        Returns:
        --------
        log_likelihood : float
        """
        _, log_likelihood = self._e_step(X)
        return log_likelihood
    
    def bic(self, X: np.ndarray) -> float:
        """
        计算BIC (Bayesian Information Criterion)
        BIC = -2 * log_likelihood + n_parameters * log(n_samples)
        
        越小越好
        """
        n_samples, n_features = X.shape
        
        # 计算参数数量
        if self.covariance_type == 'full':
            cov_params = self.n_components * n_features * (n_features + 1) / 2
        elif self.covariance_type == 'tied':
            cov_params = n_features * (n_features + 1) / 2
        elif self.covariance_type == 'diag':
            cov_params = self.n_components * n_features
        elif self.covariance_type == 'spherical':
            cov_params = self.n_components
        
        mean_params = self.n_components * n_features
        weight_params = self.n_components - 1
        n_parameters = int(mean_params + cov_params + weight_params)
        
        log_likelihood = self.score(X)
        return -2 * log_likelihood + n_parameters * np.log(n_samples)
    
    def aic(self, X: np.ndarray) -> float:
        """
        计算AIC (Akaike Information Criterion)
        AIC = -2 * log_likelihood + 2 * n_parameters
        
        越小越好
        """
        n_samples, n_features = X.shape
        
        # 计算参数数量（同BIC）
        if self.covariance_type == 'full':
            cov_params = self.n_components * n_features * (n_features + 1) / 2
        elif self.covariance_type == 'tied':
            cov_params = n_features * (n_features + 1) / 2
        elif self.covariance_type == 'diag':
            cov_params = self.n_components * n_features
        elif self.covariance_type == 'spherical':
            cov_params = self.n_components
        
        mean_params = self.n_components * n_features
        weight_params = self.n_components - 1
        n_parameters = int(mean_params + cov_params + weight_params)
        
        log_likelihood = self.score(X)
        return -2 * log_likelihood + 2 * n_parameters
    
    def sample(self, n_samples: int = 1, random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        从拟合的GMM中采样
        
        Parameters:
        -----------
        n_samples : int
            采样数量
        
        Returns:
        --------
        X : ndarray, shape (n_samples, n_features)
            采样数据
        labels : ndarray, shape (n_samples,)
            每个样本的成分标签
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        n_features = self.means_.shape[1]
        
        # 根据混合系数采样成分
        labels = np.random.choice(self.n_components, size=n_samples, p=self.weights_)
        
        X = np.zeros((n_samples, n_features))
        
        for k in range(self.n_components):
            mask = labels == k
            n_k = np.sum(mask)
            
            if n_k > 0:
                if self.covariance_type == 'full':
                    X[mask] = np.random.multivariate_normal(
                        self.means_[k], self.covariances_[k], n_k)
                elif self.covariance_type == 'tied':
                    X[mask] = np.random.multivariate_normal(
                        self.means_[k], self.covariances_, n_k)
                elif self.covariance_type == 'diag':
                    X[mask] = np.random.normal(
                        self.means_[k], np.sqrt(self.covariances_[k]), (n_k, n_features))
                elif self.covariance_type == 'spherical':
                    X[mask] = np.random.normal(
                        self.means_[k], np.sqrt(self.covariances_[k]), (n_k, n_features))
        
        return X, labels


def plot_covariance_ellipse(mean, cov, ax, n_std=2.0, **kwargs):
    """
    绘制协方差椭圆
    
    Parameters:
    -----------
    mean : array-like, shape (2,)
        均值
    cov : array-like, shape (2, 2)
        协方差矩阵
    ax : matplotlib axis
    n_std : float
        标准差倍数
    """
    # 特征分解
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # 计算椭圆角度
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    
    # 椭圆宽高
    width, height = 2 * n_std * np.sqrt(eigenvalues)
    
    # 绘制椭圆
    ellipse = Ellipse(mean, width, height, angle=angle, **kwargs)
    ax.add_patch(ellipse)


def demo1_2d_gmm():
    """
    示例1: 二维高斯混合模型
    """
    print("=" * 70)
    print("示例1: 二维高斯混合模型的EM算法")
    print("=" * 70)
    
    # 生成数据
    np.random.seed(42)
    n_samples = 300
    
    # 三个高斯分布
    mean1 = np.array([0, 0])
    cov1 = np.array([[1, 0.5], [0.5, 1]])
    
    mean2 = np.array([5, 5])
    cov2 = np.array([[1.5, -0.7], [-0.7, 1.5]])
    
    mean3 = np.array([5, -5])
    cov3 = np.array([[0.8, 0.3], [0.3, 0.8]])
    
    n1, n2, n3 = 100, 120, 80
    X1 = np.random.multivariate_normal(mean1, cov1, n1)
    X2 = np.random.multivariate_normal(mean2, cov2, n2)
    X3 = np.random.multivariate_normal(mean3, cov3, n3)
    
    X = np.vstack([X1, X2, X3])
    true_labels = np.concatenate([np.zeros(n1), np.ones(n2), 2*np.ones(n3)])
    
    # 打乱数据
    indices = np.random.permutation(n_samples)
    X = X[indices]
    true_labels = true_labels[indices]
    
    print(f"\n数据: {n_samples}个样本，3个成分")
    print(f"成分样本数: [{n1}, {n2}, {n3}]")
    
    # 拟合GMM
    print("\n" + "-" * 70)
    print("EM算法拟合:")
    print("-" * 70)
    
    gmm = GaussianMixtureEM(n_components=3, covariance_type='full', 
                           max_iter=100, verbose=True)
    gmm.fit(X, random_state=42)
    
    # 预测
    predicted_labels = gmm.predict(X)
    
    print(f"\n模型参数:")
    print(f"权重: {gmm.weights_}")
    print(f"均值:\n{gmm.means_}")
    
    # BIC和AIC
    print(f"\nBIC: {gmm.bic(X):.2f}")
    print(f"AIC: {gmm.aic(X):.2f}")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 原始数据
    ax = axes[0, 0]
    colors = ['red', 'blue', 'green']
    for k in range(3):
        mask = true_labels == k
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[k], alpha=0.6, s=30, 
                  label=f'真实成分{k}', edgecolors='k', linewidth=0.5)
    ax.set_xlabel('X₁')
    ax.set_ylabel('X₂')
    ax.set_title('原始数据（真实标签）')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # GMM拟合结果
    ax = axes[0, 1]
    for k in range(3):
        mask = predicted_labels == k
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[k], alpha=0.6, s=30, 
                  label=f'聚类{k}', edgecolors='k', linewidth=0.5)
    
    # 绘制均值和协方差椭圆
    for k in range(3):
        ax.scatter(gmm.means_[k, 0], gmm.means_[k, 1], 
                  c='black', s=200, marker='x', linewidths=3)
        plot_covariance_ellipse(gmm.means_[k], gmm.covariances_[k], ax,
                               n_std=2.0, edgecolor=colors[k], facecolor='none',
                               linewidth=2, linestyle='--')
    
    ax.set_xlabel('X₁')
    ax.set_ylabel('X₂')
    ax.set_title('GMM拟合结果（预测标签 + 2σ椭圆）')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # 对数似然曲线
    ax = axes[1, 0]
    ax.plot(range(len(gmm.log_likelihoods_)), gmm.log_likelihoods_, 
            'o-', linewidth=2, markersize=6)
    ax.set_xlabel('迭代次数')
    ax.set_ylabel('对数似然')
    ax.set_title('EM算法收敛过程')
    ax.grid(True, alpha=0.3)
    
    # 概率密度等高线
    ax = axes[1, 1]
    
    # 创建网格
    x_min, x_max = X[:, 0].min() - 2, X[:, 0].max() + 2
    y_min, y_max = X[:, 1].min() - 2, X[:, 1].max() + 2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # 计算概率密度
    X_grid = np.c_[xx.ravel(), yy.ravel()]
    log_prob = gmm._estimate_log_gaussian_prob(X_grid)
    weighted_log_prob = log_prob + np.log(gmm.weights_)
    Z = np.exp(gmm._logsumexp(weighted_log_prob, axis=1))
    Z = Z.reshape(xx.shape)
    
    # 绘制等高线
    contour = ax.contour(xx, yy, Z, levels=10, colors='black', alpha=0.4)
    ax.clabel(contour, inline=True, fontsize=8)
    
    # 叠加数据点
    for k in range(3):
        mask = predicted_labels == k
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[k], alpha=0.3, s=20)
    
    ax.set_xlabel('X₁')
    ax.set_ylabel('X₂')
    ax.set_title('混合高斯概率密度等高线')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig('em/gmm_2d.png', dpi=300, bbox_inches='tight')
    print("\n图像已保存: em/gmm_2d.png")
    plt.close()


def demo2_model_selection():
    """
    示例2: 使用BIC/AIC进行模型选择
    """
    print("\n" + "=" * 70)
    print("示例2: GMM模型选择（BIC/AIC）")
    print("=" * 70)
    
    # 生成数据（真实有3个成分）
    np.random.seed(42)
    n1, n2, n3 = 100, 120, 80
    X1 = np.random.normal([0, 0], 1, (n1, 2))
    X2 = np.random.normal([5, 5], 1.2, (n2, 2))
    X3 = np.random.normal([5, -5], 0.8, (n3, 2))
    X = np.vstack([X1, X2, X3])
    np.random.shuffle(X)
    
    print(f"\n数据: {len(X)}个样本")
    print(f"真实成分数: 3")
    
    # 测试不同的成分数
    n_components_range = range(1, 8)
    bic_scores = []
    aic_scores = []
    log_likelihoods = []
    
    print("\n测试不同的成分数:")
    print("-" * 50)
    
    for n_components in n_components_range:
        gmm = GaussianMixtureEM(n_components=n_components, 
                               covariance_type='full',
                               max_iter=100, verbose=False)
        gmm.fit(X, random_state=42)
        
        bic = gmm.bic(X)
        aic = gmm.aic(X)
        ll = gmm.log_likelihoods_[-1]
        
        bic_scores.append(bic)
        aic_scores.append(aic)
        log_likelihoods.append(ll)
        
        print(f"K={n_components}: BIC={bic:8.2f}, AIC={aic:8.2f}, LogLik={ll:8.2f}")
    
    # 找到最优K
    best_k_bic = n_components_range[np.argmin(bic_scores)]
    best_k_aic = n_components_range[np.argmin(aic_scores)]
    
    print(f"\n最优成分数:")
    print(f"  BIC: K = {best_k_bic}")
    print(f"  AIC: K = {best_k_aic}")
    print(f"  真实: K = 3")
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # BIC曲线
    ax = axes[0]
    ax.plot(n_components_range, bic_scores, 'o-', linewidth=2, markersize=8)
    ax.axvline(best_k_bic, color='red', linestyle='--', label=f'最优K={best_k_bic}')
    ax.axvline(3, color='green', linestyle=':', label='真实K=3')
    ax.set_xlabel('成分数K')
    ax.set_ylabel('BIC')
    ax.set_title('BIC模型选择')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # AIC曲线
    ax = axes[1]
    ax.plot(n_components_range, aic_scores, 'o-', linewidth=2, markersize=8, color='orange')
    ax.axvline(best_k_aic, color='red', linestyle='--', label=f'最优K={best_k_aic}')
    ax.axvline(3, color='green', linestyle=':', label='真实K=3')
    ax.set_xlabel('成分数K')
    ax.set_ylabel('AIC')
    ax.set_title('AIC模型选择')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 对数似然曲线
    ax = axes[2]
    ax.plot(n_components_range, log_likelihoods, 'o-', linewidth=2, markersize=8, color='green')
    ax.axvline(3, color='green', linestyle=':', label='真实K=3')
    ax.set_xlabel('成分数K')
    ax.set_ylabel('对数似然')
    ax.set_title('对数似然 vs 成分数')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('em/gmm_model_selection.png', dpi=300, bbox_inches='tight')
    print("\n图像已保存: em/gmm_model_selection.png")
    plt.close()


def demo3_covariance_types():
    """
    示例3: 不同协方差类型的对比
    """
    print("\n" + "=" * 70)
    print("示例3: GMM不同协方差类型对比")
    print("=" * 70)
    
    # 生成数据
    np.random.seed(42)
    n_samples = 300
    
    # 两个各向异性的高斯分布
    X1 = np.random.multivariate_normal([0, 0], [[2, 1], [1, 1]], 150)
    X2 = np.random.multivariate_normal([5, 3], [[1, -0.5], [-0.5, 2]], 150)
    X = np.vstack([X1, X2])
    np.random.shuffle(X)
    
    print(f"\n数据: {n_samples}个样本，2个成分")
    
    # 测试不同协方差类型
    cov_types = ['full', 'tied', 'diag', 'spherical']
    models = {}
    
    print("\n拟合不同协方差类型:")
    print("-" * 50)
    
    for cov_type in cov_types:
        print(f"\n协方差类型: {cov_type}")
        gmm = GaussianMixtureEM(n_components=2, covariance_type=cov_type,
                               max_iter=100, verbose=False)
        gmm.fit(X, random_state=42)
        models[cov_type] = gmm
        
        print(f"  对数似然: {gmm.log_likelihoods_[-1]:.2f}")
        print(f"  BIC: {gmm.bic(X):.2f}")
        print(f"  AIC: {gmm.aic(X):.2f}")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    colors = ['red', 'blue']
    
    for idx, cov_type in enumerate(cov_types):
        ax = axes[idx]
        gmm = models[cov_type]
        labels = gmm.predict(X)
        
        # 绘制数据点
        for k in range(2):
            mask = labels == k
            ax.scatter(X[mask, 0], X[mask, 1], c=colors[k], alpha=0.6, s=30,
                      edgecolors='k', linewidth=0.5)
        
        # 绘制均值
        for k in range(2):
            ax.scatter(gmm.means_[k, 0], gmm.means_[k, 1],
                      c='black', s=200, marker='x', linewidths=3)
        
        # 绘制协方差椭圆
        if cov_type == 'full':
            for k in range(2):
                plot_covariance_ellipse(gmm.means_[k], gmm.covariances_[k], ax,
                                       n_std=2.0, edgecolor=colors[k], 
                                       facecolor='none', linewidth=2, linestyle='--')
        elif cov_type == 'tied':
            for k in range(2):
                plot_covariance_ellipse(gmm.means_[k], gmm.covariances_, ax,
                                       n_std=2.0, edgecolor=colors[k], 
                                       facecolor='none', linewidth=2, linestyle='--')
        elif cov_type == 'diag':
            for k in range(2):
                cov = np.diag(gmm.covariances_[k])
                plot_covariance_ellipse(gmm.means_[k], cov, ax,
                                       n_std=2.0, edgecolor=colors[k], 
                                       facecolor='none', linewidth=2, linestyle='--')
        elif cov_type == 'spherical':
            for k in range(2):
                cov = gmm.covariances_[k] * np.eye(2)
                plot_covariance_ellipse(gmm.means_[k], cov, ax,
                                       n_std=2.0, edgecolor=colors[k], 
                                       facecolor='none', linewidth=2, linestyle='--')
        
        ax.set_xlabel('X₁')
        ax.set_ylabel('X₂')
        ax.set_title(f'{cov_type.capitalize()} 协方差\n'
                    f'BIC={gmm.bic(X):.1f}')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig('em/gmm_covariance_types.png', dpi=300, bbox_inches='tight')
    print("\n图像已保存: em/gmm_covariance_types.png")
    plt.close()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("高斯混合模型的EM算法 (GMM-EM) - 完整演示")
    print("=" * 70)
    
    # 运行示例
    demo1_2d_gmm()
    demo2_model_selection()
    demo3_covariance_types()
    
    print("\n" + "=" * 70)
    print("所有演示完成！")
    print("=" * 70)
    print("\n生成的文件:")
    print("  1. em/gmm_2d.png - 二维GMM拟合结果")
    print("  2. em/gmm_model_selection.png - BIC/AIC模型选择")
    print("  3. em/gmm_covariance_types.png - 不同协方差类型对比")
    print("\n" + "=" * 70)
