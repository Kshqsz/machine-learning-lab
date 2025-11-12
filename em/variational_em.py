"""
变分EM算法 (Variational EM Algorithm / Variational Bayesian EM)
==============================================================

变分EM算法是贝叶斯推断中的一种近似方法，通过变分推断来估计
后验分布，而不是像标准EM那样只估计点估计。

理论基础：
---------
标准EM: 
  最大化 log P(X|θ)
  
变分EM (VBEM):
  最大化证据下界 (ELBO)
  L(q, θ) = E_q[log P(X,Z|θ)] - E_q[log q(Z)]
           = E_q[log P(X,Z|θ)] + H(q)
  
  其中 q(Z) 是隐变量Z的变分分布（近似后验）

优势：
-----
1. 贝叶斯处理：对参数引入先验分布
2. 自动正则化：防止过拟合
3. 不确定性量化：得到后验分布而非点估计
4. 模型选择：自动惩罚复杂模型

变分推断步骤：
-----------
VE步（Variational E-step）：
  固定参数，更新隐变量的变分分布
  q*(Z) = argmax_q L(q, θ)
  
VM步（Variational M-step）：
  固定隐变量分布，更新参数的变分分布
  q*(θ) = argmax_q L(q, θ)

或者使用坐标上升变分推断（CAVI）

应用：
-----
- 贝叶斯混合模型
- 主题模型（LDA）
- 贝叶斯神经网络
- 变分自编码器（VAE）

本文件实现：
-----------
1. 变分贝叶斯高斯混合模型（VB-GMM）
2. 使用共轭先验的变分推断
3. ELBO计算
4. 与标准EM的对比
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Ellipse
from typing import Tuple, Optional
from scipy.special import digamma, gammaln

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class VariationalBayesianGMM:
    """
    变分贝叶斯高斯混合模型
    
    使用变分推断估计后验分布，而不是点估计。
    
    先验分布：
    ---------
    - π ~ Dirichlet(α₀)：混合系数的先验
    - μ_k ~ N(m₀, (β₀λ_k)⁻¹)：均值的先验
    - λ_k ~ Wishart(W₀, ν₀)：精度矩阵的先验
    
    参数：
    -----
    n_components : int
        混合成分数量
    
    alpha_0 : float
        Dirichlet先验的参数
    
    beta_0 : float
        均值先验的精度参数
    
    nu_0 : float
        Wishart先验的自由度
    
    W_0 : ndarray or None
        Wishart先验的尺度矩阵
    
    m_0 : ndarray or None
        均值的先验均值
    """
    
    def __init__(self, n_components: int = 2, alpha_0: float = 1.0,
                 beta_0: float = 1.0, nu_0: Optional[float] = None,
                 W_0: Optional[np.ndarray] = None, m_0: Optional[np.ndarray] = None,
                 max_iter: int = 100, tol: float = 1e-6, verbose: bool = True):
        self.n_components = n_components
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.nu_0 = nu_0
        self.W_0 = W_0
        self.m_0 = m_0
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        
        # 变分参数（后验参数）
        self.alpha_ = None  # Dirichlet后验参数
        self.beta_ = None   # 高斯后验精度参数
        self.nu_ = None     # Wishart后验自由度
        self.W_ = None      # Wishart后验尺度矩阵
        self.m_ = None      # 高斯后验均值
        
        # 其他
        self.responsibilities_ = None
        self.elbo_values_ = []
        self.n_iter_ = 0
        self.converged_ = False
    
    def _initialize_parameters(self, X: np.ndarray, random_state: Optional[int] = None):
        """初始化变分参数"""
        n_samples, n_features = X.shape
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # 初始化先验（如果未指定）
        if self.m_0 is None:
            self.m_0 = np.mean(X, axis=0)
        
        if self.W_0 is None:
            self.W_0 = np.cov(X.T) + 1e-6 * np.eye(n_features)
        
        if self.nu_0 is None:
            self.nu_0 = n_features
        
        # 初始化变分参数
        self.alpha_ = self.alpha_0 + n_samples / self.n_components
        self.alpha_ = np.full(self.n_components, self.alpha_)
        
        self.beta_ = np.full(self.n_components, self.beta_0 + n_samples / self.n_components)
        
        self.nu_ = np.full(self.n_components, self.nu_0 + n_samples / self.n_components)
        
        # 随机初始化均值
        indices = np.random.choice(n_samples, self.n_components, replace=False)
        self.m_ = X[indices].copy()
        
        # 初始化尺度矩阵
        self.W_ = np.array([self.W_0.copy() for _ in range(self.n_components)])
        
        # 初始化responsibilities
        self.responsibilities_ = np.random.dirichlet(
            np.ones(self.n_components), n_samples)
    
    def _compute_log_det_W(self) -> np.ndarray:
        """计算log|W_k|"""
        log_det_W = np.zeros(self.n_components)
        for k in range(self.n_components):
            sign, logdet = np.linalg.slogdet(self.W_[k])
            log_det_W[k] = logdet
        return log_det_W
    
    def _compute_expected_log_weights(self) -> np.ndarray:
        """
        计算期望对数权重：E[log π_k]
        E[log π_k] = ψ(α_k) - ψ(Σ_j α_j)
        """
        return digamma(self.alpha_) - digamma(np.sum(self.alpha_))
    
    def _compute_expected_log_precision(self, n_features: int) -> np.ndarray:
        """
        计算期望对数精度：E[log |Λ_k|]
        """
        log_det_W = self._compute_log_det_W()
        expected_log_det = log_det_W.copy()
        
        for k in range(self.n_components):
            for i in range(n_features):
                expected_log_det[k] += digamma((self.nu_[k] - i) / 2)
            expected_log_det[k] += n_features * np.log(2)
        
        return expected_log_det
    
    def _ve_step(self, X: np.ndarray):
        """
        变分E步：更新隐变量的变分分布（responsibilities）
        
        计算 r_{ik} = P(z_i=k | x_i)
        """
        n_samples, n_features = X.shape
        
        # 计算期望对数权重
        expected_log_weights = self._compute_expected_log_weights()
        
        # 计算期望对数精度
        expected_log_precision = self._compute_expected_log_precision(n_features)
        
        # 计算responsibilities
        log_rho = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            diff = X - self.m_[k]
            
            # 期望马氏距离
            # E[(x-μ)^T Λ (x-μ)] = ν_k * (x-m_k)^T W_k (x-m_k) + D/β_k
            W_k_inv = np.linalg.inv(self.W_[k])
            mahalanobis = np.sum(diff @ W_k_inv * diff, axis=1)
            expected_mahalanobis = self.nu_[k] * mahalanobis + n_features / self.beta_[k]
            
            log_rho[:, k] = (expected_log_weights[k] +
                            0.5 * expected_log_precision[k] -
                            0.5 * n_features * np.log(2 * np.pi) -
                            0.5 * expected_mahalanobis)
        
        # 归一化（log-sum-exp技巧）
        log_rho_max = np.max(log_rho, axis=1, keepdims=True)
        rho = np.exp(log_rho - log_rho_max)
        self.responsibilities_ = rho / (rho.sum(axis=1, keepdims=True) + 1e-10)
    
    def _vm_step(self, X: np.ndarray):
        """
        变分M步：更新参数的变分分布
        """
        n_samples, n_features = X.shape
        
        # 计算有效样本数
        N_k = self.responsibilities_.sum(axis=0)
        
        # 计算加权均值
        X_bar_k = (self.responsibilities_.T @ X) / (N_k[:, np.newaxis] + 1e-10)
        
        # 更新变分参数
        # 1. Dirichlet参数
        self.alpha_ = self.alpha_0 + N_k
        
        # 2. 高斯-Wishart参数
        for k in range(self.n_components):
            # β参数
            self.beta_[k] = self.beta_0 + N_k[k]
            
            # 均值参数
            self.m_[k] = (self.beta_0 * self.m_0 + N_k[k] * X_bar_k[k]) / self.beta_[k]
            
            # 自由度参数
            self.nu_[k] = self.nu_0 + N_k[k]
            
            # 尺度矩阵
            diff = X - X_bar_k[k]
            S_k = (self.responsibilities_[:, k:k+1] * diff).T @ diff
            
            diff_mean = X_bar_k[k] - self.m_0
            W_k_inv = np.linalg.inv(self.W_0)
            
            W_k_inv_new = (W_k_inv + S_k +
                          (self.beta_0 * N_k[k]) / (self.beta_0 + N_k[k]) *
                          np.outer(diff_mean, diff_mean))
            
            self.W_[k] = np.linalg.inv(W_k_inv_new)
    
    def _compute_elbo(self, X: np.ndarray) -> float:
        """
        计算证据下界（ELBO）
        
        ELBO = E[log P(X,Z,θ)] - E[log q(Z,θ)]
        """
        n_samples, n_features = X.shape
        
        elbo = 0.0
        
        # 1. E[log P(X|Z,μ,Λ)]
        expected_log_precision = self._compute_expected_log_precision(n_features)
        
        for k in range(self.n_components):
            diff = X - self.m_[k]
            W_k_inv = np.linalg.inv(self.W_[k])
            mahalanobis = np.sum(diff @ W_k_inv * diff, axis=1)
            expected_mahalanobis = self.nu_[k] * mahalanobis + n_features / self.beta_[k]
            
            term = (0.5 * expected_log_precision[k] -
                   0.5 * n_features * np.log(2 * np.pi) -
                   0.5 * expected_mahalanobis)
            
            elbo += np.sum(self.responsibilities_[:, k] * term)
        
        # 2. E[log P(Z|π)]
        expected_log_weights = self._compute_expected_log_weights()
        elbo += np.sum(self.responsibilities_ * expected_log_weights)
        
        # 3. E[log P(π)] - E[log q(π)] (Dirichlet部分)
        elbo += (gammaln(np.sum(self.alpha_0 * np.ones(self.n_components))) -
                np.sum(gammaln(self.alpha_0 * np.ones(self.n_components))))
        elbo += np.sum((self.alpha_0 - 1) * expected_log_weights)
        
        elbo -= (gammaln(np.sum(self.alpha_)) - np.sum(gammaln(self.alpha_)))
        elbo -= np.sum((self.alpha_ - 1) * expected_log_weights)
        
        # 4. -E[log q(Z)]（熵项）
        log_resp = np.log(self.responsibilities_ + 1e-10)
        elbo -= np.sum(self.responsibilities_ * log_resp)
        
        # 5. E[log P(μ,Λ)] - E[log q(μ,Λ)]（简化版本）
        # 这部分计算较复杂，这里使用简化估计
        
        return elbo
    
    def fit(self, X: np.ndarray, random_state: Optional[int] = None) -> 'VariationalBayesianGMM':
        """
        拟合变分贝叶斯GMM
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features)
            训练数据
        
        Returns:
        --------
        self
        """
        # 初始化
        self._initialize_parameters(X, random_state)
        
        self.elbo_values_ = []
        self.converged_ = False
        
        if self.verbose:
            print("变分EM算法开始迭代...")
        
        for iteration in range(self.max_iter):
            # VE步
            self._ve_step(X)
            
            # VM步
            self._vm_step(X)
            
            # 计算ELBO
            elbo = self._compute_elbo(X)
            self.elbo_values_.append(elbo)
            
            if self.verbose and (iteration % 10 == 0 or iteration < 5):
                print(f"Iteration {iteration}: ELBO = {elbo:.6f}")
            
            # 检查收敛
            if iteration > 0:
                elbo_change = abs(self.elbo_values_[-1] - self.elbo_values_[-2])
                if elbo_change < self.tol:
                    self.converged_ = True
                    if self.verbose:
                        print(f"\n收敛于第 {iteration} 次迭代")
                        print(f"ELBO变化: {elbo_change:.8f} < {self.tol}")
                    break
        
        self.n_iter_ = len(self.elbo_values_)
        
        if not self.converged_ and self.verbose:
            print(f"\n警告: 未收敛（达到最大迭代次数 {self.max_iter}）")
        
        if self.verbose:
            print(f"最终ELBO: {self.elbo_values_[-1]:.6f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测聚类标签"""
        self._ve_step(X)
        return np.argmax(self.responsibilities_, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        self._ve_step(X)
        return self.responsibilities_
    
    def get_means(self) -> np.ndarray:
        """获取后验均值（点估计）"""
        return self.m_
    
    def get_weights(self) -> np.ndarray:
        """获取后验权重（点估计）"""
        return self.alpha_ / np.sum(self.alpha_)


def demo1_vbem_vs_em():
    """
    示例1: 变分贝叶斯EM vs 标准EM
    """
    print("=" * 70)
    print("示例1: 变分贝叶斯EM vs 标准EM")
    print("=" * 70)
    
    # 生成数据
    np.random.seed(42)
    n_samples = 300
    
    # 两个高斯分布
    X1 = np.random.normal([0, 0], 1, (150, 2))
    X2 = np.random.normal([4, 4], 1, (150, 2))
    X = np.vstack([X1, X2])
    np.random.shuffle(X)
    
    print(f"\n数据: {n_samples}个样本，2个成分")
    
    # 标准EM
    print("\n" + "-" * 70)
    print("标准EM算法:")
    print("-" * 70)
    
    from gmm_em import GaussianMixtureEM
    
    gmm_em = GaussianMixtureEM(n_components=2, covariance_type='full',
                               max_iter=100, verbose=False)
    gmm_em.fit(X, random_state=42)
    
    print(f"迭代次数: {gmm_em.n_iter_}")
    print(f"最终对数似然: {gmm_em.log_likelihoods_[-1]:.6f}")
    
    # 变分贝叶斯EM
    print("\n" + "-" * 70)
    print("变分贝叶斯EM算法:")
    print("-" * 70)
    
    vbgmm = VariationalBayesianGMM(n_components=2, alpha_0=1.0, beta_0=1.0,
                                   max_iter=100, verbose=True)
    vbgmm.fit(X, random_state=42)
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 标准EM结果
    ax = axes[0, 0]
    labels_em = gmm_em.predict(X)
    colors = ['red', 'blue']
    
    for k in range(2):
        mask = labels_em == k
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[k], alpha=0.6, s=30,
                  label=f'聚类{k}', edgecolors='k', linewidth=0.5)
    
    for k in range(2):
        ax.scatter(gmm_em.means_[k, 0], gmm_em.means_[k, 1],
                  c='black', s=200, marker='x', linewidths=3)
    
    ax.set_xlabel('X₁')
    ax.set_ylabel('X₂')
    ax.set_title('标准EM算法结果')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # 变分EM结果
    ax = axes[0, 1]
    labels_vb = vbgmm.predict(X)
    
    for k in range(2):
        mask = labels_vb == k
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[k], alpha=0.6, s=30,
                  label=f'聚类{k}', edgecolors='k', linewidth=0.5)
    
    means_vb = vbgmm.get_means()
    for k in range(2):
        ax.scatter(means_vb[k, 0], means_vb[k, 1],
                  c='black', s=200, marker='x', linewidths=3)
    
    ax.set_xlabel('X₁')
    ax.set_ylabel('X₂')
    ax.set_title('变分贝叶斯EM算法结果')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # 对数似然/ELBO对比
    ax = axes[1, 0]
    ax.plot(range(len(gmm_em.log_likelihoods_)), gmm_em.log_likelihoods_,
            'o-', linewidth=2, label='标准EM（对数似然）', markersize=5)
    ax.set_xlabel('迭代次数')
    ax.set_ylabel('对数似然')
    ax.set_title('标准EM收敛曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax2 = axes[1, 1]
    ax2.plot(range(len(vbgmm.elbo_values_)), vbgmm.elbo_values_,
            's-', linewidth=2, label='变分EM（ELBO）', markersize=5, color='green')
    ax2.set_xlabel('迭代次数')
    ax2.set_ylabel('ELBO')
    ax2.set_title('变分EM收敛曲线')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('em/vbem_vs_em.png', dpi=300, bbox_inches='tight')
    print("\n图像已保存: em/vbem_vs_em.png")
    plt.close()


def demo2_automatic_relevance_determination():
    """
    示例2: 自动相关性确定（ARD）
    变分贝叶斯可以自动确定有效成分数量
    """
    print("\n" + "=" * 70)
    print("示例2: 变分贝叶斯的自动模型选择")
    print("=" * 70)
    
    # 生成数据（真实只有3个成分）
    np.random.seed(42)
    
    X1 = np.random.normal([0, 0], 0.8, (100, 2))
    X2 = np.random.normal([4, 0], 0.8, (100, 2))
    X3 = np.random.normal([2, 3], 0.8, (100, 2))
    X = np.vstack([X1, X2, X3])
    np.random.shuffle(X)
    
    print(f"\n数据: {len(X)}个样本")
    print(f"真实成分数: 3")
    print(f"测试成分数: 6（过度拟合）")
    
    # 使用较多的成分拟合
    print("\n" + "-" * 70)
    print("拟合VB-GMM（6个成分）:")
    print("-" * 70)
    
    vbgmm = VariationalBayesianGMM(n_components=6, alpha_0=0.1, beta_0=1.0,
                                   max_iter=100, verbose=True)
    vbgmm.fit(X, random_state=42)
    
    # 分析有效成分
    weights = vbgmm.get_weights()
    print(f"\n后验权重:")
    for k in range(6):
        print(f"  成分{k}: {weights[k]:.4f}")
    
    # 确定有效成分（权重大于阈值）
    threshold = 0.05
    effective_components = np.sum(weights > threshold)
    print(f"\n有效成分数（权重>{threshold}）: {effective_components}")
    print(f"真实成分数: 3")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 聚类结果
    ax = axes[0]
    labels = vbgmm.predict(X)
    means = vbgmm.get_means()
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    for k in range(6):
        mask = labels == k
        if np.sum(mask) > 0:
            ax.scatter(X[mask, 0], X[mask, 1], c=colors[k], alpha=0.6, s=30,
                      label=f'成分{k} (w={weights[k]:.2f})',
                      edgecolors='k', linewidth=0.5)
            
            ax.scatter(means[k, 0], means[k, 1],
                      c='black', s=200, marker='x', linewidths=3)
    
    ax.set_xlabel('X₁')
    ax.set_ylabel('X₂')
    ax.set_title('变分贝叶斯GMM聚类结果\n（自动识别有效成分）')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # 权重分布
    ax = axes[1]
    ax.bar(range(6), weights, alpha=0.8, edgecolor='black')
    ax.axhline(threshold, color='red', linestyle='--', label=f'阈值={threshold}')
    ax.set_xlabel('成分编号')
    ax.set_ylabel('后验权重')
    ax.set_title('各成分的后验权重\n（小权重成分被自动抑制）')
    ax.set_xticks(range(6))
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('em/vbem_ard.png', dpi=300, bbox_inches='tight')
    print("\n图像已保存: em/vbem_ard.png")
    plt.close()


def demo3_bayesian_vs_frequentist():
    """
    示例3: 贝叶斯vs频率派的对比
    """
    print("\n" + "=" * 70)
    print("示例3: 贝叶斯方法 vs 频率派方法")
    print("=" * 70)
    
    # 生成小数据集（容易过拟合）
    np.random.seed(42)
    n_samples = 50
    
    X1 = np.random.normal([0, 0], 0.5, (25, 2))
    X2 = np.random.normal([2, 2], 0.5, (25, 2))
    X = np.vstack([X1, X2])
    np.random.shuffle(X)
    
    print(f"\n数据: {n_samples}个样本（小数据集）")
    print(f"真实成分数: 2")
    
    # 标准EM
    from gmm_em import GaussianMixtureEM
    
    print("\n频率派方法（标准EM）:")
    gmm_em = GaussianMixtureEM(n_components=2, covariance_type='full',
                               max_iter=100, verbose=False)
    gmm_em.fit(X, random_state=42)
    print(f"对数似然: {gmm_em.log_likelihoods_[-1]:.2f}")
    
    # 变分贝叶斯EM
    print("\n贝叶斯方法（变分EM）:")
    vbgmm = VariationalBayesianGMM(n_components=2, alpha_0=1.0, beta_0=0.1,
                                   max_iter=100, verbose=False)
    vbgmm.fit(X, random_state=42)
    print(f"ELBO: {vbgmm.elbo_values_[-1]:.2f}")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['red', 'blue']
    
    # 频率派结果
    ax = axes[0]
    labels_em = gmm_em.predict(X)
    
    for k in range(2):
        mask = labels_em == k
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[k], alpha=0.6, s=50,
                  edgecolors='k', linewidth=0.5)
    
    for k in range(2):
        ax.scatter(gmm_em.means_[k, 0], gmm_em.means_[k, 1],
                  c='black', s=300, marker='x', linewidths=4)
    
    ax.set_xlabel('X₁')
    ax.set_ylabel('X₂')
    ax.set_title('频率派方法（点估计）\n标准EM算法')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # 贝叶斯结果
    ax = axes[1]
    labels_vb = vbgmm.predict(X)
    means_vb = vbgmm.get_means()
    
    for k in range(2):
        mask = labels_vb == k
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[k], alpha=0.6, s=50,
                  edgecolors='k', linewidth=0.5)
    
    for k in range(2):
        ax.scatter(means_vb[k, 0], means_vb[k, 1],
                  c='black', s=300, marker='x', linewidths=4)
    
    ax.set_xlabel('X₁')
    ax.set_ylabel('X₂')
    ax.set_title('贝叶斯方法（后验分布）\n变分EM算法')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig('em/bayesian_vs_frequentist.png', dpi=300, bbox_inches='tight')
    print("\n图像已保存: em/bayesian_vs_frequentist.png")
    plt.close()
    
    print("\n" + "-" * 70)
    print("贝叶斯方法的优势:")
    print("-" * 70)
    print("1. 自动正则化：先验防止过拟合")
    print("2. 不确定性量化：得到后验分布")
    print("3. 自动模型选择：ARD机制")
    print("4. 小数据集表现更好")
    print("5. 避免奇异协方差矩阵")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("变分EM算法 (Variational EM / VBEM) - 完整演示")
    print("=" * 70)
    
    # 运行示例
    demo1_vbem_vs_em()
    demo2_automatic_relevance_determination()
    demo3_bayesian_vs_frequentist()
    
    print("\n" + "=" * 70)
    print("所有演示完成！")
    print("=" * 70)
    print("\n生成的文件:")
    print("  1. em/vbem_vs_em.png - 变分EM vs 标准EM对比")
    print("  2. em/vbem_ard.png - 自动相关性确定")
    print("  3. em/bayesian_vs_frequentist.png - 贝叶斯vs频率派")
    print("\n" + "=" * 70)
