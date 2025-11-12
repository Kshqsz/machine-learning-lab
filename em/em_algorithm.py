"""
EM算法 (Expectation-Maximization Algorithm)
============================================

EM算法是一种迭代算法，用于含有隐变量的概率模型参数的极大似然估计，
或极大后验概率估计。

理论基础：
---------
1. 问题设置：
   - 观测数据：Y (可观测变量)
   - 隐变量：Z (不可观测变量)
   - 完全数据：(Y, Z)
   - 模型参数：θ
   
2. 目标：最大化对数似然函数
   L(θ) = log P(Y|θ) = log Σ_Z P(Y,Z|θ)

3. EM算法的两步：
   E步(Expectation)：计算Q函数
   Q(θ, θ^(i)) = E_Z[log P(Y,Z|θ) | Y, θ^(i)]
                = Σ_Z P(Z|Y,θ^(i)) log P(Y,Z|θ)
   
   M步(Maximization)：最大化Q函数
   θ^(i+1) = argmax_θ Q(θ, θ^(i))

4. 收敛性：
   - EM算法使似然函数单调递增
   - 收敛到局部最优解或鞍点

经典应用：
---------
- 高斯混合模型(GMM)
- 隐马尔可夫模型(HMM)
- 朴素贝叶斯
- 因子分析
- 缺失数据处理

本文件实现：
-----------
1. 抽象EM算法基类
2. 硬币投掷问题示例
3. 混合伯努利模型
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from typing import Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class EMAlgorithm(ABC):
    """
    EM算法抽象基类
    
    子类需要实现：
    - e_step(): E步，计算隐变量的后验概率
    - m_step(): M步，更新模型参数
    - compute_log_likelihood(): 计算对数似然
    """
    
    def __init__(self, max_iter: int = 100, tol: float = 1e-6, verbose: bool = True):
        """
        Parameters:
        -----------
        max_iter : int
            最大迭代次数
        tol : float
            收敛阈值（对数似然的变化量）
        verbose : bool
            是否打印迭代信息
        """
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.log_likelihoods_ = []
        self.n_iter_ = 0
        
    @abstractmethod
    def e_step(self, X: np.ndarray) -> np.ndarray:
        """
        E步：计算隐变量的后验概率
        
        Parameters:
        -----------
        X : ndarray
            观测数据
            
        Returns:
        --------
        responsibilities : ndarray
            隐变量的后验概率（责任）
        """
        pass
    
    @abstractmethod
    def m_step(self, X: np.ndarray, responsibilities: np.ndarray):
        """
        M步：更新模型参数
        
        Parameters:
        -----------
        X : ndarray
            观测数据
        responsibilities : ndarray
            隐变量的后验概率
        """
        pass
    
    @abstractmethod
    def compute_log_likelihood(self, X: np.ndarray) -> float:
        """
        计算对数似然函数
        
        Parameters:
        -----------
        X : ndarray
            观测数据
            
        Returns:
        --------
        log_likelihood : float
            对数似然值
        """
        pass
    
    def fit(self, X: np.ndarray) -> 'EMAlgorithm':
        """
        拟合EM算法
        
        Parameters:
        -----------
        X : ndarray
            观测数据
            
        Returns:
        --------
        self : object
        """
        self.log_likelihoods_ = []
        
        for iteration in range(self.max_iter):
            # E步
            responsibilities = self.e_step(X)
            
            # M步
            self.m_step(X, responsibilities)
            
            # 计算对数似然
            log_likelihood = self.compute_log_likelihood(X)
            self.log_likelihoods_.append(log_likelihood)
            
            if self.verbose and (iteration % 10 == 0 or iteration < 5):
                print(f"Iteration {iteration}: Log-Likelihood = {log_likelihood:.6f}")
            
            # 检查收敛
            if iteration > 0:
                ll_change = abs(self.log_likelihoods_[-1] - self.log_likelihoods_[-2])
                if ll_change < self.tol:
                    if self.verbose:
                        print(f"\n收敛于第 {iteration} 次迭代")
                        print(f"对数似然变化: {ll_change:.8f} < {self.tol}")
                    break
        
        self.n_iter_ = len(self.log_likelihoods_)
        
        if self.verbose:
            print(f"\n最终对数似然: {self.log_likelihoods_[-1]:.6f}")
        
        return self


class CoinFlipEM(EMAlgorithm):
    """
    硬币投掷问题的EM算法
    
    问题描述：
    ---------
    有两枚硬币A和B，投掷概率分别为θ_A和θ_B（正面朝上的概率）。
    进行5组实验，每组随机选择一枚硬币（隐变量），投掷10次。
    观测数据：每组中正面朝上的次数
    目标：估计θ_A和θ_B
    
    模型：
    -----
    - 隐变量Z_i ∈ {A, B}：第i组选择的硬币
    - 观测Y_i：第i组正面朝上的次数（共n_i次投掷）
    - P(Y_i | Z_i=A) = Binomial(Y_i | n_i, θ_A)
    - P(Y_i | Z_i=B) = Binomial(Y_i | n_i, θ_B)
    """
    
    def __init__(self, n_flips: int = 10, max_iter: int = 100, 
                 tol: float = 1e-6, verbose: bool = True):
        """
        Parameters:
        -----------
        n_flips : int
            每组实验的投掷次数
        """
        super().__init__(max_iter, tol, verbose)
        self.n_flips = n_flips
        self.theta_A = None  # 硬币A的参数
        self.theta_B = None  # 硬币B的参数
        
    def initialize_parameters(self, random_state: Optional[int] = None):
        """随机初始化参数"""
        if random_state is not None:
            np.random.seed(random_state)
        self.theta_A = np.random.uniform(0.3, 0.7)
        self.theta_B = np.random.uniform(0.3, 0.7)
        
    def e_step(self, X: np.ndarray) -> np.ndarray:
        """
        E步：计算每组实验来自硬币A的后验概率
        
        Parameters:
        -----------
        X : ndarray, shape (n_experiments,)
            每组实验中正面朝上的次数
            
        Returns:
        --------
        responsibilities : ndarray, shape (n_experiments, 2)
            responsibilities[:, 0] = P(Z=A | Y, θ)
            responsibilities[:, 1] = P(Z=B | Y, θ)
        """
        n_experiments = len(X)
        responsibilities = np.zeros((n_experiments, 2))
        
        for i in range(n_experiments):
            heads = X[i]
            tails = self.n_flips - heads
            
            # P(Y_i | Z_i=A, θ) = θ_A^heads * (1-θ_A)^tails
            likelihood_A = (self.theta_A ** heads) * ((1 - self.theta_A) ** tails)
            
            # P(Y_i | Z_i=B, θ) = θ_B^heads * (1-θ_B)^tails
            likelihood_B = (self.theta_B ** heads) * ((1 - self.theta_B) ** tails)
            
            # 假设先验P(Z=A) = P(Z=B) = 0.5
            # 后验概率：P(Z_i=A | Y_i, θ) = P(Y_i | Z_i=A) / [P(Y_i | Z_i=A) + P(Y_i | Z_i=B)]
            total = likelihood_A + likelihood_B
            responsibilities[i, 0] = likelihood_A / total  # P(Z=A | Y)
            responsibilities[i, 1] = likelihood_B / total  # P(Z=B | Y)
        
        return responsibilities
    
    def m_step(self, X: np.ndarray, responsibilities: np.ndarray):
        """
        M步：更新θ_A和θ_B
        
        Parameters:
        -----------
        X : ndarray, shape (n_experiments,)
            每组实验中正面朝上的次数
        responsibilities : ndarray, shape (n_experiments, 2)
            隐变量的后验概率
        """
        # 更新θ_A：加权平均
        # θ_A = Σ_i P(Z_i=A|Y_i) * Y_i / Σ_i P(Z_i=A|Y_i) * n_flips
        weighted_heads_A = np.sum(responsibilities[:, 0] * X)
        weighted_total_A = np.sum(responsibilities[:, 0] * self.n_flips)
        self.theta_A = weighted_heads_A / weighted_total_A
        
        # 更新θ_B
        weighted_heads_B = np.sum(responsibilities[:, 1] * X)
        weighted_total_B = np.sum(responsibilities[:, 1] * self.n_flips)
        self.theta_B = weighted_heads_B / weighted_total_B
    
    def compute_log_likelihood(self, X: np.ndarray) -> float:
        """
        计算对数似然：log P(Y | θ) = Σ_i log [P(Y_i|Z=A)P(Z=A) + P(Y_i|Z=B)P(Z=B)]
        """
        log_likelihood = 0.0
        
        for heads in X:
            tails = self.n_flips - heads
            
            # P(Y_i | Z=A, θ) * P(Z=A)，假设P(Z=A) = 0.5
            likelihood_A = 0.5 * (self.theta_A ** heads) * ((1 - self.theta_A) ** tails)
            
            # P(Y_i | Z=B, θ) * P(Z=B)
            likelihood_B = 0.5 * (self.theta_B ** heads) * ((1 - self.theta_B) ** tails)
            
            # log P(Y_i | θ)
            log_likelihood += np.log(likelihood_A + likelihood_B)
        
        return log_likelihood


class MixtureBernoulliEM(EMAlgorithm):
    """
    混合伯努利模型的EM算法
    
    模型描述：
    ---------
    假设有K个伯努利分布的混合：
    P(x) = Σ_k π_k * Bernoulli(x | μ_k)
    
    其中：
    - π_k: 第k个分量的混合系数（权重），Σ_k π_k = 1
    - μ_k: 第k个伯努利分布的参数
    
    这是一个简单的离散混合模型示例。
    """
    
    def __init__(self, n_components: int = 2, max_iter: int = 100,
                 tol: float = 1e-6, verbose: bool = True):
        """
        Parameters:
        -----------
        n_components : int
            混合成分的数量
        """
        super().__init__(max_iter, tol, verbose)
        self.n_components = n_components
        self.weights_ = None  # 混合系数 π
        self.means_ = None    # 伯努利参数 μ
        
    def initialize_parameters(self, random_state: Optional[int] = None):
        """初始化参数"""
        if random_state is not None:
            np.random.seed(random_state)
        
        # 随机初始化混合系数
        self.weights_ = np.random.dirichlet(np.ones(self.n_components))
        
        # 随机初始化伯努利参数
        self.means_ = np.random.uniform(0.3, 0.7, self.n_components)
    
    def e_step(self, X: np.ndarray) -> np.ndarray:
        """
        E步：计算responsibilities（后验概率）
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples,)
            观测数据（0或1）
            
        Returns:
        --------
        responsibilities : ndarray, shape (n_samples, n_components)
            responsibilities[i, k] = P(Z_i=k | x_i, θ)
        """
        n_samples = len(X)
        responsibilities = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            # P(x_i | Z_i=k, θ) = μ_k^{x_i} * (1-μ_k)^{1-x_i}
            likelihood = (self.means_[k] ** X) * ((1 - self.means_[k]) ** (1 - X))
            
            # P(x_i, Z_i=k | θ) = P(x_i | Z_i=k) * P(Z_i=k)
            responsibilities[:, k] = self.weights_[k] * likelihood
        
        # 归一化：P(Z_i=k | x_i, θ) = P(x_i, Z_i=k) / Σ_k P(x_i, Z_i=k)
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        
        return responsibilities
    
    def m_step(self, X: np.ndarray, responsibilities: np.ndarray):
        """
        M步：更新参数
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples,)
            观测数据
        responsibilities : ndarray, shape (n_samples, n_components)
            后验概率
        """
        n_samples = len(X)
        
        # 更新混合系数：π_k = (1/N) * Σ_i P(Z_i=k | x_i)
        n_k = responsibilities.sum(axis=0)
        self.weights_ = n_k / n_samples
        
        # 更新伯努利参数：μ_k = Σ_i P(Z_i=k | x_i) * x_i / Σ_i P(Z_i=k | x_i)
        for k in range(self.n_components):
            self.means_[k] = np.sum(responsibilities[:, k] * X) / n_k[k]
    
    def compute_log_likelihood(self, X: np.ndarray) -> float:
        """
        计算对数似然：log P(X | θ) = Σ_i log [Σ_k π_k * P(x_i | μ_k)]
        """
        n_samples = len(X)
        log_likelihood = 0.0
        
        for i in range(n_samples):
            # P(x_i | θ) = Σ_k π_k * μ_k^{x_i} * (1-μ_k)^{1-x_i}
            likelihood = 0.0
            for k in range(self.n_components):
                p_x_given_k = (self.means_[k] ** X[i]) * ((1 - self.means_[k]) ** (1 - X[i]))
                likelihood += self.weights_[k] * p_x_given_k
            
            log_likelihood += np.log(likelihood + 1e-10)  # 避免log(0)
        
        return log_likelihood
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测每个样本的聚类标签
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples,)
            观测数据
            
        Returns:
        --------
        labels : ndarray, shape (n_samples,)
            聚类标签
        """
        responsibilities = self.e_step(X)
        return np.argmax(responsibilities, axis=1)


def demo1_coin_flip():
    """
    示例1: 硬币投掷问题
    经典的EM算法示例
    """
    print("=" * 70)
    print("示例1: 硬币投掷问题的EM算法")
    print("=" * 70)
    
    # 生成模拟数据
    # 真实参数
    true_theta_A = 0.7  # 硬币A正面朝上概率
    true_theta_B = 0.3  # 硬币B正面朝上概率
    n_flips = 10
    n_experiments = 5
    
    np.random.seed(42)
    
    # 随机选择硬币并投掷
    true_coins = np.random.choice(['A', 'B'], size=n_experiments)
    observed_heads = []
    
    print(f"\n真实参数: θ_A = {true_theta_A}, θ_B = {true_theta_B}")
    print(f"实验次数: {n_experiments}, 每次投掷: {n_flips}次\n")
    print("观测数据生成过程:")
    
    for i, coin in enumerate(true_coins):
        if coin == 'A':
            heads = np.random.binomial(n_flips, true_theta_A)
        else:
            heads = np.random.binomial(n_flips, true_theta_B)
        observed_heads.append(heads)
        print(f"  实验{i+1}: 硬币{coin}, 正面次数 = {heads}/{n_flips}")
    
    observed_heads = np.array(observed_heads)
    
    # EM算法估计
    print("\n" + "-" * 70)
    print("EM算法迭代过程:")
    print("-" * 70)
    
    model = CoinFlipEM(n_flips=n_flips, max_iter=100, tol=1e-6, verbose=True)
    model.initialize_parameters(random_state=123)
    
    print(f"\n初始参数: θ_A = {model.theta_A:.4f}, θ_B = {model.theta_B:.4f}")
    print()
    
    model.fit(observed_heads)
    
    print(f"\n估计参数: θ_A = {model.theta_A:.4f}, θ_B = {model.theta_B:.4f}")
    print(f"真实参数: θ_A = {true_theta_A:.4f}, θ_B = {true_theta_B:.4f}")
    print(f"\n参数误差: |Δθ_A| = {abs(model.theta_A - true_theta_A):.4f}, "
          f"|Δθ_B| = {abs(model.theta_B - true_theta_B):.4f}")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 对数似然曲线
    ax = axes[0]
    ax.plot(range(len(model.log_likelihoods_)), model.log_likelihoods_, 
            'o-', linewidth=2, markersize=6)
    ax.set_xlabel('迭代次数')
    ax.set_ylabel('对数似然')
    ax.set_title('EM算法收敛过程')
    ax.grid(True, alpha=0.3)
    
    # 参数演化（需要修改代码记录每次迭代的参数）
    # 这里展示最终结果的对比
    ax = axes[1]
    x = ['θ_A', 'θ_B']
    true_params = [true_theta_A, true_theta_B]
    estimated_params = [model.theta_A, model.theta_B]
    
    x_pos = np.arange(len(x))
    width = 0.35
    
    ax.bar(x_pos - width/2, true_params, width, label='真实值', alpha=0.8)
    ax.bar(x_pos + width/2, estimated_params, width, label='估计值', alpha=0.8)
    
    ax.set_ylabel('概率')
    ax.set_title('参数估计结果对比')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('em/em_coin_flip.png', dpi=300, bbox_inches='tight')
    print("\n图像已保存: em/em_coin_flip.png")
    plt.close()


def demo2_mixture_bernoulli():
    """
    示例2: 混合伯努利模型
    展示EM算法在混合模型中的应用
    """
    print("\n" + "=" * 70)
    print("示例2: 混合伯努利模型的EM算法")
    print("=" * 70)
    
    # 生成模拟数据
    np.random.seed(42)
    n_samples = 200
    true_weights = np.array([0.3, 0.7])
    true_means = np.array([0.2, 0.8])
    
    # 生成混合数据
    component_labels = np.random.choice(2, size=n_samples, p=true_weights)
    X = np.zeros(n_samples)
    for i in range(n_samples):
        X[i] = np.random.binomial(1, true_means[component_labels[i]])
    
    print(f"\n真实参数:")
    print(f"  混合系数: π = {true_weights}")
    print(f"  伯努利参数: μ = {true_means}")
    print(f"  数据: {n_samples}个样本")
    print(f"  各成分样本数: 成分0={np.sum(component_labels==0)}, "
          f"成分1={np.sum(component_labels==1)}")
    
    # EM算法
    print("\n" + "-" * 70)
    print("EM算法迭代过程:")
    print("-" * 70)
    
    model = MixtureBernoulliEM(n_components=2, max_iter=100, tol=1e-6, verbose=True)
    model.initialize_parameters(random_state=123)
    
    print(f"\n初始参数:")
    print(f"  混合系数: π = {model.weights_}")
    print(f"  伯努利参数: μ = {model.means_}")
    print()
    
    model.fit(X)
    
    # 对齐参数（因为混合模型的成分标签可能交换）
    if model.means_[0] > model.means_[1]:
        model.weights_ = model.weights_[::-1]
        model.means_ = model.means_[::-1]
    
    print(f"\n估计参数:")
    print(f"  混合系数: π = {model.weights_}")
    print(f"  伯努利参数: μ = {model.means_}")
    
    print(f"\n真实参数:")
    print(f"  混合系数: π = {true_weights}")
    print(f"  伯努利参数: μ = {true_means}")
    
    # 预测聚类
    predicted_labels = model.predict(X)
    
    # 计算准确率（需要对齐标签）
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(component_labels, predicted_labels)
    row_ind, col_ind = linear_sum_assignment(-cm)
    accuracy = cm[row_ind, col_ind].sum() / n_samples
    
    print(f"\n聚类准确率: {accuracy:.2%}")
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 对数似然曲线
    ax = axes[0]
    ax.plot(range(len(model.log_likelihoods_)), model.log_likelihoods_, 
            'o-', linewidth=2, markersize=6)
    ax.set_xlabel('迭代次数')
    ax.set_ylabel('对数似然')
    ax.set_title('EM算法收敛过程')
    ax.grid(True, alpha=0.3)
    
    # 参数对比
    ax = axes[1]
    params = ['π_0', 'π_1', 'μ_0', 'μ_1']
    true_vals = [true_weights[0], true_weights[1], true_means[0], true_means[1]]
    est_vals = [model.weights_[0], model.weights_[1], model.means_[0], model.means_[1]]
    
    x_pos = np.arange(len(params))
    width = 0.35
    
    ax.bar(x_pos - width/2, true_vals, width, label='真实值', alpha=0.8)
    ax.bar(x_pos + width/2, est_vals, width, label='估计值', alpha=0.8)
    
    ax.set_ylabel('参数值')
    ax.set_title('参数估计结果对比')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(params)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 数据分布和聚类结果
    ax = axes[2]
    
    # 按聚类标签统计
    for k in range(2):
        mask = predicted_labels == k
        counts_0 = np.sum((X[mask] == 0))
        counts_1 = np.sum((X[mask] == 1))
        total = counts_0 + counts_1
        
        x_pos = [k - 0.15, k + 0.15]
        heights = [counts_0, counts_1]
        colors = ['lightcoral', 'lightblue']
        
        ax.bar(x_pos, heights, width=0.3, color=colors, alpha=0.7, 
               edgecolor='black', linewidth=1)
    
    ax.set_xlabel('聚类')
    ax.set_ylabel('样本数量')
    ax.set_title('聚类结果分布')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['聚类0', '聚类1'])
    ax.legend(['值=0', '值=1'])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('em/em_mixture_bernoulli.png', dpi=300, bbox_inches='tight')
    print("\n图像已保存: em/em_mixture_bernoulli.png")
    plt.close()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("EM算法 (Expectation-Maximization) - 完整演示")
    print("=" * 70)
    
    # 运行示例
    demo1_coin_flip()
    demo2_mixture_bernoulli()
    
    print("\n" + "=" * 70)
    print("所有演示完成！")
    print("=" * 70)
    print("\n生成的文件:")
    print("  1. em/em_coin_flip.png - 硬币投掷问题")
    print("  2. em/em_mixture_bernoulli.png - 混合伯努利模型")
    print("\n" + "=" * 70)
