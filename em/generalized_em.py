"""
广义EM算法 (Generalized EM Algorithm, GEM)
==========================================

广义EM算法是标准EM算法的推广，放宽了M步的要求。

理论基础：
---------
标准EM算法：
  M步：θ^(i+1) = argmax_θ Q(θ, θ^(i))
  要求找到Q函数的全局最优解

广义EM算法（GEM）：
  M步：选择θ^(i+1)使得 Q(θ^(i+1), θ^(i)) ≥ Q(θ^(i), θ^(i))
  只要求Q函数值不减少即可（增加或保持不变）

优势：
-----
1. 降低计算复杂度：不需要精确求解最优化问题
2. 更灵活：可以使用梯度上升、坐标上升等方法
3. 仍保证收敛：似然函数单调非递减

变体：
-----
1. GEM（一步梯度上升）
2. ECM（期望条件最大化）
3. SAGE（空间交替广义EM）
4. AECM（交替ECM）

应用场景：
---------
- M步难以精确求解的模型
- 高维参数空间
- 需要快速迭代的场景
- 在线学习和流数据处理

本文件实现：
-----------
1. 广义EM算法基类
2. 使用梯度上升的GEM
3. 混合高斯模型的GEM示例
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from typing import Tuple, Optional, Dict, Any, Callable
from abc import ABC, abstractmethod

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class GeneralizedEMAlgorithm(ABC):
    """
    广义EM算法抽象基类
    
    与标准EM的区别：
    - M步只需要使Q函数值增加，不要求达到最大值
    """
    
    def __init__(self, max_iter: int = 100, tol: float = 1e-6, 
                 m_step_iterations: int = 1, verbose: bool = True):
        """
        Parameters:
        -----------
        max_iter : int
            最大EM迭代次数
        tol : float
            收敛阈值
        m_step_iterations : int
            M步的迭代次数（如梯度上升的步数）
        verbose : bool
            是否打印信息
        """
        self.max_iter = max_iter
        self.tol = tol
        self.m_step_iterations = m_step_iterations
        self.verbose = verbose
        self.log_likelihoods_ = []
        self.q_values_ = []  # 记录Q函数值
        self.n_iter_ = 0
        
    @abstractmethod
    def e_step(self, X: np.ndarray) -> Any:
        """E步：计算Q函数或后验概率"""
        pass
    
    @abstractmethod
    def gem_m_step(self, X: np.ndarray, e_step_result: Any):
        """
        广义M步：使Q函数值增加
        
        可以是：
        - 一步或多步梯度上升
        - 坐标上升
        - 其他优化方法
        """
        pass
    
    @abstractmethod
    def compute_log_likelihood(self, X: np.ndarray) -> float:
        """计算对数似然"""
        pass
    
    def compute_q_function(self, X: np.ndarray, e_step_result: Any) -> float:
        """
        计算Q函数值（可选实现）
        Q(θ, θ^(i)) = E_Z[log P(Y,Z|θ) | Y, θ^(i)]
        """
        return 0.0  # 默认实现，子类可以覆盖
    
    def fit(self, X: np.ndarray) -> 'GeneralizedEMAlgorithm':
        """拟合GEM算法"""
        self.log_likelihoods_ = []
        self.q_values_ = []
        
        for iteration in range(self.max_iter):
            # E步
            e_step_result = self.e_step(X)
            
            # 记录当前Q函数值（可选）
            q_before = self.compute_q_function(X, e_step_result)
            
            # 广义M步
            self.gem_m_step(X, e_step_result)
            
            # 验证Q函数增加（可选）
            q_after = self.compute_q_function(X, e_step_result)
            self.q_values_.append((q_before, q_after))
            
            # 计算对数似然
            log_likelihood = self.compute_log_likelihood(X)
            self.log_likelihoods_.append(log_likelihood)
            
            if self.verbose and (iteration % 10 == 0 or iteration < 5):
                print(f"Iteration {iteration}: Log-Likelihood = {log_likelihood:.6f}")
                if q_before != 0.0 and q_after != 0.0:
                    print(f"  Q函数: {q_before:.6f} -> {q_after:.6f} (Δ={q_after-q_before:.6f})")
            
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


class GradientAscentGEM(GeneralizedEMAlgorithm):
    """
    使用梯度上升的广义EM算法
    
    M步使用梯度上升来增加Q函数，而不是精确求最大值。
    这个示例实现了一个简单的混合伯努利模型。
    """
    
    def __init__(self, n_components: int = 2, learning_rate: float = 0.01,
                 max_iter: int = 100, tol: float = 1e-6, 
                 m_step_iterations: int = 5, verbose: bool = True):
        """
        Parameters:
        -----------
        n_components : int
            混合成分数量
        learning_rate : float
            梯度上升的学习率
        m_step_iterations : int
            每次M步的梯度上升步数
        """
        super().__init__(max_iter, tol, m_step_iterations, verbose)
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.weights_ = None
        self.means_ = None
        
    def initialize_parameters(self, random_state: Optional[int] = None):
        """初始化参数"""
        if random_state is not None:
            np.random.seed(random_state)
        self.weights_ = np.random.dirichlet(np.ones(self.n_components))
        self.means_ = np.random.uniform(0.3, 0.7, self.n_components)
    
    def e_step(self, X: np.ndarray) -> np.ndarray:
        """E步：计算后验概率"""
        n_samples = len(X)
        responsibilities = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            likelihood = (self.means_[k] ** X) * ((1 - self.means_[k]) ** (1 - X))
            responsibilities[:, k] = self.weights_[k] * likelihood
        
        # 归一化
        responsibilities /= responsibilities.sum(axis=1, keepdims=True) + 1e-10
        
        return responsibilities
    
    def gem_m_step(self, X: np.ndarray, responsibilities: np.ndarray):
        """
        广义M步：使用梯度上升更新参数
        
        进行m_step_iterations步梯度上升
        """
        for _ in range(self.m_step_iterations):
            # 计算梯度
            n_k = responsibilities.sum(axis=0)
            
            # 更新means的梯度
            for k in range(self.n_components):
                # Q函数对μ_k的偏导数
                # ∂Q/∂μ_k = Σ_i γ_ik * [x_i/μ_k - (1-x_i)/(1-μ_k)]
                grad_mean = np.sum(responsibilities[:, k] * 
                                  (X / (self.means_[k] + 1e-10) - 
                                   (1 - X) / (1 - self.means_[k] + 1e-10)))
                
                # 梯度上升更新
                self.means_[k] += self.learning_rate * grad_mean
                
                # 投影到有效范围[0.01, 0.99]
                self.means_[k] = np.clip(self.means_[k], 0.01, 0.99)
            
            # 更新weights（可以用闭式解，因为很简单）
            self.weights_ = n_k / len(X)
            
            # 归一化weights
            self.weights_ /= self.weights_.sum()
    
    def compute_log_likelihood(self, X: np.ndarray) -> float:
        """计算对数似然"""
        n_samples = len(X)
        log_likelihood = 0.0
        
        for i in range(n_samples):
            likelihood = 0.0
            for k in range(self.n_components):
                p_x = (self.means_[k] ** X[i]) * ((1 - self.means_[k]) ** (1 - X[i]))
                likelihood += self.weights_[k] * p_x
            log_likelihood += np.log(likelihood + 1e-10)
        
        return log_likelihood
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测聚类标签"""
        responsibilities = self.e_step(X)
        return np.argmax(responsibilities, axis=1)


class SimpleGaussianMixtureGEM(GeneralizedEMAlgorithm):
    """
    简化的一维高斯混合模型的GEM算法
    
    使用坐标上升法更新参数：
    - 固定其他参数，只更新均值
    - 固定其他参数，只更新方差
    - 轮流进行
    """
    
    def __init__(self, n_components: int = 2, max_iter: int = 100, 
                 tol: float = 1e-6, m_step_iterations: int = 3, verbose: bool = True):
        """
        Parameters:
        -----------
        n_components : int
            混合成分数量
        m_step_iterations : int
            坐标上升的迭代次数
        """
        super().__init__(max_iter, tol, m_step_iterations, verbose)
        self.n_components = n_components
        self.weights_ = None
        self.means_ = None
        self.variances_ = None
        
    def initialize_parameters(self, X: np.ndarray, random_state: Optional[int] = None):
        """初始化参数"""
        if random_state is not None:
            np.random.seed(random_state)
        
        self.weights_ = np.ones(self.n_components) / self.n_components
        
        # 从数据中随机选择初始均值
        idx = np.random.choice(len(X), self.n_components, replace=False)
        self.means_ = X[idx].copy()
        
        # 初始方差设为数据方差
        self.variances_ = np.full(self.n_components, np.var(X))
    
    def _gaussian_pdf(self, x: np.ndarray, mean: float, variance: float) -> np.ndarray:
        """计算高斯概率密度"""
        return (1.0 / np.sqrt(2 * np.pi * variance)) * \
               np.exp(-0.5 * ((x - mean) ** 2) / variance)
    
    def e_step(self, X: np.ndarray) -> np.ndarray:
        """E步：计算responsibilities"""
        n_samples = len(X)
        responsibilities = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            responsibilities[:, k] = self.weights_[k] * \
                self._gaussian_pdf(X, self.means_[k], self.variances_[k])
        
        # 归一化
        responsibilities /= responsibilities.sum(axis=1, keepdims=True) + 1e-10
        
        return responsibilities
    
    def gem_m_step(self, X: np.ndarray, responsibilities: np.ndarray):
        """
        广义M步：使用坐标上升
        
        轮流更新：
        1. 固定variance，更新mean
        2. 固定mean，更新variance  
        3. 更新weights
        """
        n_samples = len(X)
        
        for _ in range(self.m_step_iterations):
            n_k = responsibilities.sum(axis=0)
            
            # 1. 更新均值（固定方差）
            for k in range(self.n_components):
                self.means_[k] = np.sum(responsibilities[:, k] * X) / (n_k[k] + 1e-10)
            
            # 2. 更新方差（固定均值）
            for k in range(self.n_components):
                self.variances_[k] = np.sum(responsibilities[:, k] * 
                                           (X - self.means_[k]) ** 2) / (n_k[k] + 1e-10)
                # 防止方差过小
                self.variances_[k] = max(self.variances_[k], 1e-3)
            
            # 3. 更新weights
            self.weights_ = n_k / n_samples
    
    def compute_log_likelihood(self, X: np.ndarray) -> float:
        """计算对数似然"""
        n_samples = len(X)
        log_likelihood = 0.0
        
        for i in range(n_samples):
            likelihood = 0.0
            for k in range(self.n_components):
                likelihood += self.weights_[k] * \
                    self._gaussian_pdf(X[i], self.means_[k], self.variances_[k])
            log_likelihood += np.log(likelihood + 1e-10)
        
        return log_likelihood
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测聚类标签"""
        responsibilities = self.e_step(X)
        return np.argmax(responsibilities, axis=1)


def demo1_gem_vs_em():
    """
    示例1: GEM与标准EM的对比
    使用混合伯努利模型
    """
    print("=" * 70)
    print("示例1: 广义EM算法 vs 标准EM算法")
    print("=" * 70)
    
    # 生成数据
    np.random.seed(42)
    n_samples = 300
    true_weights = np.array([0.4, 0.6])
    true_means = np.array([0.25, 0.75])
    
    component_labels = np.random.choice(2, size=n_samples, p=true_weights)
    X = np.zeros(n_samples)
    for i in range(n_samples):
        X[i] = np.random.binomial(1, true_means[component_labels[i]])
    
    print(f"\n数据: {n_samples}个样本")
    print(f"真实参数: π={true_weights}, μ={true_means}")
    
    # 标准EM（使用闭式解）
    print("\n" + "-" * 70)
    print("标准EM算法（M步使用闭式解）:")
    print("-" * 70)
    
    from em_algorithm import MixtureBernoulliEM
    
    em_model = MixtureBernoulliEM(n_components=2, max_iter=50, verbose=False)
    em_model.initialize_parameters(random_state=123)
    em_model.fit(X)
    
    print(f"迭代次数: {em_model.n_iter_}")
    print(f"最终对数似然: {em_model.log_likelihoods_[-1]:.6f}")
    
    # 广义EM（使用梯度上升）
    print("\n" + "-" * 70)
    print("广义EM算法（M步使用梯度上升）:")
    print("-" * 70)
    
    gem_model = GradientAscentGEM(n_components=2, learning_rate=0.01, 
                                  max_iter=50, m_step_iterations=5, verbose=False)
    gem_model.initialize_parameters(random_state=123)
    gem_model.fit(X)
    
    print(f"迭代次数: {gem_model.n_iter_}")
    print(f"最终对数似然: {gem_model.log_likelihoods_[-1]:.6f}")
    
    # 可视化对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 对数似然对比
    ax = axes[0]
    ax.plot(range(len(em_model.log_likelihoods_)), em_model.log_likelihoods_, 
            'o-', linewidth=2, label='标准EM（闭式解）', markersize=5)
    ax.plot(range(len(gem_model.log_likelihoods_)), gem_model.log_likelihoods_, 
            's-', linewidth=2, label='广义EM（梯度上升）', markersize=5)
    ax.set_xlabel('迭代次数')
    ax.set_ylabel('对数似然')
    ax.set_title('EM vs GEM: 收敛速度对比')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 参数估计对比
    ax = axes[1]
    
    # 对齐参数
    if em_model.means_[0] > em_model.means_[1]:
        em_model.weights_ = em_model.weights_[::-1]
        em_model.means_ = em_model.means_[::-1]
    
    if gem_model.means_[0] > gem_model.means_[1]:
        gem_model.weights_ = gem_model.weights_[::-1]
        gem_model.means_ = gem_model.means_[::-1]
    
    params = ['π_0', 'π_1', 'μ_0', 'μ_1']
    true_vals = [true_weights[0], true_weights[1], true_means[0], true_means[1]]
    em_vals = [em_model.weights_[0], em_model.weights_[1], 
               em_model.means_[0], em_model.means_[1]]
    gem_vals = [gem_model.weights_[0], gem_model.weights_[1], 
                gem_model.means_[0], gem_model.means_[1]]
    
    x_pos = np.arange(len(params))
    width = 0.25
    
    ax.bar(x_pos - width, true_vals, width, label='真实值', alpha=0.8)
    ax.bar(x_pos, em_vals, width, label='标准EM', alpha=0.8)
    ax.bar(x_pos + width, gem_vals, width, label='广义EM', alpha=0.8)
    
    ax.set_ylabel('参数值')
    ax.set_title('参数估计结果对比')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(params)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('em/gem_vs_em.png', dpi=300, bbox_inches='tight')
    print("\n图像已保存: em/gem_vs_em.png")
    plt.close()


def demo2_coordinate_ascent_gem():
    """
    示例2: 使用坐标上升的GEM算法
    一维高斯混合模型
    """
    print("\n" + "=" * 70)
    print("示例2: 坐标上升GEM算法（一维高斯混合模型）")
    print("=" * 70)
    
    # 生成数据
    np.random.seed(42)
    n_samples = 400
    
    # 两个高斯分布
    n1 = int(n_samples * 0.4)
    n2 = n_samples - n1
    
    X1 = np.random.normal(loc=-2, scale=0.8, size=n1)
    X2 = np.random.normal(loc=2, scale=1.2, size=n2)
    X = np.concatenate([X1, X2])
    np.random.shuffle(X)
    
    true_labels = np.concatenate([np.zeros(n1), np.ones(n2)])
    
    print(f"\n数据: {n_samples}个样本")
    print(f"真实分布:")
    print(f"  成分0: N(-2, 0.8²), 样本数={n1}")
    print(f"  成分1: N(2, 1.2²), 样本数={n2}")
    
    # GEM算法
    print("\n" + "-" * 70)
    print("坐标上升GEM算法:")
    print("-" * 70)
    
    model = SimpleGaussianMixtureGEM(n_components=2, max_iter=50, 
                                     m_step_iterations=3, verbose=True)
    model.initialize_parameters(X, random_state=123)
    
    print(f"\n初始参数:")
    print(f"  权重: {model.weights_}")
    print(f"  均值: {model.means_}")
    print(f"  方差: {model.variances_}")
    print()
    
    model.fit(X)
    
    # 对齐参数
    if model.means_[0] > model.means_[1]:
        model.weights_ = model.weights_[::-1]
        model.means_ = model.means_[::-1]
        model.variances_ = model.variances_[::-1]
    
    print(f"\n估计参数:")
    print(f"  权重: {model.weights_}")
    print(f"  均值: {model.means_}")
    print(f"  标准差: {np.sqrt(model.variances_)}")
    
    print(f"\n真实参数:")
    print(f"  权重: [{n1/n_samples:.3f}, {n2/n_samples:.3f}]")
    print(f"  均值: [-2.0, 2.0]")
    print(f"  标准差: [0.8, 1.2]")
    
    # 预测聚类
    predicted_labels = model.predict(X)
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 对数似然曲线
    ax = axes[0, 0]
    ax.plot(range(len(model.log_likelihoods_)), model.log_likelihoods_, 
            'o-', linewidth=2, markersize=6)
    ax.set_xlabel('迭代次数')
    ax.set_ylabel('对数似然')
    ax.set_title('GEM算法收敛过程')
    ax.grid(True, alpha=0.3)
    
    # 数据分布和拟合结果
    ax = axes[0, 1]
    ax.hist(X, bins=50, density=True, alpha=0.6, color='gray', 
            edgecolor='black', label='数据分布')
    
    # 绘制拟合的混合高斯分布
    x_range = np.linspace(X.min(), X.max(), 1000)
    
    for k in range(2):
        y = model.weights_[k] * model._gaussian_pdf(x_range, model.means_[k], 
                                                      model.variances_[k])
        ax.plot(x_range, y, linewidth=2, label=f'成分{k}')
    
    # 总混合分布
    y_total = sum(model.weights_[k] * model._gaussian_pdf(x_range, model.means_[k], 
                                                            model.variances_[k])
                  for k in range(2))
    ax.plot(x_range, y_total, 'k--', linewidth=2, label='混合分布')
    
    ax.set_xlabel('x')
    ax.set_ylabel('概率密度')
    ax.set_title('高斯混合模型拟合结果')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 聚类结果（散点图）
    ax = axes[1, 0]
    colors = ['red', 'blue']
    for k in range(2):
        mask = predicted_labels == k
        ax.scatter(X[mask], np.zeros(np.sum(mask)), 
                  c=colors[k], alpha=0.5, s=20, label=f'聚类{k}')
    
    # 绘制均值位置
    for k in range(2):
        ax.axvline(model.means_[k], color=colors[k], linestyle='--', 
                  linewidth=2, label=f'μ_{k}={model.means_[k]:.2f}')
    
    ax.set_xlabel('x')
    ax.set_ylabel('')
    ax.set_yticks([])
    ax.set_title('聚类结果')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 参数演化（如果记录了历史）
    ax = axes[1, 1]
    ax.text(0.5, 0.5, '坐标上升GEM算法\n\n'
            '特点：\n'
            '• 轮流更新参数\n'
            '• 每次只优化一组参数\n'
            '• 其他参数保持不变\n\n'
            '优势：\n'
            '• 降低计算复杂度\n'
            '• 避免求解复杂优化问题\n'
            '• 适合高维参数空间',
            ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('em/gem_coordinate_ascent.png', dpi=300, bbox_inches='tight')
    print("\n图像已保存: em/gem_coordinate_ascent.png")
    plt.close()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("广义EM算法 (Generalized EM Algorithm) - 完整演示")
    print("=" * 70)
    
    # 运行示例
    demo1_gem_vs_em()
    demo2_coordinate_ascent_gem()
    
    print("\n" + "=" * 70)
    print("所有演示完成！")
    print("=" * 70)
    print("\n生成的文件:")
    print("  1. em/gem_vs_em.png - GEM vs 标准EM对比")
    print("  2. em/gem_coordinate_ascent.png - 坐标上升GEM算法")
    print("\n" + "=" * 70)
