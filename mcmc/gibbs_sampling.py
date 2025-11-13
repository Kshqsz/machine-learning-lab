"""
吉布斯抽样 (Gibbs Sampling)
============================

吉布斯抽样是Metropolis-Hastings算法的一个特例，通过逐个采样每个维度的条件分布来生成样本。
当条件分布易于采样时，吉布斯抽样特别高效，且接受率为100%。

理论基础
--------

1. **马尔可夫链蒙特卡洛 (MCMC)**:
   - 构造马尔可夫链，使其平稳分布为目标分布
   - 通过迭代采样达到平稳分布
   
2. **条件分布采样**:
   对于d维分布 π(x₁, x₂, ..., xₐ)，每次固定其他维度，采样第i维：
   
   xᵢ⁽ᵗ⁺¹⁾ ~ π(xᵢ | x₁⁽ᵗ⁺¹⁾, ..., xᵢ₋₁⁽ᵗ⁺¹⁾, xᵢ₊₁⁽ᵗ⁾, ..., xₐ⁽ᵗ⁾)
   
3. **细致平衡条件**:
   吉布斯采样满足细致平衡条件，保证收敛到目标分布：
   
   π(x)P(x→x') = π(x')P(x'→x)

4. **优势**:
   - 接受率100%（无需拒绝步骤）
   - 当条件分布简单时效率高
   - 适合高维问题
   
5. **劣势**:
   - 需要知道条件分布
   - 变量强相关时混合慢
   - 可能陷入局部模式

算法步骤
--------

1. 初始化 x⁽⁰⁾ = (x₁⁽⁰⁾, x₂⁽⁰⁾, ..., xₐ⁽⁰⁾)
2. 对于 t = 0, 1, 2, ..., n_iterations-1:
   - 采样 x₁⁽ᵗ⁺¹⁾ ~ π(x₁ | x₂⁽ᵗ⁾, x₃⁽ᵗ⁾, ..., xₐ⁽ᵗ⁾)
   - 采样 x₂⁽ᵗ⁺¹⁾ ~ π(x₂ | x₁⁽ᵗ⁺¹⁾, x₃⁽ᵗ⁾, ..., xₐ⁽ᵗ⁾)
   - ...
   - 采样 xₐ⁽ᵗ⁺¹⁾ ~ π(xₐ | x₁⁽ᵗ⁺¹⁾, x₂⁽ᵗ⁺¹⁾, ..., xₐ₋₁⁽ᵗ⁺¹⁾)
3. 丢弃burn-in期样本
4. 可选：进行thinning减少相关性

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Callable, Optional, Tuple, Dict, List
import warnings

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class GibbsSampler:
    """
    吉布斯抽样器
    
    通过逐维度采样条件分布来生成多维分布的样本
    
    参数
    ----
    conditional_samplers : List[Callable]
        条件采样函数列表，每个函数接受当前状态和维度索引，返回该维度的新样本
    dim : int
        维度数
    initial_state : Optional[np.ndarray]
        初始状态，如果为None则随机初始化
    """
    
    def __init__(
        self,
        conditional_samplers: List[Callable],
        dim: int,
        initial_state: Optional[np.ndarray] = None
    ):
        self.conditional_samplers = conditional_samplers
        self.dim = dim
        
        if initial_state is not None:
            self.current_state = initial_state.copy()
        else:
            self.current_state = np.zeros(dim)
        
        self.samples = []
        self.n_iterations = 0
        
    def sample_step(self) -> np.ndarray:
        """
        执行一步吉布斯采样（遍历所有维度）
        
        返回
        ----
        np.ndarray : 新状态
        """
        for i in range(self.dim):
            # 采样第i维的条件分布
            self.current_state[i] = self.conditional_samplers[i](self.current_state, i)
        
        return self.current_state.copy()
    
    def sample(
        self,
        n_iterations: int,
        burn_in: int = 1000,
        thin: int = 1,
        verbose: bool = True
    ) -> np.ndarray:
        """
        运行吉布斯采样
        
        参数
        ----
        n_iterations : int
            迭代次数
        burn_in : int
            burn-in期长度
        thin : int
            thinning间隔
        verbose : bool
            是否显示进度
            
        返回
        ----
        np.ndarray : 采样结果，形状为 (n_samples, dim)
        """
        self.samples = []
        total_iterations = burn_in + n_iterations * thin
        
        for t in range(total_iterations):
            state = self.sample_step()
            
            # burn-in后，按thin间隔保存样本
            if t >= burn_in and (t - burn_in) % thin == 0:
                self.samples.append(state.copy())
            
            if verbose and (t + 1) % max(1, total_iterations // 10) == 0:
                print(f"迭代 {t + 1}/{total_iterations} ({100 * (t + 1) / total_iterations:.1f}%)")
        
        self.samples = np.array(self.samples)
        self.n_iterations = len(self.samples)
        
        if verbose:
            print(f"\n完成! 生成 {self.n_iterations} 个样本")
            print(f"样本均值: {np.mean(self.samples, axis=0)}")
            print(f"样本标准差: {np.std(self.samples, axis=0)}")
        
        return self.samples
    
    def get_statistics(self) -> Dict:
        """
        计算采样统计信息
        
        返回
        ----
        Dict : 包含均值、标准差、协方差等统计信息
        """
        if len(self.samples) == 0:
            raise ValueError("尚未生成样本，请先调用sample()方法")
        
        stats_dict = {
            'mean': np.mean(self.samples, axis=0),
            'std': np.std(self.samples, axis=0),
            'median': np.median(self.samples, axis=0),
            'min': np.min(self.samples, axis=0),
            'max': np.max(self.samples, axis=0),
            'n_samples': len(self.samples),
            'dim': self.dim
        }
        
        # 如果是多维，计算协方差和相关系数
        if self.dim > 1:
            stats_dict['cov'] = np.cov(self.samples.T)
            stats_dict['corr'] = np.corrcoef(self.samples.T)
        
        return stats_dict
    
    def compute_acf(self, max_lag: int = 50) -> np.ndarray:
        """
        计算自相关函数 (ACF)
        
        参数
        ----
        max_lag : int
            最大滞后
            
        返回
        ----
        np.ndarray : ACF值，形状为 (max_lag + 1, dim)
        """
        if len(self.samples) == 0:
            raise ValueError("尚未生成样本")
        
        acf = np.zeros((max_lag + 1, self.dim))
        
        for d in range(self.dim):
            x = self.samples[:, d]
            x_centered = x - np.mean(x)
            c0 = np.dot(x_centered, x_centered) / len(x)
            
            for lag in range(max_lag + 1):
                if lag == 0:
                    acf[lag, d] = 1.0
                else:
                    c_lag = np.dot(x_centered[:-lag], x_centered[lag:]) / len(x)
                    acf[lag, d] = c_lag / c0
        
        return acf
    
    def compute_ess(self, max_lag: int = 50) -> np.ndarray:
        """
        计算有效样本量 (ESS)
        
        使用公式: ESS = n / (1 + 2 * Σ ρₖ)
        其中 ρₖ 是滞后k的自相关系数
        
        参数
        ----
        max_lag : int
            最大滞后
            
        返回
        ----
        np.ndarray : 每个维度的ESS
        """
        acf = self.compute_acf(max_lag)
        n = len(self.samples)
        
        ess = np.zeros(self.dim)
        for d in range(self.dim):
            # 累加正的自相关系数，直到遇到负值
            sum_acf = 0.0
            for lag in range(1, max_lag + 1):
                if acf[lag, d] > 0:
                    sum_acf += acf[lag, d]
                else:
                    break
            
            ess[d] = n / (1.0 + 2.0 * sum_acf)
        
        return ess


# ============================================================================
# 示例应用
# ============================================================================

def demo_1_bivariate_normal():
    """
    示例1: 二元正态分布
    
    目标分布: N([μ₁, μ₂], Σ)
    其中 Σ = [[σ₁², ρσ₁σ₂], [ρσ₁σ₂, σ₂²]]
    
    条件分布:
    - X₁|X₂ ~ N(μ₁ + ρ(σ₁/σ₂)(x₂ - μ₂), σ₁²(1 - ρ²))
    - X₂|X₁ ~ N(μ₂ + ρ(σ₂/σ₁)(x₁ - μ₁), σ₂²(1 - ρ²))
    """
    print("=" * 70)
    print("示例1: 二元正态分布的吉布斯采样")
    print("=" * 70)
    
    # 参数设置
    mu = np.array([2.0, -1.0])
    sigma = np.array([1.5, 0.8])
    rho = 0.7
    
    print(f"\n目标分布: 二元正态分布")
    print(f"均值: μ = {mu}")
    print(f"标准差: σ = {sigma}")
    print(f"相关系数: ρ = {rho}")
    
    # 定义条件采样函数
    def sample_x1_given_x2(state, i):
        """X₁|X₂的条件分布"""
        x2 = state[1]
        cond_mean = mu[0] + rho * (sigma[0] / sigma[1]) * (x2 - mu[1])
        cond_std = sigma[0] * np.sqrt(1 - rho**2)
        return np.random.normal(cond_mean, cond_std)
    
    def sample_x2_given_x1(state, i):
        """X₂|X₁的条件分布"""
        x1 = state[0]
        cond_mean = mu[1] + rho * (sigma[1] / sigma[0]) * (x1 - mu[0])
        cond_std = sigma[1] * np.sqrt(1 - rho**2)
        return np.random.normal(cond_mean, cond_std)
    
    # 创建采样器
    sampler = GibbsSampler(
        conditional_samplers=[sample_x1_given_x2, sample_x2_given_x1],
        dim=2,
        initial_state=np.array([0.0, 0.0])
    )
    
    # 采样
    samples = sampler.sample(n_iterations=5000, burn_in=1000, thin=2, verbose=True)
    
    # 统计信息
    stats_dict = sampler.get_statistics()
    print(f"\n样本统计:")
    print(f"均值: {stats_dict['mean']}")
    print(f"标准差: {stats_dict['std']}")
    print(f"相关系数矩阵:\n{stats_dict['corr']}")
    
    # 计算ESS
    ess = sampler.compute_ess()
    print(f"\n有效样本量:")
    print(f"X₁: {ess[0]:.2f} ({100 * ess[0] / len(samples):.1f}%)")
    print(f"X₂: {ess[1]:.2f} ({100 * ess[1] / len(samples):.1f}%)")
    
    # 真实分布（用于对比）
    cov_matrix = np.array([
        [sigma[0]**2, rho * sigma[0] * sigma[1]],
        [rho * sigma[0] * sigma[1], sigma[1]**2]
    ])
    
    print(f"\n真实参数:")
    print(f"均值: {mu}")
    print(f"标准差: {sigma}")
    print(f"相关系数: {rho}")
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 轨迹图 - X₁
    axes[0, 0].plot(samples[:500, 0], alpha=0.7)
    axes[0, 0].axhline(mu[0], color='r', linestyle='--', label=f'真实均值={mu[0]}')
    axes[0, 0].set_xlabel('迭代次数')
    axes[0, 0].set_ylabel('X₁')
    axes[0, 0].set_title('X₁的轨迹图（前500次）')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 轨迹图 - X₂
    axes[1, 0].plot(samples[:500, 1], alpha=0.7, color='orange')
    axes[1, 0].axhline(mu[1], color='r', linestyle='--', label=f'真实均值={mu[1]}')
    axes[1, 0].set_xlabel('迭代次数')
    axes[1, 0].set_ylabel('X₂')
    axes[1, 0].set_title('X₂的轨迹图（前500次）')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 3. 边缘分布 - X₁
    axes[0, 1].hist(samples[:, 0], bins=50, density=True, alpha=0.6, edgecolor='black')
    x1_range = np.linspace(samples[:, 0].min(), samples[:, 0].max(), 100)
    axes[0, 1].plot(x1_range, stats.norm.pdf(x1_range, mu[0], sigma[0]), 
                    'r-', linewidth=2, label='真实分布')
    axes[0, 1].set_xlabel('X₁')
    axes[0, 1].set_ylabel('密度')
    axes[0, 1].set_title('X₁的边缘分布')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 4. 边缘分布 - X₂
    axes[1, 1].hist(samples[:, 1], bins=50, density=True, alpha=0.6, 
                    color='orange', edgecolor='black')
    x2_range = np.linspace(samples[:, 1].min(), samples[:, 1].max(), 100)
    axes[1, 1].plot(x2_range, stats.norm.pdf(x2_range, mu[1], sigma[1]), 
                    'r-', linewidth=2, label='真实分布')
    axes[1, 1].set_xlabel('X₂')
    axes[1, 1].set_ylabel('密度')
    axes[1, 1].set_title('X₂的边缘分布')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 5. 联合分布散点图
    axes[0, 2].scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=1)
    axes[0, 2].scatter(mu[0], mu[1], color='red', s=100, marker='x', 
                       linewidths=3, label='真实均值')
    
    # 添加真实分布的等高线
    x1_grid = np.linspace(samples[:, 0].min(), samples[:, 0].max(), 100)
    x2_grid = np.linspace(samples[:, 1].min(), samples[:, 1].max(), 100)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    pos = np.dstack((X1, X2))
    rv = stats.multivariate_normal(mu, cov_matrix)
    axes[0, 2].contour(X1, X2, rv.pdf(pos), colors='red', alpha=0.6, levels=5)
    
    axes[0, 2].set_xlabel('X₁')
    axes[0, 2].set_ylabel('X₂')
    axes[0, 2].set_title('联合分布')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 6. 自相关函数
    acf = sampler.compute_acf(max_lag=50)
    lags = np.arange(len(acf))
    axes[1, 2].plot(lags, acf[:, 0], 'o-', label='X₁', markersize=3)
    axes[1, 2].plot(lags, acf[:, 1], 's-', label='X₂', markersize=3)
    axes[1, 2].axhline(0, color='black', linestyle='--', alpha=0.3)
    axes[1, 2].set_xlabel('滞后 (Lag)')
    axes[1, 2].set_ylabel('ACF')
    axes[1, 2].set_title('自相关函数')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mcmc/gibbs_bivariate_normal.png', dpi=150, bbox_inches='tight')
    print("\n图片已保存: mcmc/gibbs_bivariate_normal.png")
    plt.close()


def demo_2_mixture_model():
    """
    示例2: 混合模型的数据增广吉布斯采样
    
    观测模型: y ~ 0.3*N(-2, 1²) + 0.7*N(2, 1.5²)
    
    引入隐变量 z ∈ {0, 1} 表示组别
    - P(z=0) = 0.3, P(z=1) = 0.7
    - P(y|z=0) = N(-2, 1²)
    - P(y|z=1) = N(2, 1.5²)
    
    吉布斯采样:
    1. 采样 z|y (根据后验概率)
    2. 采样 y|z (从对应的正态分布)
    """
    print("\n" + "=" * 70)
    print("示例2: 混合模型的数据增广吉布斯采样")
    print("=" * 70)
    
    # 混合模型参数
    weights = np.array([0.3, 0.7])
    means = np.array([-2.0, 2.0])
    stds = np.array([1.0, 1.5])
    
    print(f"\n混合模型:")
    print(f"权重: {weights}")
    print(f"均值: {means}")
    print(f"标准差: {stds}")
    
    # 生成观测数据
    np.random.seed(42)
    n_obs = 100
    true_z = np.random.choice(2, size=n_obs, p=weights)
    y_obs = np.array([np.random.normal(means[z], stds[z]) for z in true_z])
    
    print(f"\n生成 {n_obs} 个观测值")
    print(f"真实组别分布: 组0={np.sum(true_z == 0)}, 组1={np.sum(true_z == 1)}")
    
    # 为每个观测值创建吉布斯采样器
    # 这里我们采样 (z, μ₀, μ₁) 的后验分布
    # 简化起见，我们假设标准差已知，只估计均值和组别
    
    # 使用共轭先验: μₖ ~ N(0, 10²)
    prior_mean = 0.0
    prior_std = 10.0
    
    # 初始化
    z_samples = []
    mu_samples = []
    
    # 初始值
    current_z = np.random.choice(2, size=n_obs)
    current_mu = np.random.normal(0, 1, size=2)
    
    n_iterations = 2000
    burn_in = 500
    
    print(f"\n开始吉布斯采样...")
    
    for it in range(n_iterations):
        # 1. 采样 z_i | y_i, μ (对每个观测)
        for i in range(n_obs):
            # 计算后验概率
            log_prob = np.zeros(2)
            for k in range(2):
                log_prob[k] = (np.log(weights[k]) + 
                              stats.norm.logpdf(y_obs[i], current_mu[k], stds[k]))
            
            # 归一化
            prob = np.exp(log_prob - np.max(log_prob))
            prob = prob / prob.sum()
            
            # 采样
            current_z[i] = np.random.choice(2, p=prob)
        
        # 2. 采样 μₖ | y, z (对每个组别)
        for k in range(2):
            mask = (current_z == k)
            n_k = mask.sum()
            
            if n_k > 0:
                # 后验分布的参数（共轭）
                y_k = y_obs[mask]
                
                # 精度（方差的倒数）
                prior_precision = 1 / (prior_std ** 2)
                likelihood_precision = n_k / (stds[k] ** 2)
                posterior_precision = prior_precision + likelihood_precision
                posterior_var = 1 / posterior_precision
                
                # 均值
                posterior_mean = posterior_var * (
                    prior_precision * prior_mean + 
                    likelihood_precision * y_k.mean()
                )
                
                current_mu[k] = np.random.normal(posterior_mean, np.sqrt(posterior_var))
            else:
                # 如果组为空，从先验采样
                current_mu[k] = np.random.normal(prior_mean, prior_std)
        
        # 保存样本（burn-in后）
        if it >= burn_in:
            z_samples.append(current_z.copy())
            mu_samples.append(current_mu.copy())
        
        if (it + 1) % 500 == 0:
            print(f"迭代 {it + 1}/{n_iterations}")
    
    z_samples = np.array(z_samples)
    mu_samples = np.array(mu_samples)
    
    print(f"\n完成! 生成 {len(mu_samples)} 个样本")
    
    # 估计的均值（需要处理标签切换问题）
    mu_est = mu_samples.mean(axis=0)
    
    # 如果估计的μ₀ > μ₁，说明标签反了
    if mu_est[0] > mu_est[1]:
        mu_est = mu_est[::-1]
        print("\n检测到标签切换，已修正")
    
    print(f"\n估计的均值:")
    print(f"μ₀ = {mu_est[0]:.3f} (真实: {means[0]})")
    print(f"μ₁ = {mu_est[1]:.3f} (真实: {means[1]})")
    
    # 估计的组别（使用后验众数）
    z_est = stats.mode(z_samples, axis=0)[0][0]
    if mu_est[0] > mu_est[1]:
        z_est = 1 - z_est
    
    accuracy = np.mean(z_est == true_z)
    print(f"\n组别分类准确率: {100 * accuracy:.1f}%")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 观测数据直方图
    axes[0, 0].hist(y_obs, bins=30, density=True, alpha=0.6, edgecolor='black')
    
    # 真实混合分布
    y_range = np.linspace(y_obs.min(), y_obs.max(), 200)
    true_density = (weights[0] * stats.norm.pdf(y_range, means[0], stds[0]) +
                   weights[1] * stats.norm.pdf(y_range, means[1], stds[1]))
    axes[0, 0].plot(y_range, true_density, 'r-', linewidth=2, label='真实分布')
    
    # 估计的混合分布
    est_density = (weights[0] * stats.norm.pdf(y_range, mu_est[0], stds[0]) +
                  weights[1] * stats.norm.pdf(y_range, mu_est[1], stds[1]))
    axes[0, 0].plot(y_range, est_density, 'g--', linewidth=2, label='估计分布')
    
    axes[0, 0].set_xlabel('y')
    axes[0, 0].set_ylabel('密度')
    axes[0, 0].set_title('观测数据与混合分布')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. μ₀的轨迹图
    axes[0, 1].plot(mu_samples[:, 0], alpha=0.7)
    axes[0, 1].axhline(means[0], color='r', linestyle='--', label=f'真实值={means[0]}')
    axes[0, 1].axhline(mu_est[0], color='g', linestyle='--', label=f'估计值={mu_est[0]:.2f}')
    axes[0, 1].set_xlabel('迭代次数')
    axes[0, 1].set_ylabel('μ₀')
    axes[0, 1].set_title('μ₀的轨迹图')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. μ₁的轨迹图
    axes[1, 0].plot(mu_samples[:, 1], alpha=0.7, color='orange')
    axes[1, 0].axhline(means[1], color='r', linestyle='--', label=f'真实值={means[1]}')
    axes[1, 0].axhline(mu_est[1], color='g', linestyle='--', label=f'估计值={mu_est[1]:.2f}')
    axes[1, 0].set_xlabel('迭代次数')
    axes[1, 0].set_ylabel('μ₁')
    axes[1, 0].set_title('μ₁的轨迹图')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 两个均值的联合分布
    axes[1, 1].scatter(mu_samples[:, 0], mu_samples[:, 1], alpha=0.3, s=1)
    axes[1, 1].scatter(means[0], means[1], color='red', s=200, marker='x',
                      linewidths=3, label='真实值', zorder=5)
    axes[1, 1].scatter(mu_est[0], mu_est[1], color='green', s=200, marker='+',
                      linewidths=3, label='估计值', zorder=5)
    axes[1, 1].set_xlabel('μ₀')
    axes[1, 1].set_ylabel('μ₁')
    axes[1, 1].set_title('两个均值的联合后验分布')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mcmc/gibbs_mixture_model.png', dpi=150, bbox_inches='tight')
    print("\n图片已保存: mcmc/gibbs_mixture_model.png")
    plt.close()


def demo_3_ising_model():
    """
    示例3: 伊辛模型 (Ising Model) 的吉布斯采样
    
    伊辛模型是统计物理中的经典模型，描述磁性材料的相变。
    
    模型定义:
    - 每个格点 i 有自旋 σᵢ ∈ {-1, +1}
    - 能量函数: E(σ) = -J Σ_{<i,j>} σᵢσⱼ - h Σᵢ σᵢ
    - 玻尔兹曼分布: P(σ) ∝ exp(-βE(σ))
    
    其中:
    - J: 相互作用强度
    - h: 外磁场
    - β: 逆温度 (1/kT)
    
    条件分布:
    P(σᵢ=+1 | σ_{-i}) ∝ exp(β(J Σⱼ σⱼ + h))
    """
    print("\n" + "=" * 70)
    print("示例3: 伊辛模型的吉布斯采样")
    print("=" * 70)
    
    # 参数设置
    grid_size = 20  # 20x20格子
    J = 1.0  # 相互作用强度
    h = 0.0  # 外磁场
    beta = 0.4  # 逆温度（临界点约为 0.44）
    
    print(f"\n伊辛模型参数:")
    print(f"格子大小: {grid_size}x{grid_size}")
    print(f"相互作用强度 J = {J}")
    print(f"外磁场 h = {h}")
    print(f"逆温度 β = {beta}")
    print(f"温度 T = {1/beta:.2f}")
    
    # 初始化：随机自旋
    np.random.seed(42)
    spins = np.random.choice([-1, 1], size=(grid_size, grid_size))
    
    # 邻居索引（周期边界条件）
    def get_neighbors(i, j, size):
        """获取格点(i,j)的四个邻居"""
        return [
            ((i-1) % size, j),
            ((i+1) % size, j),
            (i, (j-1) % size),
            (i, (j+1) % size)
        ]
    
    # 计算平均磁化强度
    def magnetization(spins):
        return np.mean(spins)
    
    # 吉布斯采样
    n_iterations = 10000
    burn_in = 2000
    record_every = 10
    
    magnetizations = []
    spin_snapshots = []
    snapshot_iterations = [0, 500, 2000, 9999]
    
    print(f"\n开始吉布斯采样...")
    
    for it in range(n_iterations):
        # 随机扫描：随机选择一个格点更新
        i = np.random.randint(0, grid_size)
        j = np.random.randint(0, grid_size)
        
        # 计算邻居自旋之和
        neighbor_sum = 0
        for ni, nj in get_neighbors(i, j, grid_size):
            neighbor_sum += spins[ni, nj]
        
        # 计算条件概率 P(σᵢⱼ = +1 | σ_{-ij})
        # P(+1) / P(-1) = exp(2β(J*neighbor_sum + h))
        delta_E = 2 * (J * neighbor_sum + h)
        prob_plus = 1 / (1 + np.exp(-beta * delta_E))
        
        # 采样新的自旋
        spins[i, j] = 1 if np.random.rand() < prob_plus else -1
        
        # 记录磁化强度
        if it >= burn_in and it % record_every == 0:
            magnetizations.append(magnetization(spins))
        
        # 记录快照
        if it in snapshot_iterations:
            spin_snapshots.append(spins.copy())
        
        if (it + 1) % 2000 == 0:
            print(f"迭代 {it + 1}/{n_iterations}, 磁化强度 = {magnetization(spins):.3f}")
    
    magnetizations = np.array(magnetizations)
    
    print(f"\n完成! 生成 {len(magnetizations)} 个样本")
    print(f"平均磁化强度: {magnetizations.mean():.4f} ± {magnetizations.std():.4f}")
    
    # 可视化
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1-4. 自旋配置快照
    for idx, (snapshot, it) in enumerate(zip(spin_snapshots, snapshot_iterations)):
        ax = fig.add_subplot(gs[0, idx])
        im = ax.imshow(snapshot, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
        ax.set_title(f'迭代 {it}')
        ax.axis('off')
        if idx == 3:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 5. 磁化强度轨迹
    ax = fig.add_subplot(gs[1, :2])
    ax.plot(magnetizations, alpha=0.7)
    ax.axhline(0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('采样次数')
    ax.set_ylabel('磁化强度')
    ax.set_title('磁化强度的时间演化')
    ax.grid(True, alpha=0.3)
    
    # 6. 磁化强度分布
    ax = fig.add_subplot(gs[1, 2:])
    ax.hist(magnetizations, bins=50, density=True, alpha=0.6, edgecolor='black')
    ax.axvline(magnetizations.mean(), color='r', linestyle='--', 
               label=f'均值={magnetizations.mean():.4f}')
    ax.set_xlabel('磁化强度')
    ax.set_ylabel('密度')
    ax.set_title('磁化强度分布')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 7. 自相关函数
    ax = fig.add_subplot(gs[2, :2])
    
    # 计算ACF
    max_lag = 200
    m_centered = magnetizations - magnetizations.mean()
    c0 = np.dot(m_centered, m_centered) / len(m_centered)
    acf = np.zeros(max_lag + 1)
    
    for lag in range(max_lag + 1):
        if lag == 0:
            acf[lag] = 1.0
        else:
            c_lag = np.dot(m_centered[:-lag], m_centered[lag:]) / len(m_centered)
            acf[lag] = c_lag / c0
    
    ax.plot(acf, 'o-', markersize=3)
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('滞后 (Lag)')
    ax.set_ylabel('ACF')
    ax.set_title('磁化强度的自相关函数')
    ax.grid(True, alpha=0.3)
    
    # 8. 最终配置的放大图
    ax = fig.add_subplot(gs[2, 2:])
    final_spins = spin_snapshots[-1]
    im = ax.imshow(final_spins, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
    ax.set_title('最终自旋配置')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.savefig('mcmc/gibbs_ising_model.png', dpi=150, bbox_inches='tight')
    print("\n图片已保存: mcmc/gibbs_ising_model.png")
    plt.close()


def demo_4_comparison_correlation():
    """
    示例4: 不同相关性下的吉布斯采样效率对比
    
    比较不同相关系数下吉布斯采样的混合速度和有效样本量
    """
    print("\n" + "=" * 70)
    print("示例4: 相关性对吉布斯采样效率的影响")
    print("=" * 70)
    
    rhos = [0.0, 0.5, 0.9, 0.99]  # 不同的相关系数
    mu = np.array([0.0, 0.0])
    sigma = np.array([1.0, 1.0])
    
    results = []
    
    for rho in rhos:
        print(f"\n测试相关系数 ρ = {rho}")
        
        # 定义条件采样函数
        def sample_x1_given_x2(state, i):
            x2 = state[1]
            cond_mean = mu[0] + rho * (sigma[0] / sigma[1]) * (x2 - mu[1])
            cond_std = sigma[0] * np.sqrt(1 - rho**2)
            return np.random.normal(cond_mean, cond_std)
        
        def sample_x2_given_x1(state, i):
            x1 = state[0]
            cond_mean = mu[1] + rho * (sigma[1] / sigma[0]) * (x1 - mu[0])
            cond_std = sigma[1] * np.sqrt(1 - rho**2)
            return np.random.normal(cond_mean, cond_std)
        
        # 创建采样器
        sampler = GibbsSampler(
            conditional_samplers=[sample_x1_given_x2, sample_x2_given_x1],
            dim=2,
            initial_state=np.array([0.0, 0.0])
        )
        
        # 采样
        samples = sampler.sample(n_iterations=3000, burn_in=500, thin=1, verbose=False)
        
        # 计算统计信息
        ess = sampler.compute_ess(max_lag=100)
        acf = sampler.compute_acf(max_lag=50)
        
        results.append({
            'rho': rho,
            'samples': samples,
            'ess': ess,
            'acf': acf,
            'sampler': sampler
        })
        
        print(f"  有效样本量: X₁={ess[0]:.1f} ({100*ess[0]/len(samples):.1f}%), "
              f"X₂={ess[1]:.1f} ({100*ess[1]/len(samples):.1f}%)")
    
    # 可视化对比
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    for idx, result in enumerate(results):
        rho = result['rho']
        samples = result['samples']
        acf = result['acf']
        ess = result['ess']
        
        # 第一行: 轨迹图（X₁）
        axes[0, idx].plot(samples[:500, 0], alpha=0.7)
        axes[0, idx].axhline(0, color='r', linestyle='--', alpha=0.5)
        axes[0, idx].set_title(f'ρ = {rho}')
        axes[0, idx].set_xlabel('迭代')
        axes[0, idx].set_ylabel('X₁')
        axes[0, idx].grid(True, alpha=0.3)
        
        # 第二行: 自相关函数
        axes[1, idx].plot(acf[:, 0], 'o-', label='X₁', markersize=3)
        axes[1, idx].plot(acf[:, 1], 's-', label='X₂', markersize=3)
        axes[1, idx].axhline(0, color='black', linestyle='--', alpha=0.3)
        axes[1, idx].set_xlabel('滞后')
        axes[1, idx].set_ylabel('ACF')
        axes[1, idx].set_title(f'ESS: {ess[0]:.0f} ({100*ess[0]/len(samples):.0f}%)')
        axes[1, idx].legend()
        axes[1, idx].grid(True, alpha=0.3)
        
        # 第三行: 散点图
        axes[2, idx].scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=1)
        axes[2, idx].set_xlabel('X₁')
        axes[2, idx].set_ylabel('X₂')
        axes[2, idx].set_title(f'样本相关系数: {np.corrcoef(samples.T)[0,1]:.3f}')
        axes[2, idx].grid(True, alpha=0.3)
        axes[2, idx].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('mcmc/gibbs_correlation_comparison.png', dpi=150, bbox_inches='tight')
    print("\n图片已保存: mcmc/gibbs_correlation_comparison.png")
    plt.close()
    
    # 总结
    print("\n" + "=" * 70)
    print("效率总结:")
    print("=" * 70)
    print(f"{'相关系数':^10} {'ESS (X₁)':^15} {'ESS比例':^15} {'混合速度':^15}")
    print("-" * 70)
    
    for result in results:
        rho = result['rho']
        ess = result['ess'][0]
        n_samples = len(result['samples'])
        ess_ratio = ess / n_samples
        
        if ess_ratio > 0.5:
            mixing = "非常好"
        elif ess_ratio > 0.3:
            mixing = "良好"
        elif ess_ratio > 0.1:
            mixing = "中等"
        else:
            mixing = "较差"
        
        print(f"{rho:^10.2f} {ess:^15.1f} {100*ess_ratio:^14.1f}% {mixing:^15}")
    
    print("\n关键发现:")
    print("- 变量相关性越强，吉布斯采样的混合速度越慢")
    print("- 当ρ→1时，条件分布变得非常集中，采样器难以探索整个空间")
    print("- 强相关情况下，需要更长的burn-in和更多的样本")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("吉布斯抽样 (Gibbs Sampling)")
    print("=" * 70)
    
    # 运行所有示例
    demo_1_bivariate_normal()
    demo_2_mixture_model()
    demo_3_ising_model()
    demo_4_comparison_correlation()
    
    print("\n" + "=" * 70)
    print("所有示例运行完成!")
    print("=" * 70)
    print("\n总结:")
    print("1. 二元正态分布: 展示了吉布斯采样的基本原理和收敛性")
    print("2. 混合模型: 演示了数据增广技术在隐变量模型中的应用")
    print("3. 伊辛模型: 展示了吉布斯采样在物理模型中的应用")
    print("4. 相关性影响: 分析了变量相关性对采样效率的影响")
    print("\n关键洞察:")
    print("- 吉布斯采样在条件分布易于采样时非常高效")
    print("- 接受率为100%，无需调参")
    print("- 但在强相关情况下混合速度较慢")
    print("- 适合高维问题，特别是有条件独立结构的模型")
