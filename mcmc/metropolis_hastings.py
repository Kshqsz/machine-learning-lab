"""
Metropolis-Hastings 算法实现
==============================

Metropolis-Hastings算法是马尔可夫链蒙特卡洛（MCMC）方法的核心算法，
用于从复杂的概率分布中采样。

算法原理：
----------
1. 构造一个马尔可夫链，其平稳分布为目标分布π(x)
2. 从提议分布q(x'|x)中采样候选点x'
3. 计算接受概率α = min(1, π(x')q(x|x') / (π(x)q(x'|x)))
4. 以概率α接受x'，否则保持在x

算法特点：
----------
- 只需要知道目标分布的未归一化形式p̃(x)
- 提议分布可以是任意的（对称或非对称）
- 保证收敛到目标分布（在一定条件下）
- 需要burn-in期和thinning来减少自相关

常见提议分布：
--------------
1. 对称随机游走：q(x'|x) = q(x|x'), α = min(1, π(x')/π(x))
2. 独立采样器：q(x'|x) = q(x'), α = min(1, π(x')q(x)/(π(x)q(x')))
3. 自适应步长：根据接受率调整提议分布的方差

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Callable, Tuple, Optional, Dict, Any
import warnings

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class MetropolisHastings:
    """
    Metropolis-Hastings采样器
    
    支持多种提议分布和自适应步长调整
    """
    
    def __init__(
        self,
        target_log_prob: Callable[[np.ndarray], float],
        proposal_type: str = 'symmetric',
        proposal_scale: float = 1.0,
        dim: int = 1
    ):
        """
        初始化Metropolis-Hastings采样器
        
        参数:
        ----
        target_log_prob : callable
            目标分布的对数概率函数（可以是未归一化的）
        proposal_type : str
            提议分布类型：'symmetric' (随机游走), 'independent'
        proposal_scale : float
            提议分布的尺度参数（标准差）
        dim : int
            采样空间的维度
        """
        self.target_log_prob = target_log_prob
        self.proposal_type = proposal_type
        self.proposal_scale = proposal_scale
        self.dim = dim
        
        # 统计信息
        self.samples = None
        self.n_accepted = 0
        self.n_total = 0
        self.acceptance_history = []
        
    def propose(
        self,
        current: np.ndarray,
        **kwargs
    ) -> Tuple[np.ndarray, float]:
        """
        从提议分布中采样
        
        参数:
        ----
        current : ndarray
            当前状态
        
        返回:
        ----
        proposal : ndarray
            提议的新状态
        log_proposal_ratio : float
            对数提议比率 log(q(x|x') / q(x'|x))
        """
        if self.proposal_type == 'symmetric':
            # 对称随机游走：q(x'|x) = N(x, σ²I)
            proposal = current + np.random.normal(0, self.proposal_scale, size=self.dim)
            log_proposal_ratio = 0.0  # 对称分布，比率为1
            
        elif self.proposal_type == 'independent':
            # 独立采样器：q(x'|x) = q(x')，这里使用标准正态分布
            proposal = np.random.normal(0, self.proposal_scale, size=self.dim)
            # log q(x|x') - log q(x'|x)
            log_qx_given_xprime = -0.5 * np.sum((current / self.proposal_scale) ** 2)
            log_qxprime_given_x = -0.5 * np.sum((proposal / self.proposal_scale) ** 2)
            log_proposal_ratio = log_qx_given_xprime - log_qxprime_given_x
            
        else:
            raise ValueError(f"未知的提议类型: {self.proposal_type}")
        
        return proposal, log_proposal_ratio
    
    def acceptance_probability(
        self,
        current: np.ndarray,
        proposal: np.ndarray,
        log_proposal_ratio: float
    ) -> float:
        """
        计算接受概率
        
        α = min(1, π(x') q(x|x') / (π(x) q(x'|x)))
          = min(1, exp(log π(x') - log π(x) + log q(x|x') - log q(x'|x)))
        """
        log_target_current = self.target_log_prob(current)
        log_target_proposal = self.target_log_prob(proposal)
        
        # 对数接受概率
        log_alpha = log_target_proposal - log_target_current + log_proposal_ratio
        
        # 返回接受概率
        return min(1.0, np.exp(log_alpha))
    
    def sample(
        self,
        n_samples: int,
        initial_state: Optional[np.ndarray] = None,
        burn_in: int = 1000,
        thin: int = 1,
        adaptive: bool = False,
        target_acceptance: float = 0.234,
        verbose: bool = True
    ) -> np.ndarray:
        """
        使用Metropolis-Hastings算法采样
        
        参数:
        ----
        n_samples : int
            需要的样本数量（burn-in后）
        initial_state : ndarray, optional
            初始状态，如果为None则随机初始化
        burn_in : int
            burn-in期长度
        thin : int
            thinning间隔（每thin步保存一个样本）
        adaptive : bool
            是否使用自适应步长
        target_acceptance : float
            目标接受率（用于自适应步长）
        verbose : bool
            是否打印进度信息
        
        返回:
        ----
        samples : ndarray, shape (n_samples, dim)
            采样得到的样本
        """
        # 初始化
        if initial_state is None:
            current = np.random.randn(self.dim)
        else:
            current = initial_state.copy()
        
        # 总迭代次数
        total_iterations = burn_in + n_samples * thin
        
        # 存储样本
        samples = np.zeros((n_samples, self.dim))
        
        # 重置统计
        self.n_accepted = 0
        self.n_total = 0
        self.acceptance_history = []
        
        # 自适应参数
        adaptation_interval = 100 if adaptive else total_iterations + 1
        
        if verbose:
            print(f"开始采样：总迭代 {total_iterations}, burn-in {burn_in}, thin {thin}")
            print(f"提议类型: {self.proposal_type}, 初始尺度: {self.proposal_scale:.4f}")
        
        sample_idx = 0
        for i in range(total_iterations):
            # 提议新状态
            proposal, log_proposal_ratio = self.propose(current)
            
            # 计算接受概率
            alpha = self.acceptance_probability(current, proposal, log_proposal_ratio)
            
            # 接受或拒绝
            if np.random.rand() < alpha:
                current = proposal
                self.n_accepted += 1
            
            self.n_total += 1
            
            # 保存样本（在burn-in后且满足thinning条件）
            if i >= burn_in and (i - burn_in) % thin == 0:
                samples[sample_idx] = current
                sample_idx += 1
            
            # 自适应步长调整
            if adaptive and (i + 1) % adaptation_interval == 0:
                current_acceptance = self.n_accepted / self.n_total
                self.acceptance_history.append(current_acceptance)
                
                # 调整步长（使用Robbins-Monro算法）
                if current_acceptance > target_acceptance:
                    self.proposal_scale *= 1.1  # 增大步长
                else:
                    self.proposal_scale *= 0.9  # 减小步长
                
                if verbose and i < burn_in:
                    print(f"  迭代 {i+1}: 接受率 {current_acceptance:.4f}, "
                          f"新尺度 {self.proposal_scale:.4f}")
                
                # 重置计数器
                self.n_accepted = 0
                self.n_total = 0
        
        # 最终接受率
        final_acceptance = self.n_accepted / self.n_total if self.n_total > 0 else 0
        
        if verbose:
            print(f"\n采样完成！")
            print(f"最终接受率: {final_acceptance:.4f}")
            print(f"最终提议尺度: {self.proposal_scale:.4f}")
        
        self.samples = samples
        return samples
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取采样统计信息
        """
        if self.samples is None:
            return {}
        
        # 计算自相关
        autocorr = self._compute_autocorrelation(self.samples, max_lag=50)
        
        # 计算有效样本量
        ess = self._compute_ess(autocorr)
        
        return {
            'mean': np.mean(self.samples, axis=0),
            'std': np.std(self.samples, axis=0),
            'autocorrelation': autocorr,
            'ess': ess,
            'ess_per_sample': ess / len(self.samples)
        }
    
    def _compute_autocorrelation(
        self,
        samples: np.ndarray,
        max_lag: int = 50
    ) -> np.ndarray:
        """
        计算自相关函数
        """
        n = len(samples)
        max_lag = min(max_lag, n - 1)
        
        if self.dim == 1:
            samples = samples.flatten()
            mean = np.mean(samples)
            var = np.var(samples)
            
            autocorr = np.zeros(max_lag + 1)
            for lag in range(max_lag + 1):
                autocorr[lag] = np.mean((samples[:-lag or None] - mean) * 
                                       (samples[lag:] - mean)) / var
        else:
            # 多维情况，计算第一维的自相关
            samples_1d = samples[:, 0]
            mean = np.mean(samples_1d)
            var = np.var(samples_1d)
            
            autocorr = np.zeros(max_lag + 1)
            for lag in range(max_lag + 1):
                autocorr[lag] = np.mean((samples_1d[:-lag or None] - mean) * 
                                       (samples_1d[lag:] - mean)) / var
        
        return autocorr
    
    def _compute_ess(self, autocorr: np.ndarray) -> float:
        """
        计算有效样本量（Effective Sample Size）
        
        ESS = N / (1 + 2·Σρ(k))
        """
        n = len(self.samples)
        
        # 找到第一个负自相关的位置
        cutoff = 1
        for i in range(1, len(autocorr)):
            if autocorr[i] < 0:
                break
            cutoff = i + 1
        
        # 计算ESS
        ess = n / (1 + 2 * np.sum(autocorr[1:cutoff]))
        
        return ess


def demo1_standard_normal():
    """
    示例1: 从标准正态分布采样
    
    这是最简单的例子，用于验证算法的正确性
    """
    print("=" * 70)
    print("示例1: Metropolis-Hastings - 标准正态分布")
    print("=" * 70)
    
    # 目标分布：N(0, 1)
    def log_prob(x):
        return -0.5 * np.sum(x ** 2)
    
    # 创建采样器（对称随机游走）
    sampler = MetropolisHastings(
        target_log_prob=log_prob,
        proposal_type='symmetric',
        proposal_scale=2.5,
        dim=1
    )
    
    # 采样
    print("\n使用对称随机游走（提议尺度=2.5）")
    samples = sampler.sample(
        n_samples=10000,
        initial_state=np.array([5.0]),  # 从远离目标的位置开始
        burn_in=1000,
        thin=1,
        verbose=True
    )
    
    # 获取统计信息
    sample_stats = sampler.get_statistics()
    print(f"\n样本统计:")
    print(f"  均值: {sample_stats['mean'][0]:.4f} (真实值: 0.0000)")
    print(f"  标准差: {sample_stats['std'][0]:.4f} (真实值: 1.0000)")
    print(f"  有效样本量: {sample_stats['ess']:.0f} ({sample_stats['ess_per_sample']:.2%})")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 轨迹图
    ax = axes[0, 0]
    ax.plot(samples[:500], alpha=0.7, linewidth=1)
    ax.axhline(y=0, color='r', linestyle='--', label='真实均值')
    ax.set_xlabel('迭代次数')
    ax.set_ylabel('样本值')
    ax.set_title('马尔可夫链轨迹（前500个样本）')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 直方图与真实分布
    ax = axes[0, 1]
    ax.hist(samples.flatten(), bins=50, density=True, alpha=0.7, 
            edgecolor='black', label='MCMC样本')
    x = np.linspace(-4, 4, 100)
    ax.plot(x, stats.norm.pdf(x, 0, 1), 'r-', linewidth=2, label='真实分布 N(0,1)')
    ax.set_xlabel('x')
    ax.set_ylabel('概率密度')
    ax.set_title('样本分布 vs 目标分布')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 自相关函数
    ax = axes[1, 0]
    lags = np.arange(len(sample_stats['autocorrelation']))
    ax.bar(lags, sample_stats['autocorrelation'], alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('滞后 (Lag)')
    ax.set_ylabel('自相关系数')
    ax.set_title('自相关函数（显示样本之间的相关性）')
    ax.grid(True, alpha=0.3)
    
    # 4. Q-Q图
    ax = axes[1, 1]
    stats.probplot(samples.flatten(), dist="norm", plot=ax)
    ax.set_title('Q-Q图（检验正态性）')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mcmc/metropolis_hastings_normal.png', dpi=300, bbox_inches='tight')
    print("\n图像已保存: mcmc/metropolis_hastings_normal.png")


def demo2_bimodal_distribution():
    """
    示例2: 双峰分布
    
    测试算法在多峰分布上的表现
    """
    print("\n" + "=" * 70)
    print("示例2: Metropolis-Hastings - 双峰分布")
    print("=" * 70)
    
    # 目标分布：0.3·N(-3, 1) + 0.7·N(3, 1)
    def log_prob(x):
        x_val = x[0] if len(x.shape) > 0 else x
        # 使用log-sum-exp技巧避免数值下溢
        log_p1 = np.log(0.3) - 0.5 * (x_val + 3) ** 2
        log_p2 = np.log(0.7) - 0.5 * (x_val - 3) ** 2
        return np.logaddexp(log_p1, log_p2)
    
    # 测试不同的提议尺度
    scales = [0.5, 2.0, 5.0]
    results = []
    
    for scale in scales:
        print(f"\n提议尺度 = {scale}")
        print("-" * 50)
        
        sampler = MetropolisHastings(
            target_log_prob=log_prob,
            proposal_type='symmetric',
            proposal_scale=scale,
            dim=1
        )
        
        samples = sampler.sample(
            n_samples=10000,
            initial_state=np.array([-3.0]),
            burn_in=2000,
            thin=1,
            verbose=False
        )
        
        sample_stats = sampler.get_statistics()
        acceptance_rate = sampler.n_accepted / sampler.n_total
        
        print(f"  接受率: {acceptance_rate:.4f}")
        print(f"  样本均值: {sample_stats['mean'][0]:.4f}")
        print(f"  有效样本量比例: {sample_stats['ess_per_sample']:.4f}")
        
        results.append({
            'scale': scale,
            'samples': samples,
            'stats': sample_stats,
            'acceptance_rate': acceptance_rate
        })
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 真实分布
    x = np.linspace(-8, 8, 1000)
    true_density = 0.3 * stats.norm.pdf(x, -3, 1) + 0.7 * stats.norm.pdf(x, 3, 1)
    
    for idx, result in enumerate(results):
        # 轨迹图
        ax = axes[0, idx]
        trace = result['samples'][:1000].flatten()
        ax.plot(trace, alpha=0.7, linewidth=0.8)
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('样本值')
        ax.set_title(f'轨迹图（尺度={result["scale"]}）\n接受率={result["acceptance_rate"]:.3f}')
        ax.grid(True, alpha=0.3)
        
        # 直方图
        ax = axes[1, idx]
        ax.hist(result['samples'].flatten(), bins=50, density=True, 
                alpha=0.7, edgecolor='black', label='MCMC样本')
        ax.plot(x, true_density, 'r-', linewidth=2, label='真实分布')
        ax.set_xlabel('x')
        ax.set_ylabel('概率密度')
        ax.set_title(f'分布（ESS={result["stats"]["ess_per_sample"]:.2%}）')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mcmc/metropolis_hastings_bimodal.png', dpi=300, bbox_inches='tight')
    print("\n图像已保存: mcmc/metropolis_hastings_bimodal.png")


def demo3_adaptive_metropolis():
    """
    示例3: 自适应Metropolis-Hastings
    
    演示自适应步长调整的效果
    """
    print("\n" + "=" * 70)
    print("示例3: 自适应Metropolis-Hastings")
    print("=" * 70)
    
    # 目标分布：N(0, 1)
    def log_prob(x):
        return -0.5 * np.sum(x ** 2)
    
    # 比较固定步长和自适应步长
    print("\n方法1: 固定步长（初始=0.1，不自适应）")
    print("-" * 50)
    sampler_fixed = MetropolisHastings(
        target_log_prob=log_prob,
        proposal_type='symmetric',
        proposal_scale=0.1,
        dim=1
    )
    
    samples_fixed = sampler_fixed.sample(
        n_samples=5000,
        burn_in=2000,
        thin=1,
        adaptive=False,
        verbose=False
    )
    
    sample_stats_fixed = sampler_fixed.get_statistics()
    print(f"  最终步长: {sampler_fixed.proposal_scale:.4f}")
    print(f"  接受率: {sampler_fixed.n_accepted / sampler_fixed.n_total:.4f}")
    print(f"  有效样本量比例: {sample_stats_fixed['ess_per_sample']:.4f}")
    
    print("\n方法2: 自适应步长（初始=0.1，目标接受率=0.234）")
    print("-" * 50)
    sampler_adaptive = MetropolisHastings(
        target_log_prob=log_prob,
        proposal_type='symmetric',
        proposal_scale=0.1,
        dim=1
    )
    
    samples_adaptive = sampler_adaptive.sample(
        n_samples=5000,
        burn_in=2000,
        thin=1,
        adaptive=True,
        target_acceptance=0.234,
        verbose=True
    )
    
    sample_stats_adaptive = sampler_adaptive.get_statistics()
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 自相关比较
    ax = axes[0, 0]
    ax.plot(sample_stats_fixed['autocorrelation'], label='固定步长', linewidth=2)
    ax.plot(sample_stats_adaptive['autocorrelation'], label='自适应步长', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('滞后 (Lag)')
    ax.set_ylabel('自相关系数')
    ax.set_title('自相关函数比较')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 轨迹比较
    ax = axes[0, 1]
    ax.plot(samples_fixed[:500], alpha=0.7, label='固定步长')
    ax.plot(samples_adaptive[:500], alpha=0.7, label='自适应步长')
    ax.set_xlabel('迭代次数')
    ax.set_ylabel('样本值')
    ax.set_title('马尔可夫链轨迹比较（前500个样本）')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 分布比较
    ax = axes[1, 0]
    x = np.linspace(-4, 4, 100)
    ax.hist(samples_fixed.flatten(), bins=40, density=True, alpha=0.5, 
            label='固定步长', edgecolor='black')
    ax.hist(samples_adaptive.flatten(), bins=40, density=True, alpha=0.5, 
            label='自适应步长', edgecolor='black')
    ax.plot(x, stats.norm.pdf(x, 0, 1), 'r-', linewidth=2, label='真实分布')
    ax.set_xlabel('x')
    ax.set_ylabel('概率密度')
    ax.set_title('样本分布比较')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 效率比较
    ax = axes[1, 1]
    methods = ['固定步长', '自适应步长']
    ess_values = [sample_stats_fixed['ess'], sample_stats_adaptive['ess']]
    colors = ['skyblue', 'lightcoral']
    bars = ax.bar(methods, ess_values, color=colors, edgecolor='black', alpha=0.7)
    ax.set_ylabel('有效样本量')
    ax.set_title('采样效率比较')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上添加数值
    for bar, ess in zip(bars, ess_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{ess:.0f}\n({ess/5000:.1%})',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('mcmc/metropolis_hastings_adaptive.png', dpi=300, bbox_inches='tight')
    print("\n图像已保存: mcmc/metropolis_hastings_adaptive.png")


def demo4_2d_distribution():
    """
    示例4: 二维分布采样
    
    演示Metropolis-Hastings在多维空间的应用
    """
    print("\n" + "=" * 70)
    print("示例4: Metropolis-Hastings - 二维相关正态分布")
    print("=" * 70)
    
    # 目标分布：二维正态分布，具有相关性
    # 协方差矩阵
    rho = 0.8
    cov = np.array([[1.0, rho], [rho, 1.0]])
    cov_inv = np.linalg.inv(cov)
    
    def log_prob(x):
        """二维正态分布的对数概率"""
        return -0.5 * x @ cov_inv @ x
    
    print(f"\n目标分布：N([0, 0], Σ)")
    print(f"协方差矩阵:")
    print(f"  [[1.0, {rho}],")
    print(f"   [{rho}, 1.0]]")
    
    # 创建采样器
    sampler = MetropolisHastings(
        target_log_prob=log_prob,
        proposal_type='symmetric',
        proposal_scale=2.0,
        dim=2
    )
    
    # 采样
    samples = sampler.sample(
        n_samples=10000,
        initial_state=np.array([3.0, -3.0]),
        burn_in=2000,
        thin=2,
        verbose=True
    )
    
    sample_stats = sampler.get_statistics()
    print(f"\n样本统计:")
    print(f"  均值: [{sample_stats['mean'][0]:.4f}, {sample_stats['mean'][1]:.4f}]")
    print(f"  标准差: [{sample_stats['std'][0]:.4f}, {sample_stats['std'][1]:.4f}]")
    print(f"  样本相关系数: {np.corrcoef(samples.T)[0, 1]:.4f} (真实值: {rho})")
    
    # 可视化
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. 2D散点图和等高线
    ax = fig.add_subplot(gs[0:2, 0:2])
    
    # 绘制样本点
    ax.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=10, label='MCMC样本')
    
    # 绘制真实分布的等高线
    x1 = np.linspace(-4, 4, 100)
    x2 = np.linspace(-4, 4, 100)
    X1, X2 = np.meshgrid(x1, x2)
    pos = np.dstack((X1, X2))
    
    # 计算真实分布的概率密度
    rv = stats.multivariate_normal([0, 0], cov)
    Z = rv.pdf(pos)
    
    contours = ax.contour(X1, X2, Z, levels=8, colors='red', alpha=0.6, linewidths=2)
    ax.clabel(contours, inline=True, fontsize=8)
    
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title('二维MCMC样本与真实分布等高线')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # 2. x1的边缘分布
    ax = fig.add_subplot(gs[0:2, 2])
    ax.hist(samples[:, 0], bins=40, density=True, alpha=0.7, 
            orientation='horizontal', edgecolor='black', label='MCMC样本')
    x = np.linspace(-4, 4, 100)
    ax.plot(stats.norm.pdf(x, 0, 1), x, 'r-', linewidth=2, label='真实分布')
    ax.set_ylabel('x₁')
    ax.set_xlabel('概率密度')
    ax.set_title('x₁边缘分布')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. x2的边缘分布
    ax = fig.add_subplot(gs[2, 0:2])
    ax.hist(samples[:, 1], bins=40, density=True, alpha=0.7, 
            edgecolor='black', label='MCMC样本')
    x = np.linspace(-4, 4, 100)
    ax.plot(x, stats.norm.pdf(x, 0, 1), 'r-', linewidth=2, label='真实分布')
    ax.set_xlabel('x₂')
    ax.set_ylabel('概率密度')
    ax.set_title('x₂边缘分布')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 轨迹图
    ax = fig.add_subplot(gs[2, 2])
    n_trace = min(500, len(samples))
    ax.plot(samples[:n_trace, 0], alpha=0.7, label='x₁', linewidth=1)
    ax.plot(samples[:n_trace, 1], alpha=0.7, label='x₂', linewidth=1)
    ax.set_xlabel('迭代次数')
    ax.set_ylabel('样本值')
    ax.set_title(f'马尔可夫链轨迹（前{n_trace}个）')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig('mcmc/metropolis_hastings_2d.png', dpi=300, bbox_inches='tight')
    print("\n图像已保存: mcmc/metropolis_hastings_2d.png")


def demo5_banana_distribution():
    """
    示例5: 香蕉形分布（Rosenbrock分布）
    
    测试算法在具有强相关和非凸分布上的表现
    """
    print("\n" + "=" * 70)
    print("示例5: Metropolis-Hastings - 香蕉形分布")
    print("=" * 70)
    
    # 香蕉形分布（Rosenbrock分布的变形）
    B = 0.03  # 控制"弯曲"程度的参数
    
    def log_prob(x):
        """
        x = [x1, x2]
        p(x) ∝ exp(-x1²/200 - 0.5(x2 + Bx1² - 100B)²)
        """
        x1, x2 = x[0], x[1]
        log_p = -x1**2 / 200 - 0.5 * (x2 + B * x1**2 - 100 * B)**2
        return log_p
    
    print(f"\n香蕉形分布参数: B = {B}")
    print("该分布具有强非线性相关性和香蕉状轮廓")
    
    # 创建采样器
    sampler = MetropolisHastings(
        target_log_prob=log_prob,
        proposal_type='symmetric',
        proposal_scale=3.0,
        dim=2
    )
    
    # 采样
    samples = sampler.sample(
        n_samples=15000,
        initial_state=np.array([0.0, 0.0]),
        burn_in=5000,
        thin=3,
        verbose=True
    )
    
    sample_stats = sampler.get_statistics()
    print(f"\n样本统计:")
    print(f"  均值: [{sample_stats['mean'][0]:.4f}, {sample_stats['mean'][1]:.4f}]")
    print(f"  标准差: [{sample_stats['std'][0]:.4f}, {sample_stats['std'][1]:.4f}]")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. 2D热力图和样本轨迹
    ax = axes[0, 0]
    
    # 创建网格
    x1_range = np.linspace(-30, 30, 200)
    x2_range = np.linspace(-1, 5, 200)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    
    # 计算对数概率
    Z = np.zeros_like(X1)
    for i in range(len(x1_range)):
        for j in range(len(x2_range)):
            Z[j, i] = log_prob(np.array([X1[j, i], X2[j, i]]))
    
    # 绘制热力图
    im = ax.contourf(X1, X2, np.exp(Z), levels=20, cmap='YlOrRd', alpha=0.8)
    plt.colorbar(im, ax=ax, label='概率密度')
    
    # 绘制样本轨迹（前1000个）
    n_show = min(1000, len(samples))
    ax.plot(samples[:n_show, 0], samples[:n_show, 1], 
            'b-', alpha=0.3, linewidth=0.5, label='MCMC轨迹')
    ax.scatter(samples[0, 0], samples[0, 1], c='green', s=100, 
              marker='o', edgecolors='black', linewidths=2, label='起点', zorder=5)
    ax.scatter(samples[n_show-1, 0], samples[n_show-1, 1], c='red', s=100,
              marker='s', edgecolors='black', linewidths=2, label='终点', zorder=5)
    
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title('香蕉形分布与MCMC轨迹')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 样本散点图
    ax = axes[0, 1]
    ax.hexbin(samples[:, 0], samples[:, 1], gridsize=50, cmap='Blues', mincnt=1)
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title('MCMC样本密度图（Hexbin）')
    ax.grid(True, alpha=0.3)
    
    # 3. x1边缘分布
    ax = axes[1, 0]
    ax.hist(samples[:, 0], bins=50, density=True, alpha=0.7, 
            edgecolor='black', color='skyblue')
    ax.set_xlabel('x₁')
    ax.set_ylabel('概率密度')
    ax.set_title('x₁的边缘分布')
    ax.grid(True, alpha=0.3)
    
    # 4. x2边缘分布
    ax = axes[1, 1]
    ax.hist(samples[:, 1], bins=50, density=True, alpha=0.7, 
            edgecolor='black', color='lightcoral')
    ax.set_xlabel('x₂')
    ax.set_ylabel('概率密度')
    ax.set_title('x₂的边缘分布')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mcmc/metropolis_hastings_banana.png', dpi=300, bbox_inches='tight')
    print("\n图像已保存: mcmc/metropolis_hastings_banana.png")


if __name__ == "__main__":
    print("=" * 70)
    print("Metropolis-Hastings 算法 - 完整演示")
    print("=" * 70)
    
    # 运行所有示例
    demo1_standard_normal()
    demo2_bimodal_distribution()
    demo3_adaptive_metropolis()
    demo4_2d_distribution()
    demo5_banana_distribution()
    
    print("\n" + "=" * 70)
    print("所有演示完成！")
    print("=" * 70)
    print("\n生成的文件:")
    print("  1. mcmc/metropolis_hastings_normal.png - 标准正态分布")
    print("  2. mcmc/metropolis_hastings_bimodal.png - 双峰分布")
    print("  3. mcmc/metropolis_hastings_adaptive.png - 自适应步长")
    print("  4. mcmc/metropolis_hastings_2d.png - 二维相关分布")
    print("  5. mcmc/metropolis_hastings_banana.png - 香蕉形分布")
    print("=" * 70)
