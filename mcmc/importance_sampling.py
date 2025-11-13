"""
重要性抽样法 (Importance Sampling)
====================================

重要性抽样是一种蒙特卡洛方法，用于估计难以直接计算的期望值。
核心思想是从一个容易采样的提议分布中抽样，然后通过重要性权重
来修正样本，使得估计量无偏。

理论基础：
---------
目标：估计 E_p[f(X)] = ∫ f(x)p(x)dx

直接蒙特卡洛：
    E_p[f(X)] ≈ (1/N) Σ f(x_i), x_i ~ p(x)

重要性抽样：
    E_p[f(X)] = ∫ f(x)p(x)dx 
              = ∫ f(x)[p(x)/q(x)]q(x)dx
              = E_q[f(X)·w(X)]
    
    其中 w(x) = p(x)/q(x) 是重要性权重

估计量：
    E_p[f(X)] ≈ (1/N) Σ f(x_i)w(x_i), x_i ~ q(x)

方差：
    Var[估计量] ∝ ∫ f²(x)p²(x)/q(x)dx - [E_p[f(X)]]²

最优提议分布：
    q*(x) ∝ |f(x)|p(x)
    此时方差最小

应用场景：
---------
- 估计罕见事件概率
- 贝叶斯推断中的后验期望
- 强化学习中的策略评估
- 金融中的风险评估
- 物理模拟
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy import stats
from typing import Callable, Tuple, Dict

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ImportanceSampler:
    """
    重要性抽样器
    
    Parameters:
    -----------
    target_pdf : callable
        目标分布的概率密度函数 p(x)
    
    proposal_pdf : callable
        提议分布的概率密度函数 q(x)
    
    proposal_sampler : callable
        从提议分布采样的函数
    
    Attributes:
    -----------
    samples : ndarray
        采样的样本
    
    weights : ndarray
        重要性权重
    
    normalized_weights : ndarray
        归一化的权重
    """
    
    def __init__(self, target_pdf: Callable, proposal_pdf: Callable,
                 proposal_sampler: Callable):
        self.target_pdf = target_pdf
        self.proposal_pdf = proposal_pdf
        self.proposal_sampler = proposal_sampler
        
        self.samples = None
        self.weights = None
        self.normalized_weights = None
    
    def sample(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成样本和重要性权重
        
        Parameters:
        -----------
        n_samples : int
            样本数量
        
        Returns:
        --------
        samples : ndarray
            采样的样本
        
        weights : ndarray
            重要性权重
        """
        # 从提议分布采样
        samples = np.array([self.proposal_sampler() for _ in range(n_samples)])
        
        # 计算重要性权重 w(x) = p(x)/q(x)
        weights = np.array([
            self.target_pdf(x) / self.proposal_pdf(x) 
            if self.proposal_pdf(x) > 0 else 0
            for x in samples
        ])
        
        # 归一化权重（自归一化重要性抽样）
        normalized_weights = weights / np.sum(weights)
        
        self.samples = samples
        self.weights = weights
        self.normalized_weights = normalized_weights
        
        return samples, weights
    
    def estimate_expectation(self, func: Callable) -> Dict:
        """
        估计期望值 E_p[f(X)]
        
        Parameters:
        -----------
        func : callable
            要计算期望的函数 f(x)
        
        Returns:
        --------
        results : dict
            包含估计值、标准误差等信息
        """
        if self.samples is None:
            raise ValueError("请先调用sample()方法生成样本")
        
        # 计算函数值
        f_values = np.array([func(x) for x in self.samples])
        
        # 普通重要性抽样估计
        estimate = np.mean(f_values * self.weights)
        
        # 自归一化重要性抽样估计
        self_normalized_estimate = np.sum(f_values * self.normalized_weights)
        
        # 估计标准误差
        # Var[估计量] ≈ (1/N) * Var[f(X)w(X)]
        variance = np.var(f_values * self.weights)
        std_error = np.sqrt(variance / len(self.samples))
        
        # 有效样本量 (Effective Sample Size)
        # ESS = (Σw_i)² / Σw_i²
        ess = np.sum(self.weights)**2 / np.sum(self.weights**2)
        
        return {
            'estimate': estimate,
            'self_normalized_estimate': self_normalized_estimate,
            'std_error': std_error,
            'variance': variance,
            'ess': ess,
            'ess_ratio': ess / len(self.samples)
        }
    
    def get_weight_statistics(self) -> Dict:
        """获取权重统计信息"""
        if self.weights is None:
            raise ValueError("请先调用sample()方法生成样本")
        
        return {
            'mean_weight': np.mean(self.weights),
            'std_weight': np.std(self.weights),
            'min_weight': np.min(self.weights),
            'max_weight': np.max(self.weights),
            'cv': np.std(self.weights) / np.mean(self.weights),  # 变异系数
            'ess': np.sum(self.weights)**2 / np.sum(self.weights**2),
            'ess_ratio': (np.sum(self.weights)**2 / np.sum(self.weights**2)) / len(self.weights)
        }


def demo1_basic_example():
    """
    示例1: 基础示例 - 估计正态分布尾部概率
    目标：估计 P(X > 3) where X ~ N(0, 1)
    """
    print("=" * 70)
    print("示例1: 重要性抽样 - 估计正态分布尾部概率")
    print("=" * 70)
    
    # 目标分布: N(0, 1)
    target_dist = stats.norm(0, 1)
    
    def target_pdf(x):
        return target_dist.pdf(x)
    
    # 要估计的量: P(X > 3)
    threshold = 3.0
    
    def indicator_func(x):
        return 1.0 if x > threshold else 0.0
    
    # 真实值
    true_value = 1 - target_dist.cdf(threshold)
    print(f"\n目标：估计 P(X > {threshold}) where X ~ N(0, 1)")
    print(f"真实值: {true_value:.6f}")
    
    # 方法1: 直接蒙特卡洛（效率低）
    print("\n" + "-" * 50)
    print("方法1: 直接蒙特卡洛采样")
    n_samples = 10000
    samples_mc = target_dist.rvs(n_samples)
    estimate_mc = np.mean([indicator_func(x) for x in samples_mc])
    std_error_mc = np.sqrt(estimate_mc * (1 - estimate_mc) / n_samples)
    
    print(f"样本数: {n_samples}")
    print(f"估计值: {estimate_mc:.6f}")
    print(f"标准误差: {std_error_mc:.6f}")
    print(f"相对误差: {abs(estimate_mc - true_value) / true_value * 100:.2f}%")
    
    # 方法2: 重要性抽样（提议分布：N(4, 1)，在尾部采样）
    print("\n" + "-" * 50)
    print("方法2: 重要性抽样（提议分布：N(4, 1)）")
    
    proposal_dist = stats.norm(4, 1)
    
    def proposal_pdf(x):
        return proposal_dist.pdf(x)
    
    def proposal_sampler():
        return proposal_dist.rvs()
    
    sampler = ImportanceSampler(target_pdf, proposal_pdf, proposal_sampler)
    samples_is, weights = sampler.sample(n_samples)
    
    result = sampler.estimate_expectation(indicator_func)
    
    print(f"样本数: {n_samples}")
    print(f"估计值: {result['self_normalized_estimate']:.6f}")
    print(f"标准误差: {result['std_error']:.6f}")
    print(f"相对误差: {abs(result['self_normalized_estimate'] - true_value) / true_value * 100:.2f}%")
    print(f"有效样本量比例: {result['ess_ratio']:.4f}")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 目标分布与提议分布
    ax = axes[0, 0]
    x = np.linspace(-2, 8, 1000)
    ax.plot(x, target_dist.pdf(x), 'b-', linewidth=2, label='目标分布 N(0,1)')
    ax.plot(x, proposal_dist.pdf(x), 'r--', linewidth=2, label='提议分布 N(4,1)')
    ax.axvline(x=threshold, color='g', linestyle=':', linewidth=2, label=f'阈值 x={threshold}')
    ax.fill_between(x[x > threshold], 0, target_dist.pdf(x[x > threshold]), 
                     alpha=0.3, color='blue', label='目标区域')
    ax.set_xlabel('x')
    ax.set_ylabel('密度')
    ax.set_title('目标分布与提议分布')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 样本分布对比
    ax = axes[0, 1]
    ax.hist(samples_mc, bins=50, alpha=0.5, density=True, 
            color='blue', edgecolor='black', label='直接采样')
    ax.hist(samples_is, bins=50, alpha=0.5, density=True,
            color='red', edgecolor='black', label='重要性采样')
    ax.axvline(x=threshold, color='g', linestyle=':', linewidth=2)
    ax.set_xlabel('x')
    ax.set_ylabel('密度')
    ax.set_title('采样结果对比')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 重要性权重分布
    ax = axes[1, 0]
    ax.hist(weights, bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax.set_xlabel('权重 w(x)')
    ax.set_ylabel('频数')
    ax.set_title(f'重要性权重分布 (均值={np.mean(weights):.4f})')
    ax.grid(True, alpha=0.3)
    
    # 4. 估计值收敛
    ax = axes[1, 1]
    
    # 直接MC的累积估计
    cumsum_mc = np.cumsum([indicator_func(x) for x in samples_mc])
    counts_mc = np.arange(1, len(samples_mc) + 1)
    estimates_mc = cumsum_mc / counts_mc
    
    # 重要性采样的累积估计
    f_values_is = np.array([indicator_func(x) for x in samples_is])
    cumsum_is = np.cumsum(f_values_is * sampler.normalized_weights * len(samples_is))
    estimates_is = cumsum_is / counts_mc
    
    ax.plot(counts_mc, estimates_mc, 'b-', alpha=0.7, linewidth=2, label='直接MC')
    ax.plot(counts_mc, estimates_is, 'r-', alpha=0.7, linewidth=2, label='重要性采样')
    ax.axhline(y=true_value, color='g', linestyle='--', linewidth=2, label='真实值')
    ax.set_xlabel('样本数')
    ax.set_ylabel('累积估计值')
    ax.set_title('估计值收敛过程')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([100, n_samples])
    ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('mcmc/importance_sampling_tail.png', dpi=300, bbox_inches='tight')
    print("\n图像已保存: mcmc/importance_sampling_tail.png")
    plt.close()


def demo2_expectation_estimation():
    """
    示例2: 估计期望值
    目标：估计 E[X²] where X ~ Exp(1)
    """
    print("\n" + "=" * 70)
    print("示例2: 重要性抽样 - 估计期望值")
    print("=" * 70)
    
    # 目标分布: Exp(1)
    target_dist = stats.expon(scale=1)
    
    def target_pdf(x):
        return target_dist.pdf(x) if x >= 0 else 0
    
    # 要估计的量: E[X²]
    def func(x):
        return x**2
    
    # 真实值: E[X²] = Var(X) + E[X]² = 1 + 1 = 2
    true_value = 2.0
    print(f"\n目标：估计 E[X²] where X ~ Exp(1)")
    print(f"真实值: {true_value:.6f}")
    
    n_samples = 5000
    
    # 方法1: 直接蒙特卡洛
    print("\n" + "-" * 50)
    print("方法1: 直接蒙特卡洛")
    samples_mc = target_dist.rvs(n_samples)
    f_values_mc = np.array([func(x) for x in samples_mc])
    estimate_mc = np.mean(f_values_mc)
    std_error_mc = np.std(f_values_mc) / np.sqrt(n_samples)
    
    print(f"估计值: {estimate_mc:.6f}")
    print(f"标准误差: {std_error_mc:.6f}")
    print(f"相对误差: {abs(estimate_mc - true_value) / true_value * 100:.2f}%")
    
    # 方法2: 重要性抽样（提议分布：Exp(0.5)，重尾）
    print("\n" + "-" * 50)
    print("方法2: 重要性抽样（提议分布：Exp(0.5)）")
    
    proposal_dist = stats.expon(scale=2)  # rate=0.5
    
    def proposal_pdf(x):
        return proposal_dist.pdf(x) if x >= 0 else 0
    
    def proposal_sampler():
        return proposal_dist.rvs()
    
    sampler = ImportanceSampler(target_pdf, proposal_pdf, proposal_sampler)
    samples_is, weights = sampler.sample(n_samples)
    
    result = sampler.estimate_expectation(func)
    
    print(f"估计值: {result['self_normalized_estimate']:.6f}")
    print(f"标准误差: {result['std_error']:.6f}")
    print(f"相对误差: {abs(result['self_normalized_estimate'] - true_value) / true_value * 100:.2f}%")
    print(f"有效样本量比例: {result['ess_ratio']:.4f}")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 目标分布与提议分布
    ax = axes[0, 0]
    x = np.linspace(0, 8, 1000)
    ax.plot(x, target_dist.pdf(x), 'b-', linewidth=2, label='目标: Exp(1)')
    ax.plot(x, proposal_dist.pdf(x), 'r--', linewidth=2, label='提议: Exp(0.5)')
    ax.fill_between(x, 0, target_dist.pdf(x), alpha=0.2, color='blue')
    ax.set_xlabel('x')
    ax.set_ylabel('密度')
    ax.set_title('分布对比')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 函数f(x)=x²在两个分布下
    ax = axes[0, 1]
    ax.plot(x, func(x) * target_dist.pdf(x), 'b-', linewidth=2, 
            label='f(x)·p(x)')
    ax.plot(x, func(x) * proposal_dist.pdf(x), 'r--', linewidth=2,
            label='f(x)·q(x)')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)·密度')
    ax.set_title('被积函数对比')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 重要性权重与x的关系
    ax = axes[1, 0]
    weight_func = lambda x: target_pdf(x) / proposal_pdf(x) if x >= 0 else 0
    weights_plot = [weight_func(xi) for xi in x]
    ax.plot(x, weights_plot, 'g-', linewidth=2)
    ax.scatter(samples_is[:100], weights[:100], alpha=0.5, s=20, c='orange')
    ax.set_xlabel('x')
    ax.set_ylabel('权重 w(x) = p(x)/q(x)')
    ax.set_title('重要性权重函数')
    ax.grid(True, alpha=0.3)
    
    # 4. 估计值方差对比
    ax = axes[1, 1]
    
    # Bootstrap估计方差
    n_bootstrap = 100
    estimates_mc_boot = []
    estimates_is_boot = []
    
    for _ in range(n_bootstrap):
        # MC bootstrap
        idx_mc = np.random.choice(len(samples_mc), len(samples_mc), replace=True)
        estimates_mc_boot.append(np.mean(f_values_mc[idx_mc]))
        
        # IS bootstrap
        idx_is = np.random.choice(len(samples_is), len(samples_is), replace=True)
        f_is = np.array([func(samples_is[i]) for i in idx_is])
        w_is = weights[idx_is]
        w_is_norm = w_is / np.sum(w_is)
        estimates_is_boot.append(np.sum(f_is * w_is_norm))
    
    ax.hist(estimates_mc_boot, bins=30, alpha=0.5, color='blue', 
            edgecolor='black', label='直接MC')
    ax.hist(estimates_is_boot, bins=30, alpha=0.5, color='red',
            edgecolor='black', label='重要性采样')
    ax.axvline(x=true_value, color='g', linestyle='--', linewidth=2, label='真实值')
    ax.set_xlabel('估计值')
    ax.set_ylabel('频数')
    ax.set_title(f'Bootstrap方差对比 (n={n_bootstrap})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mcmc/importance_sampling_expectation.png', dpi=300, bbox_inches='tight')
    print("图像已保存: mcmc/importance_sampling_expectation.png")
    plt.close()


def demo3_proposal_comparison():
    """
    示例3: 不同提议分布的效果对比
    目标：估计 E[X³] where X ~ N(2, 1)
    """
    print("\n" + "=" * 70)
    print("示例3: 不同提议分布的效果对比")
    print("=" * 70)
    
    # 目标分布: N(2, 1)
    target_dist = stats.norm(2, 1)
    
    def target_pdf(x):
        return target_dist.pdf(x)
    
    # 要估计的量: E[X³]
    def func(x):
        return x**3
    
    # 真实值: E[X³] = E[X]³ + 3E[X]Var(X) = 8 + 6 = 14
    true_value = 14.0
    print(f"\n目标：估计 E[X³] where X ~ N(2, 1)")
    print(f"真实值: {true_value:.6f}")
    
    n_samples = 5000
    
    # 测试不同的提议分布
    proposals = [
        {'name': 'N(2, 1) [最优]', 'dist': stats.norm(2, 1)},
        {'name': 'N(2, 2)', 'dist': stats.norm(2, 2)},
        {'name': 'N(0, 1) [差]', 'dist': stats.norm(0, 1)},
        {'name': 'Uniform(-2, 6)', 'dist': stats.uniform(-2, 8)},
    ]
    
    results = []
    
    print("\n提议分布对比:")
    print("-" * 70)
    
    for prop in proposals:
        prop_dist = prop['dist']
        
        def proposal_pdf(x):
            return prop_dist.pdf(x)
        
        def proposal_sampler():
            return prop_dist.rvs()
        
        sampler = ImportanceSampler(target_pdf, proposal_pdf, proposal_sampler)
        samples, weights = sampler.sample(n_samples)
        
        result = sampler.estimate_expectation(func)
        weight_stats = sampler.get_weight_statistics()
        
        results.append({
            'name': prop['name'],
            'estimate': result['self_normalized_estimate'],
            'std_error': result['std_error'],
            'ess_ratio': result['ess_ratio'],
            'cv': weight_stats['cv'],
            'weights': weights
        })
        
        print(f"\n{prop['name']}:")
        print(f"  估计值: {result['self_normalized_estimate']:.6f}")
        print(f"  标准误差: {result['std_error']:.6f}")
        print(f"  相对误差: {abs(result['self_normalized_estimate'] - true_value) / true_value * 100:.2f}%")
        print(f"  ESS比例: {result['ess_ratio']:.4f}")
        print(f"  权重变异系数: {weight_stats['cv']:.4f}")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 所有分布对比
    ax = axes[0, 0]
    x = np.linspace(-3, 7, 1000)
    ax.plot(x, target_dist.pdf(x), 'b-', linewidth=3, label='目标: N(2,1)')
    
    colors = ['green', 'orange', 'red', 'purple']
    for i, prop in enumerate(proposals):
        ax.plot(x, prop['dist'].pdf(x), '--', linewidth=2, 
                color=colors[i], label=prop['name'])
    
    ax.set_xlabel('x')
    ax.set_ylabel('密度')
    ax.set_title('目标分布与提议分布对比')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 估计值对比
    ax = axes[0, 1]
    names = [r['name'] for r in results]
    estimates = [r['estimate'] for r in results]
    std_errors = [r['std_error'] for r in results]
    
    x_pos = np.arange(len(names))
    bars = ax.bar(x_pos, estimates, yerr=std_errors, capsize=5,
                   alpha=0.7, color=colors)
    ax.axhline(y=true_value, color='red', linestyle='--', 
               linewidth=2, label='真实值')
    ax.set_ylabel('估计值')
    ax.set_title('不同提议分布的估计结果')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. ESS比例对比
    ax = axes[1, 0]
    ess_ratios = [r['ess_ratio'] for r in results]
    bars = ax.bar(x_pos, ess_ratios, alpha=0.7, color=colors)
    ax.set_ylabel('ESS比例')
    ax.set_title('有效样本量比例（越高越好）')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # 在柱上标注数值
    for i, (bar, val) in enumerate(zip(bars, ess_ratios)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom')
    
    # 4. 权重变异系数对比
    ax = axes[1, 1]
    cvs = [r['cv'] for r in results]
    bars = ax.bar(x_pos, cvs, alpha=0.7, color=colors)
    ax.set_ylabel('变异系数 (CV)')
    ax.set_title('权重变异系数（越小越好）')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 在柱上标注数值
    for i, (bar, val) in enumerate(zip(bars, cvs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('mcmc/importance_sampling_comparison.png', dpi=300, bbox_inches='tight')
    print("\n图像已保存: mcmc/importance_sampling_comparison.png")
    plt.close()


def demo4_self_normalized_is():
    """
    示例4: 自归一化重要性抽样
    展示当归一化常数未知时如何使用自归一化IS
    """
    print("\n" + "=" * 70)
    print("示例4: 自归一化重要性抽样")
    print("=" * 70)
    
    # 目标分布（未归一化）: exp(-x²/2 - 0.3x⁴)
    def unnormalized_target(x):
        return np.exp(-x**2 / 2 - 0.3 * x**4)
    
    # 提议分布: N(0, 1)
    proposal_dist = stats.norm(0, 1)
    
    def proposal_pdf(x):
        return proposal_dist.pdf(x)
    
    def proposal_sampler():
        return proposal_dist.rvs()
    
    print("\n目标分布: 未归一化分布 p̃(x) = exp(-x²/2 - 0.3x⁴)")
    print("提议分布: N(0, 1)")
    
    # 采样
    n_samples = 10000
    samples = np.array([proposal_sampler() for _ in range(n_samples)])
    
    # 计算未归一化权重
    unnormalized_weights = np.array([
        unnormalized_target(x) / proposal_pdf(x) 
        for x in samples
    ])
    
    # 自归一化
    normalized_weights = unnormalized_weights / np.sum(unnormalized_weights)
    
    # 估计归一化常数
    Z_estimate = np.mean(unnormalized_weights)
    Z_std_error = np.std(unnormalized_weights) / np.sqrt(n_samples)
    
    print(f"\n样本数: {n_samples}")
    print(f"归一化常数估计: Z ≈ {Z_estimate:.6f} ± {Z_std_error:.6f}")
    
    # 估计期望值 E[X]
    x_values = samples
    E_x_estimate = np.sum(x_values * normalized_weights)
    print(f"E[X] ≈ {E_x_estimate:.6f}")
    
    # 估计方差 Var[X]
    E_x2_estimate = np.sum(x_values**2 * normalized_weights)
    var_x_estimate = E_x2_estimate - E_x_estimate**2
    print(f"Var[X] ≈ {var_x_estimate:.6f}")
    
    # 有效样本量
    ess = np.sum(unnormalized_weights)**2 / np.sum(unnormalized_weights**2)
    ess_ratio = ess / n_samples
    print(f"有效样本量比例: {ess_ratio:.4f}")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 未归一化目标分布与提议分布
    ax = axes[0, 0]
    x = np.linspace(-3, 3, 1000)
    unnorm_vals = [unnormalized_target(xi) for xi in x]
    
    ax.plot(x, unnorm_vals, 'b-', linewidth=2, label='未归一化目标 p̃(x)')
    ax.plot(x, proposal_dist.pdf(x) * max(unnorm_vals) / max(proposal_dist.pdf(x)),
            'r--', linewidth=2, label='提议分布（缩放）')
    ax.set_xlabel('x')
    ax.set_ylabel('密度（未归一化）')
    ax.set_title('未归一化目标分布')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 归一化后的分布与样本
    ax = axes[0, 1]
    # 归一化
    normalized_vals = np.array(unnorm_vals) / (Z_estimate * np.trapz(unnorm_vals, x) / len(x))
    
    ax.hist(samples, bins=60, density=True, alpha=0.6, weights=normalized_weights * len(samples),
            color='skyblue', edgecolor='black', label='加权样本')
    ax.plot(x, normalized_vals, 'r-', linewidth=2, label='估计的目标分布')
    ax.set_xlabel('x')
    ax.set_ylabel('密度')
    ax.set_title('自归一化结果')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 未归一化权重分布
    ax = axes[1, 0]
    ax.hist(unnormalized_weights, bins=50, alpha=0.7, 
            color='orange', edgecolor='black')
    ax.axvline(x=Z_estimate, color='r', linestyle='--', 
               linewidth=2, label=f'均值={Z_estimate:.4f}')
    ax.set_xlabel('未归一化权重')
    ax.set_ylabel('频数')
    ax.set_title('未归一化权重分布')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 归一化权重分布
    ax = axes[1, 1]
    ax.hist(normalized_weights, bins=50, alpha=0.7,
            color='green', edgecolor='black')
    ax.axvline(x=1/n_samples, color='r', linestyle='--',
               linewidth=2, label=f'均匀权重={1/n_samples:.6f}')
    ax.set_xlabel('归一化权重')
    ax.set_ylabel('频数')
    ax.set_title('归一化权重分布')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mcmc/importance_sampling_self_normalized.png', dpi=300, bbox_inches='tight')
    print("\n图像已保存: mcmc/importance_sampling_self_normalized.png")
    plt.close()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("重要性抽样法 (Importance Sampling) - 完整演示")
    print("=" * 70)
    
    # 运行所有示例
    demo1_basic_example()
    demo2_expectation_estimation()
    demo3_proposal_comparison()
    demo4_self_normalized_is()
    
    print("\n" + "=" * 70)
    print("所有演示完成！")
    print("=" * 70)
    print("\n生成的文件:")
    print("  1. mcmc/importance_sampling_tail.png - 尾部概率估计")
    print("  2. mcmc/importance_sampling_expectation.png - 期望值估计")
    print("  3. mcmc/importance_sampling_comparison.png - 提议分布对比")
    print("  4. mcmc/importance_sampling_self_normalized.png - 自归一化IS")
    print("\n" + "=" * 70)
