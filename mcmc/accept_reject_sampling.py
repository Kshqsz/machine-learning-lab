"""
接受-拒绝采样法 (Accept-Reject Sampling / Rejection Sampling)
=====================================================

接受-拒绝采样是一种从复杂分布中抽样的蒙特卡洛方法。
当目标分布难以直接采样时，可以使用一个容易采样的提议分布，
通过接受-拒绝机制来生成目标分布的样本。

理论基础：
---------
目标分布: p(x) - 难以直接采样
提议分布: q(x) - 容易采样
常数M: 满足 M·q(x) ≥ p(x) 对所有x成立

算法步骤：
1. 从提议分布q(x)采样: x ~ q(x)
2. 从均匀分布采样: u ~ U(0,1)
3. 如果 u ≤ p(x)/(M·q(x))，接受x；否则拒绝并重复

接受率: 1/M
- M越小，接受率越高，效率越高
- 需要选择合适的q(x)和M来提高效率

应用场景：
---------
- 从非标准分布采样
- 贝叶斯推断
- 随机模拟
- 计算复杂积分
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy import stats
from typing import Callable, Tuple, Optional

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class AcceptRejectSampler:
    """
    接受-拒绝采样器
    
    Parameters:
    -----------
    target_pdf : callable
        目标分布的概率密度函数 p(x)
    
    proposal_pdf : callable
        提议分布的概率密度函数 q(x)
    
    proposal_sampler : callable
        从提议分布采样的函数
    
    M : float
        常数，满足 M·q(x) ≥ p(x)
    
    Attributes:
    -----------
    n_accepted : int
        接受的样本数
    
    n_rejected : int
        拒绝的样本数
    
    acceptance_rate : float
        接受率
    """
    
    def __init__(self, target_pdf: Callable, proposal_pdf: Callable,
                 proposal_sampler: Callable, M: float):
        self.target_pdf = target_pdf
        self.proposal_pdf = proposal_pdf
        self.proposal_sampler = proposal_sampler
        self.M = M
        
        self.n_accepted = 0
        self.n_rejected = 0
    
    def sample(self, n_samples: int) -> Tuple[np.ndarray, float]:
        """
        生成n_samples个样本
        
        Parameters:
        -----------
        n_samples : int
            需要的样本数量
        
        Returns:
        --------
        samples : ndarray
            接受的样本
        
        acceptance_rate : float
            实际接受率
        """
        samples = []
        n_accepted_current = 0
        n_rejected_current = 0
        
        while n_accepted_current < n_samples:
            # 1. 从提议分布采样
            x = self.proposal_sampler()
            
            # 2. 计算接受概率
            p_x = self.target_pdf(x)
            q_x = self.proposal_pdf(x)
            
            # 避免除零
            if q_x == 0:
                n_rejected_current += 1
                continue
            
            acceptance_prob = p_x / (self.M * q_x)
            
            # 3. 接受-拒绝判断
            u = np.random.uniform(0, 1)
            
            if u <= acceptance_prob:
                samples.append(x)
                n_accepted_current += 1
            else:
                n_rejected_current += 1
        
        self.n_accepted += n_accepted_current
        self.n_rejected += n_rejected_current
        
        acceptance_rate = n_accepted_current / (n_accepted_current + n_rejected_current)
        
        return np.array(samples), acceptance_rate
    
    def get_statistics(self) -> dict:
        """获取采样统计信息"""
        total = self.n_accepted + self.n_rejected
        acceptance_rate = self.n_accepted / total if total > 0 else 0
        
        return {
            'n_accepted': self.n_accepted,
            'n_rejected': self.n_rejected,
            'total_samples': total,
            'acceptance_rate': acceptance_rate,
            'theoretical_rate': 1 / self.M
        }


def demo1_basic_example():
    """
    示例1: 基础示例 - Beta分布采样
    目标分布: Beta(2, 5)
    提议分布: Uniform(0, 1)
    """
    print("=" * 70)
    print("示例1: 接受-拒绝采样 - Beta分布")
    print("=" * 70)
    
    # 定义目标分布 Beta(2, 5)
    alpha, beta = 2, 5
    target_dist = stats.beta(alpha, beta)
    
    def target_pdf(x):
        if 0 <= x <= 1:
            return target_dist.pdf(x)
        return 0
    
    # 定义提议分布 U(0, 1)
    def proposal_pdf(x):
        return 1.0 if 0 <= x <= 1 else 0
    
    def proposal_sampler():
        return np.random.uniform(0, 1)
    
    # 找到M: max(p(x)/q(x))
    # Beta(2,5)在x=0.2处最大
    x_test = np.linspace(0, 1, 1000)
    p_test = target_dist.pdf(x_test)
    M = np.max(p_test)
    
    print(f"\n目标分布: Beta({alpha}, {beta})")
    print(f"提议分布: Uniform(0, 1)")
    print(f"常数M: {M:.4f}")
    print(f"理论接受率: {1/M:.4f}")
    
    # 采样
    sampler = AcceptRejectSampler(target_pdf, proposal_pdf, proposal_sampler, M)
    samples, acceptance_rate = sampler.sample(5000)
    
    print(f"\n生成样本数: {len(samples)}")
    print(f"实际接受率: {acceptance_rate:.4f}")
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. 目标分布 vs 提议分布
    ax = axes[0]
    x = np.linspace(0, 1, 1000)
    ax.plot(x, target_dist.pdf(x), 'b-', linewidth=2, label='目标分布 p(x)')
    ax.plot(x, M * np.ones_like(x), 'r--', linewidth=2, label=f'M·q(x) = {M:.2f}')
    ax.fill_between(x, 0, target_dist.pdf(x), alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('密度')
    ax.set_title('接受-拒绝采样原理')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 采样结果直方图
    ax = axes[1]
    ax.hist(samples, bins=50, density=True, alpha=0.6, 
            color='skyblue', edgecolor='black', label='采样结果')
    ax.plot(x, target_dist.pdf(x), 'r-', linewidth=2, label='真实分布')
    ax.set_xlabel('x')
    ax.set_ylabel('密度')
    ax.set_title(f'采样结果 (n={len(samples)})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Q-Q图（检验分布一致性）
    ax = axes[2]
    theoretical_quantiles = np.sort(target_dist.rvs(len(samples)))
    sample_quantiles = np.sort(samples)
    ax.scatter(theoretical_quantiles, sample_quantiles, alpha=0.5, s=10)
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='理想情况')
    ax.set_xlabel('理论分位数')
    ax.set_ylabel('样本分位数')
    ax.set_title('Q-Q图（分布一致性检验）')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mcmc/accept_reject_beta.png', dpi=300, bbox_inches='tight')
    print("\n图像已保存: mcmc/accept_reject_beta.png")
    plt.close()


def demo2_normal_distribution():
    """
    示例2: 从标准正态分布采样
    目标分布: N(0, 1)
    提议分布: Cauchy(0, 1)
    """
    print("\n" + "=" * 70)
    print("示例2: 接受-拒绝采样 - 标准正态分布")
    print("=" * 70)
    
    # 目标分布: 标准正态
    target_dist = stats.norm(0, 1)
    
    def target_pdf(x):
        return target_dist.pdf(x)
    
    # 提议分布: 标准Cauchy
    proposal_dist = stats.cauchy(0, 1)
    
    def proposal_pdf(x):
        return proposal_dist.pdf(x)
    
    def proposal_sampler():
        return proposal_dist.rvs()
    
    # 计算M
    # 对于N(0,1)和Cauchy(0,1)，M ≈ sqrt(2π·e) ≈ 4.1
    M = np.sqrt(2 * np.pi * np.e)
    
    print(f"\n目标分布: N(0, 1)")
    print(f"提议分布: Cauchy(0, 1)")
    print(f"常数M: {M:.4f}")
    print(f"理论接受率: {1/M:.4f}")
    
    # 采样
    sampler = AcceptRejectSampler(target_pdf, proposal_pdf, proposal_sampler, M)
    samples, acceptance_rate = sampler.sample(10000)
    
    print(f"\n生成样本数: {len(samples)}")
    print(f"实际接受率: {acceptance_rate:.4f}")
    
    # 统计量比较
    print(f"\n样本统计量:")
    print(f"  均值: {np.mean(samples):.4f} (理论值: 0)")
    print(f"  标准差: {np.std(samples):.4f} (理论值: 1)")
    print(f"  偏度: {stats.skew(samples):.4f} (理论值: 0)")
    print(f"  峰度: {stats.kurtosis(samples):.4f} (理论值: 0)")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 目标分布与提议分布
    ax = axes[0, 0]
    x = np.linspace(-5, 5, 1000)
    ax.plot(x, target_dist.pdf(x), 'b-', linewidth=2, label='目标: N(0,1)')
    ax.plot(x, proposal_dist.pdf(x), 'g--', linewidth=2, label='提议: Cauchy(0,1)')
    ax.plot(x, M * proposal_dist.pdf(x), 'r:', linewidth=2, label=f'M·q(x)')
    ax.fill_between(x, 0, target_dist.pdf(x), alpha=0.2)
    ax.set_xlabel('x')
    ax.set_ylabel('密度')
    ax.set_title('分布对比')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 0.5])
    
    # 2. 采样结果
    ax = axes[0, 1]
    ax.hist(samples, bins=60, density=True, alpha=0.6,
            color='skyblue', edgecolor='black', label='采样结果')
    ax.plot(x, target_dist.pdf(x), 'r-', linewidth=2, label='真实分布')
    ax.set_xlabel('x')
    ax.set_ylabel('密度')
    ax.set_title(f'采样结果直方图 (n={len(samples)})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-4, 4])
    
    # 3. 累积分布函数对比
    ax = axes[1, 0]
    sorted_samples = np.sort(samples)
    empirical_cdf = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)
    ax.plot(sorted_samples, empirical_cdf, 'b-', linewidth=2, label='经验CDF')
    ax.plot(x, target_dist.cdf(x), 'r--', linewidth=2, label='理论CDF')
    ax.set_xlabel('x')
    ax.set_ylabel('累积概率')
    ax.set_title('累积分布函数对比')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 接受-拒绝过程可视化
    ax = axes[1, 1]
    # 生成一些样本用于可视化
    n_vis = 200
    x_prop = [proposal_sampler() for _ in range(n_vis)]
    u_samples = np.random.uniform(0, 1, n_vis)
    
    accepted = []
    rejected = []
    
    for x, u in zip(x_prop, u_samples):
        p_x = target_pdf(x)
        q_x = proposal_pdf(x)
        if u <= p_x / (M * q_x):
            accepted.append(x)
        else:
            rejected.append(x)
    
    ax.hist(accepted, bins=30, alpha=0.5, color='green', label=f'接受 (n={len(accepted)})')
    ax.hist(rejected, bins=30, alpha=0.5, color='red', label=f'拒绝 (n={len(rejected)})')
    ax.plot(x, target_dist.pdf(x) * n_vis * 0.3, 'b-', linewidth=2, label='目标分布（缩放）')
    ax.set_xlabel('x')
    ax.set_ylabel('频数')
    ax.set_title('接受-拒绝过程')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mcmc/accept_reject_normal.png', dpi=300, bbox_inches='tight')
    print("图像已保存: mcmc/accept_reject_normal.png")
    plt.close()


def demo3_mixture_distribution():
    """
    示例3: 混合高斯分布采样
    目标分布: 0.3·N(-2, 0.5²) + 0.7·N(2, 1²)
    提议分布: N(0, 3²)
    """
    print("\n" + "=" * 70)
    print("示例3: 接受-拒绝采样 - 混合高斯分布")
    print("=" * 70)
    
    # 目标分布: 混合高斯
    def target_pdf(x):
        return (0.3 * stats.norm(-2, 0.5).pdf(x) + 
                0.7 * stats.norm(2, 1).pdf(x))
    
    # 提议分布: N(0, 3)
    proposal_dist = stats.norm(0, 3)
    
    def proposal_pdf(x):
        return proposal_dist.pdf(x)
    
    def proposal_sampler():
        return proposal_dist.rvs()
    
    # 数值计算M
    x_test = np.linspace(-5, 6, 2000)
    ratio = target_pdf(x_test) / proposal_pdf(x_test)
    M = np.max(ratio) * 1.1  # 稍微放大以确保覆盖
    
    print(f"\n目标分布: 0.3·N(-2, 0.5²) + 0.7·N(2, 1²)")
    print(f"提议分布: N(0, 3²)")
    print(f"常数M: {M:.4f}")
    print(f"理论接受率: {1/M:.4f}")
    
    # 采样
    sampler = AcceptRejectSampler(target_pdf, proposal_pdf, proposal_sampler, M)
    samples, acceptance_rate = sampler.sample(8000)
    
    print(f"\n生成样本数: {len(samples)}")
    print(f"实际接受率: {acceptance_rate:.4f}")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 目标分布与提议分布
    ax = axes[0, 0]
    x = np.linspace(-5, 6, 1000)
    p_x = target_pdf(x)
    q_x = proposal_pdf(x)
    
    ax.plot(x, p_x, 'b-', linewidth=2, label='目标分布 p(x)')
    ax.plot(x, q_x, 'g--', linewidth=2, label='提议分布 q(x)')
    ax.plot(x, M * q_x, 'r:', linewidth=2, label=f'M·q(x) (M={M:.2f})')
    ax.fill_between(x, 0, p_x, alpha=0.2, color='blue')
    
    # 绘制两个高斯成分
    ax.plot(x, 0.3 * stats.norm(-2, 0.5).pdf(x), 'c--', alpha=0.5, label='成分1')
    ax.plot(x, 0.7 * stats.norm(2, 1).pdf(x), 'm--', alpha=0.5, label='成分2')
    
    ax.set_xlabel('x')
    ax.set_ylabel('密度')
    ax.set_title('混合高斯分布与提议分布')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 采样结果
    ax = axes[0, 1]
    ax.hist(samples, bins=60, density=True, alpha=0.6,
            color='skyblue', edgecolor='black', label='采样结果')
    ax.plot(x, p_x, 'r-', linewidth=2, label='真实分布')
    ax.set_xlabel('x')
    ax.set_ylabel('密度')
    ax.set_title(f'采样结果 (n={len(samples)})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 接受率随M的变化
    ax = axes[1, 0]
    M_values = np.linspace(M * 0.5, M * 2, 50)
    acceptance_rates = 1 / M_values
    ax.plot(M_values, acceptance_rates, 'b-', linewidth=2)
    ax.axvline(x=M, color='r', linestyle='--', label=f'当前M={M:.2f}')
    ax.axhline(y=1/M, color='r', linestyle=':', alpha=0.5)
    ax.set_xlabel('M')
    ax.set_ylabel('接受率')
    ax.set_title('接受率与M的关系')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 样本散点图（时序）
    ax = axes[1, 1]
    n_show = min(500, len(samples))
    ax.scatter(range(n_show), samples[:n_show], alpha=0.5, s=10)
    ax.axhline(y=-2, color='c', linestyle='--', alpha=0.5, label='成分1中心')
    ax.axhline(y=2, color='m', linestyle='--', alpha=0.5, label='成分2中心')
    ax.set_xlabel('样本序号')
    ax.set_ylabel('样本值')
    ax.set_title('样本序列（前500个）')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mcmc/accept_reject_mixture.png', dpi=300, bbox_inches='tight')
    print("图像已保存: mcmc/accept_reject_mixture.png")
    plt.close()


def demo4_efficiency_comparison():
    """
    示例4: 不同M值的效率对比
    展示M的选择对接受率和效率的影响
    """
    print("\n" + "=" * 70)
    print("示例4: 不同M值的效率对比")
    print("=" * 70)
    
    # 目标分布: Beta(2, 5)
    alpha, beta = 2, 5
    target_dist = stats.beta(alpha, beta)
    
    def target_pdf(x):
        if 0 <= x <= 1:
            return target_dist.pdf(x)
        return 0
    
    def proposal_pdf(x):
        return 1.0 if 0 <= x <= 1 else 0
    
    def proposal_sampler():
        return np.random.uniform(0, 1)
    
    # 最优M
    x_test = np.linspace(0, 1, 1000)
    M_optimal = np.max(target_dist.pdf(x_test))
    
    # 测试不同的M值
    M_values = [M_optimal, M_optimal * 1.5, M_optimal * 2, M_optimal * 3]
    results = []
    
    print("\nM值对比:")
    print("-" * 50)
    
    for M in M_values:
        sampler = AcceptRejectSampler(target_pdf, proposal_pdf, proposal_sampler, M)
        
        # 记录时间
        import time
        start_time = time.time()
        samples, acceptance_rate = sampler.sample(5000)
        elapsed_time = time.time() - start_time
        
        stats_dict = sampler.get_statistics()
        
        results.append({
            'M': M,
            'acceptance_rate': acceptance_rate,
            'total_samples': stats_dict['total_samples'],
            'time': elapsed_time
        })
        
        print(f"M = {M:.2f}:")
        print(f"  接受率: {acceptance_rate:.4f} (理论: {1/M:.4f})")
        print(f"  总采样次数: {stats_dict['total_samples']}")
        print(f"  耗时: {elapsed_time:.4f}秒")
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. M值与接受率
    ax = axes[0]
    M_array = [r['M'] for r in results]
    acc_rates = [r['acceptance_rate'] for r in results]
    theoretical_rates = [1/M for M in M_array]
    
    x_pos = np.arange(len(M_array))
    width = 0.35
    
    ax.bar(x_pos - width/2, acc_rates, width, label='实际接受率', alpha=0.8)
    ax.bar(x_pos + width/2, theoretical_rates, width, label='理论接受率', alpha=0.8)
    ax.set_xlabel('M值')
    ax.set_ylabel('接受率')
    ax.set_title('M值与接受率的关系')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{M:.2f}' for M in M_array])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. M值与总采样次数
    ax = axes[1]
    total_samples = [r['total_samples'] for r in results]
    ax.bar(range(len(M_array)), total_samples, alpha=0.8, color='orange')
    ax.set_xlabel('M值')
    ax.set_ylabel('总采样次数')
    ax.set_title('M值与采样效率')
    ax.set_xticks(range(len(M_array)))
    ax.set_xticklabels([f'{M:.2f}' for M in M_array])
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. M值与计算时间
    ax = axes[2]
    times = [r['time'] for r in results]
    ax.bar(range(len(M_array)), times, alpha=0.8, color='green')
    ax.set_xlabel('M值')
    ax.set_ylabel('计算时间（秒）')
    ax.set_title('M值与计算时间')
    ax.set_xticks(range(len(M_array)))
    ax.set_xticklabels([f'{M:.2f}' for M in M_array])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('mcmc/accept_reject_efficiency.png', dpi=300, bbox_inches='tight')
    print("\n图像已保存: mcmc/accept_reject_efficiency.png")
    plt.close()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("接受-拒绝采样法 (Accept-Reject Sampling) - 完整演示")
    print("=" * 70)
    
    # 运行所有示例
    demo1_basic_example()
    demo2_normal_distribution()
    demo3_mixture_distribution()
    demo4_efficiency_comparison()
    
    print("\n" + "=" * 70)
    print("所有演示完成！")
    print("=" * 70)
    print("\n生成的文件:")
    print("  1. mcmc/accept_reject_beta.png - Beta分布采样")
    print("  2. mcmc/accept_reject_normal.png - 标准正态分布采样")
    print("  3. mcmc/accept_reject_mixture.png - 混合高斯分布采样")
    print("  4. mcmc/accept_reject_efficiency.png - 效率对比分析")
    print("\n" + "=" * 70)
