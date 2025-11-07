"""
隐马尔可夫模型 - 后向算法 (Backward Algorithm)

问题：概率计算问题
给定模型 λ=(A,B,π) 和观测序列 O=(o₁,o₂,...,o_T)，计算观测序列出现的概率 P(O|λ)

后向算法：通过后向概率的递推计算 P(O|λ)

后向概率定义：
β_t(i) = P(o_{t+1},o_{t+2},...,o_T|q_t=s_i,λ)
表示：在时刻t状态为s_i的条件下，从时刻t+1到T的观测序列为o_{t+1},...,o_T的概率

参考：《统计学习方法》(李航) 第10章 隐马尔可夫模型
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['Arial Unicode MS']
rcParams['axes.unicode_minus'] = False


class HMMBackward:
    """隐马尔可夫模型 - 后向算法"""
    
    def __init__(self, states, observations, A, B, pi):
        """
        初始化HMM模型
        
        参数:
            states: 状态列表
            observations: 观测列表
            A: 状态转移概率矩阵 (N×N)
            B: 观测概率矩阵 (N×M)
            pi: 初始状态概率向量 (N)
        """
        self.states = states
        self.observations = observations
        self.A = np.array(A)
        self.B = np.array(B)
        self.pi = np.array(pi)
        
        self.N = len(states)  # 状态数
        self.M = len(observations)  # 观测数
        
    def backward(self, obs_seq):
        """
        后向算法计算观测序列概率
        
        算法步骤：
        1. 初始化：β_T(i) = 1, i=1,2,...,N
        2. 递推：β_t(i) = Σⱼ aᵢⱼ·bⱼ(o_{t+1})·β_{t+1}(j)
        3. 终止：P(O|λ) = Σᵢ πᵢ·bᵢ(o₁)·β₁(i)
        
        参数:
            obs_seq: 观测序列（观测值列表，如['红','白','红']）
            
        返回:
            prob: 观测序列概率 P(O|λ)
            beta: 后向概率矩阵 (T×N)
        """
        T = len(obs_seq)
        
        # 将观测序列转换为索引
        obs_indices = [self.observations.index(o) for o in obs_seq]
        
        # 初始化后向概率矩阵
        beta = np.zeros((T, self.N))
        
        print("=" * 70)
        print("后向算法 - 观测序列概率计算")
        print("=" * 70)
        print(f"\n观测序列: {obs_seq}")
        print(f"观测序列长度 T = {T}")
        print()
        
        # 步骤1: 初始化 (t=T)
        print("【步骤1：初始化】")
        print(f"设置 β_T(i) = 1, i=1,2,...,{self.N}")
        print()
        
        for i in range(self.N):
            beta[T-1, i] = 1.0
            print(f"  β_{T}({self.states[i]}) = 1.0")
        
        print(f"\nβ_{T} = {beta[T-1]}")
        print()
        
        # 步骤2: 递推 (t=T-1, T-2, ..., 1)
        print("【步骤2：递推（从后向前）】")
        print("计算 β_t(i) = Σⱼ aᵢⱼ · bⱼ(o_{t+1}) · β_{t+1}(j)")
        print()
        
        for t in range(T-2, -1, -1):
            print(f"--- 时刻 t={t+1} ---")
            print(f"观测 o_{t+2} = {obs_seq[t+1]}")
            print()
            
            for i in range(self.N):
                # 计算 Σⱼ aᵢⱼ·bⱼ(o_{t+1})·β_{t+1}(j)
                sum_val = 0.0
                details = []
                for j in range(self.N):
                    contrib = self.A[i, j] * self.B[j, obs_indices[t+1]] * beta[t+1, j]
                    sum_val += contrib
                    details.append(f"{self.A[i,j]:.2f}×{self.B[j,obs_indices[t+1]]:.2f}×{beta[t+1,j]:.6f}")
                
                beta[t, i] = sum_val
                
                print(f"  β_{t+1}({self.states[i]}):")
                print(f"    = {' + '.join(details)}")
                print(f"    = {sum_val:.6f}")
            
            print(f"\nβ_{t+1} = {beta[t]}")
            print()
        
        # 步骤3: 终止
        print("【步骤3：终止】")
        print(f"P(O|λ) = Σᵢ πᵢ · bᵢ(o₁) · β₁(i)")
        print(f"观测 o₁ = {obs_seq[0]}")
        print()
        
        prob = 0.0
        details = []
        for i in range(self.N):
            contrib = self.pi[i] * self.B[i, obs_indices[0]] * beta[0, i]
            prob += contrib
            print(f"  π({self.states[i]}) × b_{self.states[i]}({obs_seq[0]}) × β₁({self.states[i]})")
            print(f"  = {self.pi[i]:.4f} × {self.B[i, obs_indices[0]]:.4f} × {beta[0, i]:.6f}")
            print(f"  = {contrib:.6f}")
            details.append(f"{contrib:.6f}")
        
        print(f"\nP(O|λ) = {' + '.join(details)}")
        print(f"       = {prob:.6f}")
        print()
        
        print("=" * 70)
        print(f"【结果】观测序列 {obs_seq} 的概率为: {prob:.6f}")
        print("=" * 70)
        print()
        
        return prob, beta
    
    def visualize_backward(self, obs_seq, beta):
        """可视化后向概率"""
        T = len(obs_seq)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 绘制后向概率热力图
        im = ax1.imshow(beta.T, aspect='auto', cmap='Blues', interpolation='nearest')
        ax1.set_xlabel('时刻 t', fontsize=12)
        ax1.set_ylabel('状态', fontsize=12)
        ax1.set_title('后向概率 β_t(i) 热力图', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(T))
        ax1.set_xticklabels([f't={i+1}\n{obs_seq[i]}' for i in range(T)])
        ax1.set_yticks(range(self.N))
        ax1.set_yticklabels(self.states)
        
        # 添加数值标注
        for i in range(self.N):
            for t in range(T):
                text = ax1.text(t, i, f'{beta[t, i]:.4f}',
                               ha="center", va="center", color="black", fontsize=9)
        
        plt.colorbar(im, ax=ax1, label='概率值')
        
        # 绘制每个状态的后向概率曲线
        for i in range(self.N):
            ax2.plot(range(1, T+1), beta[:, i], marker='s', label=self.states[i], linewidth=2)
        
        ax2.set_xlabel('时刻 t', fontsize=12)
        ax2.set_ylabel('后向概率 β_t(i)', fontsize=12)
        ax2.set_title('各状态后向概率变化曲线', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(range(1, T+1))
        
        # 在x轴上标注观测
        for t in range(T):
            ax2.text(t+1, -0.05, obs_seq[t], ha='center', va='top', 
                    transform=ax2.get_xaxis_transform(), fontsize=9, color='blue')
        
        plt.tight_layout()
        plt.savefig('hmm_backward.png', dpi=300, bbox_inches='tight')
        print("后向算法可视化图已保存为: hmm_backward.png\n")
        plt.close()


class HMMForwardBackward:
    """结合前向和后向算法"""
    
    def __init__(self, states, observations, A, B, pi):
        self.states = states
        self.observations = observations
        self.A = np.array(A)
        self.B = np.array(B)
        self.pi = np.array(pi)
        self.N = len(states)
        self.M = len(observations)
    
    def forward(self, obs_seq):
        """前向算法（简化版，不打印详细过程）"""
        T = len(obs_seq)
        obs_indices = [self.observations.index(o) for o in obs_seq]
        alpha = np.zeros((T, self.N))
        
        # 初始化
        for i in range(self.N):
            alpha[0, i] = self.pi[i] * self.B[i, obs_indices[0]]
        
        # 递推
        for t in range(1, T):
            for i in range(self.N):
                alpha[t, i] = np.sum(alpha[t-1] * self.A[:, i]) * self.B[i, obs_indices[t]]
        
        # 终止
        prob = np.sum(alpha[T-1])
        return prob, alpha
    
    def backward(self, obs_seq):
        """后向算法（简化版，不打印详细过程）"""
        T = len(obs_seq)
        obs_indices = [self.observations.index(o) for o in obs_seq]
        beta = np.zeros((T, self.N))
        
        # 初始化
        beta[T-1, :] = 1.0
        
        # 递推
        for t in range(T-2, -1, -1):
            for i in range(self.N):
                beta[t, i] = np.sum(self.A[i, :] * self.B[:, obs_indices[t+1]] * beta[t+1, :])
        
        # 终止
        prob = np.sum(self.pi * self.B[:, obs_indices[0]] * beta[0, :])
        return prob, beta
    
    def compare_algorithms(self, obs_seq):
        """比较前向和后向算法的结果"""
        print("=" * 70)
        print("前向算法 vs 后向算法 - 结果对比")
        print("=" * 70)
        print(f"\n观测序列: {obs_seq}\n")
        
        prob_forward, alpha = self.forward(obs_seq)
        prob_backward, beta = self.backward(obs_seq)
        
        print(f"前向算法计算结果: P(O|λ) = {prob_forward:.8f}")
        print(f"后向算法计算结果: P(O|λ) = {prob_backward:.8f}")
        print(f"绝对误差: {abs(prob_forward - prob_backward):.2e}")
        print()
        
        # 可视化对比
        self.visualize_comparison(obs_seq, alpha, beta)
        
        return prob_forward, prob_backward, alpha, beta
    
    def visualize_comparison(self, obs_seq, alpha, beta):
        """可视化前向和后向概率对比"""
        T = len(obs_seq)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 前向概率热力图
        im1 = axes[0, 0].imshow(alpha.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        axes[0, 0].set_xlabel('时刻 t', fontsize=11)
        axes[0, 0].set_ylabel('状态', fontsize=11)
        axes[0, 0].set_title('前向概率 α_t(i)', fontsize=12, fontweight='bold')
        axes[0, 0].set_xticks(range(T))
        axes[0, 0].set_xticklabels([f't={i+1}\n{obs_seq[i]}' for i in range(T)])
        axes[0, 0].set_yticks(range(self.N))
        axes[0, 0].set_yticklabels(self.states)
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 后向概率热力图
        im2 = axes[0, 1].imshow(beta.T, aspect='auto', cmap='Blues', interpolation='nearest')
        axes[0, 1].set_xlabel('时刻 t', fontsize=11)
        axes[0, 1].set_ylabel('状态', fontsize=11)
        axes[0, 1].set_title('后向概率 β_t(i)', fontsize=12, fontweight='bold')
        axes[0, 1].set_xticks(range(T))
        axes[0, 1].set_xticklabels([f't={i+1}\n{obs_seq[i]}' for i in range(T)])
        axes[0, 1].set_yticks(range(self.N))
        axes[0, 1].set_yticklabels(self.states)
        plt.colorbar(im2, ax=axes[0, 1])
        
        # 前向概率曲线
        for i in range(self.N):
            axes[1, 0].plot(range(1, T+1), alpha[:, i], marker='o', label=self.states[i], linewidth=2)
        axes[1, 0].set_xlabel('时刻 t', fontsize=11)
        axes[1, 0].set_ylabel('前向概率', fontsize=11)
        axes[1, 0].set_title('前向概率变化曲线', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 后向概率曲线
        for i in range(self.N):
            axes[1, 1].plot(range(1, T+1), beta[:, i], marker='s', label=self.states[i], linewidth=2)
        axes[1, 1].set_xlabel('时刻 t', fontsize=11)
        axes[1, 1].set_ylabel('后向概率', fontsize=11)
        axes[1, 1].set_title('后向概率变化曲线', fontsize=12, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('hmm_forward_backward_comparison.png', dpi=300, bbox_inches='tight')
        print("前向-后向算法对比图已保存为: hmm_forward_backward_comparison.png\n")
        plt.close()


def main():
    """主函数 - 李航书例10.2"""
    
    print("\n" + "="*70)
    print("示例：盒子抽球模型（李航《统计学习方法》例10.2）")
    print("="*70)
    print()
    
    # 定义模型参数
    states = ['盒子1', '盒子2', '盒子3']
    observations = ['红', '白']
    
    # 初始状态概率
    pi = [0.2, 0.4, 0.4]
    
    # 状态转移概率矩阵
    A = [
        [0.5, 0.2, 0.3],  # 盒子1转移概率
        [0.3, 0.5, 0.2],  # 盒子2转移概率
        [0.2, 0.3, 0.5]   # 盒子3转移概率
    ]
    
    # 观测概率矩阵（发射概率）
    B = [
        [0.5, 0.5],  # 盒子1: P(红)=0.5, P(白)=0.5
        [0.4, 0.6],  # 盒子2: P(红)=0.4, P(白)=0.6
        [0.7, 0.3]   # 盒子3: P(红)=0.7, P(白)=0.3
    ]
    
    # 打印模型参数
    print("【模型参数】")
    print(f"状态集合: {states}")
    print(f"观测集合: {observations}")
    print()
    
    print("初始状态概率 π:")
    for i, state in enumerate(states):
        print(f"  π({state}) = {pi[i]}")
    print()
    
    print("状态转移概率矩阵 A:")
    print("  从 \\ 到 ", end="")
    for state in states:
        print(f"{state:>8}", end="")
    print()
    for i, state_from in enumerate(states):
        print(f"  {state_from:>8}", end="")
        for j in range(len(states)):
            print(f"{A[i][j]:>8.2f}", end="")
        print()
    print()
    
    print("观测概率矩阵 B:")
    print("  状态 \\ 观测 ", end="")
    for obs in observations:
        print(f"{obs:>8}", end="")
    print()
    for i, state in enumerate(states):
        print(f"  {state:>11}", end="")
        for j in range(len(observations)):
            print(f"{B[i][j]:>8.2f}", end="")
        print()
    print()
    
    # 观测序列
    obs_seq = ['红', '白', '红']
    
    # 创建HMM后向算法模型
    hmm_backward = HMMBackward(states, observations, A, B, pi)
    
    # 执行后向算法
    prob, beta = hmm_backward.backward(obs_seq)
    
    # 可视化
    hmm_backward.visualize_backward(obs_seq, beta)
    
    # 对比前向和后向算法
    print("\n" + "="*70)
    print("【算法对比】")
    print("="*70)
    print()
    
    hmm_both = HMMForwardBackward(states, observations, A, B, pi)
    hmm_both.compare_algorithms(obs_seq)
    
    # 测试多个序列
    print("\n" + "="*70)
    print("【多序列测试】")
    print("="*70)
    print()
    
    test_sequences = [
        ['红', '红', '红'],
        ['白', '白', '白'],
        ['红', '白', '白']
    ]
    
    print(f"{'观测序列':<20} {'前向算法':<15} {'后向算法':<15} {'误差':<15}")
    print("-" * 65)
    
    for seq in test_sequences:
        prob_f, _ = hmm_both.forward(seq)
        prob_b, _ = hmm_both.backward(seq)
        error = abs(prob_f - prob_b)
        print(f"{str(seq):<20} {prob_f:.8f}   {prob_b:.8f}   {error:.2e}")
    
    print()


if __name__ == "__main__":
    main()
