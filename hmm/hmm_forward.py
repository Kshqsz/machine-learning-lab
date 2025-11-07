"""
隐马尔可夫模型 - 前向算法 (Forward Algorithm)

问题：概率计算问题
给定模型 λ=(A,B,π) 和观测序列 O=(o₁,o₂,...,o_T)，计算观测序列出现的概率 P(O|λ)

前向算法：通过前向概率的递推计算 P(O|λ)

前向概率定义：
α_t(i) = P(o₁,o₂,...,o_t, q_t=s_i|λ)
表示：到时刻t为止的观测序列为o₁,o₂,...,o_t且时刻t处于状态s_i的概率

参考：《统计学习方法》(李航) 第10章 隐马尔可夫模型
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['Arial Unicode MS']
rcParams['axes.unicode_minus'] = False


class HMMForward:
    """隐马尔可夫模型 - 前向算法"""
    
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
        
    def forward(self, obs_seq):
        """
        前向算法计算观测序列概率
        
        算法步骤：
        1. 初始化：α₁(i) = πᵢ·bᵢ(o₁), i=1,2,...,N
        2. 递推：α_{t+1}(i) = [Σⱼ α_t(j)·aⱼᵢ]·bᵢ(o_{t+1})
        3. 终止：P(O|λ) = Σᵢ α_T(i)
        
        参数:
            obs_seq: 观测序列（观测值列表，如['红','白','红']）
            
        返回:
            prob: 观测序列概率 P(O|λ)
            alpha: 前向概率矩阵 (T×N)
        """
        T = len(obs_seq)
        
        # 将观测序列转换为索引
        obs_indices = [self.observations.index(o) for o in obs_seq]
        
        # 初始化前向概率矩阵
        alpha = np.zeros((T, self.N))
        
        print("=" * 70)
        print("前向算法 - 观测序列概率计算")
        print("=" * 70)
        print(f"\n观测序列: {obs_seq}")
        print(f"观测序列长度 T = {T}")
        print()
        
        # 步骤1: 初始化 (t=1)
        print("【步骤1：初始化】")
        print(f"计算 α₁(i) = πᵢ · bᵢ(o₁), i=1,2,...,{self.N}")
        print(f"观测 o₁ = {obs_seq[0]}")
        print()
        
        for i in range(self.N):
            alpha[0, i] = self.pi[i] * self.B[i, obs_indices[0]]
            print(f"  α₁({self.states[i]}) = π({self.states[i]}) × b_{self.states[i]}({obs_seq[0]})")
            print(f"                = {self.pi[i]:.4f} × {self.B[i, obs_indices[0]]:.4f}")
            print(f"                = {alpha[0, i]:.6f}")
        
        print(f"\nα₁ = {alpha[0]}")
        print()
        
        # 步骤2: 递推 (t=2,3,...,T)
        print("【步骤2：递推】")
        print("计算 α_{t+1}(i) = [Σⱼ α_t(j)·aⱼᵢ] · bᵢ(o_{t+1})")
        print()
        
        for t in range(1, T):
            print(f"--- 时刻 t={t+1} ---")
            print(f"观测 o_{t+1} = {obs_seq[t]}")
            print()
            
            for i in range(self.N):
                # 计算 Σⱼ α_t(j)·aⱼᵢ
                sum_val = 0.0
                details = []
                for j in range(self.N):
                    contrib = alpha[t-1, j] * self.A[j, i]
                    sum_val += contrib
                    details.append(f"{alpha[t-1, j]:.6f}×{self.A[j, i]:.2f}")
                
                # 乘以观测概率
                alpha[t, i] = sum_val * self.B[i, obs_indices[t]]
                
                print(f"  α_{t+1}({self.states[i]}):")
                print(f"    Σⱼ α_{t}(j)·a_j,{self.states[i]} = {' + '.join(details)}")
                print(f"                          = {sum_val:.6f}")
                print(f"    α_{t+1}({self.states[i]}) = {sum_val:.6f} × {self.B[i, obs_indices[t]]:.4f}")
                print(f"                    = {alpha[t, i]:.6f}")
            
            print(f"\nα_{t+1} = {alpha[t]}")
            print()
        
        # 步骤3: 终止
        print("【步骤3：终止】")
        print(f"P(O|λ) = Σᵢ α_T(i)")
        
        prob = np.sum(alpha[T-1])
        
        details = []
        for i in range(self.N):
            details.append(f"α_T({self.states[i]})={alpha[T-1, i]:.6f}")
        
        print(f"       = {' + '.join(details)}")
        print(f"       = {prob:.6f}")
        print()
        
        print("=" * 70)
        print(f"【结果】观测序列 {obs_seq} 的概率为: {prob:.6f}")
        print("=" * 70)
        print()
        
        return prob, alpha
    
    def visualize_forward(self, obs_seq, alpha):
        """可视化前向概率"""
        T = len(obs_seq)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 绘制前向概率热力图
        im = ax1.imshow(alpha.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax1.set_xlabel('时刻 t', fontsize=12)
        ax1.set_ylabel('状态', fontsize=12)
        ax1.set_title('前向概率 α_t(i) 热力图', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(T))
        ax1.set_xticklabels([f't={i+1}\n{obs_seq[i]}' for i in range(T)])
        ax1.set_yticks(range(self.N))
        ax1.set_yticklabels(self.states)
        
        # 添加数值标注
        for i in range(self.N):
            for t in range(T):
                text = ax1.text(t, i, f'{alpha[t, i]:.4f}',
                               ha="center", va="center", color="black", fontsize=9)
        
        plt.colorbar(im, ax=ax1, label='概率值')
        
        # 绘制每个状态的前向概率曲线
        for i in range(self.N):
            ax2.plot(range(1, T+1), alpha[:, i], marker='o', label=self.states[i], linewidth=2)
        
        ax2.set_xlabel('时刻 t', fontsize=12)
        ax2.set_ylabel('前向概率 α_t(i)', fontsize=12)
        ax2.set_title('各状态前向概率变化曲线', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(range(1, T+1))
        
        # 在x轴上标注观测
        for t in range(T):
            ax2.text(t+1, -0.05, obs_seq[t], ha='center', va='top', 
                    transform=ax2.get_xaxis_transform(), fontsize=9, color='blue')
        
        plt.tight_layout()
        plt.savefig('hmm_forward.png', dpi=300, bbox_inches='tight')
        print("前向算法可视化图已保存为: hmm_forward.png\n")
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
    
    # 创建HMM模型
    hmm = HMMForward(states, observations, A, B, pi)
    
    # 观测序列
    obs_seq = ['红', '白', '红']
    
    # 执行前向算法
    prob, alpha = hmm.forward(obs_seq)
    
    # 可视化
    hmm.visualize_forward(obs_seq, alpha)
    
    # 额外示例
    print("\n" + "="*70)
    print("【额外示例】不同观测序列的概率计算")
    print("="*70)
    print()
    
    test_sequences = [
        ['红', '红', '红'],
        ['白', '白', '白'],
        ['红', '白', '白'],
        ['白', '红', '白']
    ]
    
    print("观测序列对比：")
    print(f"{'序列':<20} {'概率 P(O|λ)':<15}")
    print("-" * 35)
    
    results = []
    for seq in test_sequences:
        prob, _ = hmm.forward(seq)
        results.append((seq, prob))
        print(f"{str(seq):<20} {prob:.8f}")
    
    print()
    print("分析：")
    max_seq, max_prob = max(results, key=lambda x: x[1])
    min_seq, min_prob = min(results, key=lambda x: x[1])
    print(f"  最可能的观测序列: {max_seq}, P(O|λ) = {max_prob:.8f}")
    print(f"  最不可能的序列:   {min_seq}, P(O|λ) = {min_prob:.8f}")
    print()


if __name__ == "__main__":
    main()
