"""
隐马尔可夫模型 - 维特比算法 (Viterbi Algorithm)

问题：预测问题（解码问题）
给定模型 λ=(A,B,π) 和观测序列 O=(o₁,o₂,...,o_T)，
求最可能的状态序列 Q*=(q₁*,q₂*,...,q_T*)

维特比算法：
- 使用动态规划求解最优路径
- 时间复杂度 O(N²T)
- 与前向算法类似，但用max代替sum

核心思想：
- 前向算法计算所有路径的概率和
- 维特比算法找到概率最大的单一路径

参考：《统计学习方法》(李航) 第10章 隐马尔可夫模型
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyArrowPatch

# 设置中文字体
rcParams['font.sans-serif'] = ['Arial Unicode MS']
rcParams['axes.unicode_minus'] = False


class HMMViterbi:
    """隐马尔可夫模型 - 维特比算法"""
    
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
        
    def viterbi(self, obs_seq, verbose=True):
        """
        维特比算法
        
        算法步骤：
        1. 初始化：δ₁(i) = πᵢ·bᵢ(o₁), ψ₁(i) = 0
        2. 递推：δₜ(i) = max_j[δₜ₋₁(j)·aⱼᵢ]·bᵢ(oₜ)
                 ψₜ(i) = argmax_j[δₜ₋₁(j)·aⱼᵢ]
        3. 终止：P* = max_i δ_T(i), q_T* = argmax_i δ_T(i)
        4. 回溯：qₜ* = ψₜ₊₁(qₜ₊₁*), t=T-1,T-2,...,1
        
        参数:
            obs_seq: 观测序列（观测值列表）
            verbose: 是否输出详细信息
            
        返回:
            best_path: 最优状态序列
            max_prob: 最大概率
            delta: δ矩阵 (T×N)
            psi: ψ矩阵 (T×N)
        """
        T = len(obs_seq)
        
        # 将观测序列转换为索引
        obs_indices = [self.observations.index(o) for o in obs_seq]
        
        # 初始化
        delta = np.zeros((T, self.N))  # δₜ(i): 到时刻t状态为i的最大概率
        psi = np.zeros((T, self.N), dtype=int)  # ψₜ(i): 到时刻t状态为i的最优前驱状态
        
        if verbose:
            print("=" * 70)
            print("维特比算法 - 最优状态序列求解")
            print("=" * 70)
            print(f"\n观测序列: {obs_seq}")
            print(f"观测序列长度 T = {T}")
            print()
        
        # 步骤1: 初始化 (t=1)
        if verbose:
            print("【步骤1：初始化】")
            print(f"计算 δ₁(i) = πᵢ · bᵢ(o₁), i=1,2,...,{self.N}")
            print(f"观测 o₁ = {obs_seq[0]}")
            print()
        
        for i in range(self.N):
            delta[0, i] = self.pi[i] * self.B[i, obs_indices[0]]
            psi[0, i] = 0
            
            if verbose:
                print(f"  δ₁({self.states[i]}) = π({self.states[i]}) × b_{self.states[i]}({obs_seq[0]})")
                print(f"              = {self.pi[i]:.4f} × {self.B[i, obs_indices[0]]:.4f}")
                print(f"              = {delta[0, i]:.6f}")
        
        if verbose:
            print(f"\nδ₁ = {delta[0]}")
            print()
        
        # 步骤2: 递推 (t=2,3,...,T)
        if verbose:
            print("【步骤2：递推】")
            print("计算 δₜ(i) = max_j[δₜ₋₁(j)·aⱼᵢ] · bᵢ(oₜ)")
            print("     ψₜ(i) = argmax_j[δₜ₋₁(j)·aⱼᵢ]")
            print()
        
        for t in range(1, T):
            if verbose:
                print(f"--- 时刻 t={t+1} ---")
                print(f"观测 o_{t+1} = {obs_seq[t]}")
                print()
            
            for i in range(self.N):
                # 计算 max_j[δₜ₋₁(j)·aⱼᵢ]
                probs = delta[t-1] * self.A[:, i]
                max_prob = np.max(probs)
                max_state = np.argmax(probs)
                
                delta[t, i] = max_prob * self.B[i, obs_indices[t]]
                psi[t, i] = max_state
                
                if verbose:
                    print(f"  δ_{t+1}({self.states[i]}):")
                    details = []
                    for j in range(self.N):
                        details.append(f"δ_{t}({self.states[j]})×a_{self.states[j]},{self.states[i]}={delta[t-1,j]:.6f}×{self.A[j,i]:.2f}={probs[j]:.6f}")
                    print(f"    max_j[{', '.join(details)}]")
                    print(f"    = max({', '.join([f'{p:.6f}' for p in probs])})")
                    print(f"    = {max_prob:.6f} (来自状态 {self.states[max_state]})")
                    print(f"    δ_{t+1}({self.states[i]}) = {max_prob:.6f} × {self.B[i, obs_indices[t]]:.4f} = {delta[t, i]:.6f}")
                    print(f"    ψ_{t+1}({self.states[i]}) = {self.states[max_state]}")
            
            if verbose:
                print(f"\nδ_{t+1} = {delta[t]}")
                print(f"ψ_{t+1} = {[self.states[psi[t, i]] for i in range(self.N)]}")
                print()
        
        # 步骤3: 终止
        if verbose:
            print("【步骤3：终止】")
        
        max_prob = np.max(delta[T-1])
        last_state = np.argmax(delta[T-1])
        
        if verbose:
            print(f"P* = max_i δ_T(i) = max({', '.join([f'{delta[T-1,i]:.6f}' for i in range(self.N)])})")
            print(f"   = {max_prob:.6f}")
            print(f"q_T* = argmax_i δ_T(i) = {self.states[last_state]}")
            print()
        
        # 步骤4: 回溯最优路径
        if verbose:
            print("【步骤4：回溯最优路径】")
        
        best_path = [0] * T
        best_path[T-1] = last_state
        
        if verbose:
            print(f"q_{T}* = {self.states[last_state]}")
        
        for t in range(T-2, -1, -1):
            best_path[t] = psi[t+1, best_path[t+1]]
            if verbose:
                print(f"q_{t+1}* = ψ_{t+2}(q_{t+2}*) = ψ_{t+2}({self.states[best_path[t+1]]}) = {self.states[best_path[t]]}")
        
        # 转换为状态名称
        best_path_states = [self.states[i] for i in best_path]
        
        if verbose:
            print()
            print("=" * 70)
            print("【结果】")
            print("=" * 70)
            print(f"观测序列: {obs_seq}")
            print(f"最优状态序列: {best_path_states}")
            print(f"最大概率: {max_prob:.8f}")
            print("=" * 70)
            print()
        
        return best_path_states, max_prob, delta, psi
    
    def visualize_viterbi(self, obs_seq, best_path, delta, psi):
        """可视化维特比算法"""
        T = len(obs_seq)
        
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. δ矩阵热力图
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(delta.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax1.set_xlabel('时刻 t', fontsize=11)
        ax1.set_ylabel('状态', fontsize=11)
        ax1.set_title('δₜ(i) - 局部最大概率', fontsize=12, fontweight='bold')
        ax1.set_xticks(range(T))
        ax1.set_xticklabels([f't={i+1}\n{obs_seq[i]}' for i in range(T)], fontsize=9)
        ax1.set_yticks(range(self.N))
        ax1.set_yticklabels(self.states)
        
        # 添加数值
        for i in range(self.N):
            for t in range(T):
                ax1.text(t, i, f'{delta[t, i]:.4f}', ha="center", va="center", 
                        color="black", fontsize=8)
        plt.colorbar(im1, ax=ax1, label='概率值')
        
        # 2. ψ矩阵（最优前驱状态）
        ax2 = fig.add_subplot(gs[0, 1])
        psi_visual = psi.T.astype(float)
        im2 = ax2.imshow(psi_visual, aspect='auto', cmap='Blues', interpolation='nearest')
        ax2.set_xlabel('时刻 t', fontsize=11)
        ax2.set_ylabel('状态', fontsize=11)
        ax2.set_title('ψₜ(i) - 最优前驱状态', fontsize=12, fontweight='bold')
        ax2.set_xticks(range(T))
        ax2.set_xticklabels([f't={i+1}\n{obs_seq[i]}' for i in range(T)], fontsize=9)
        ax2.set_yticks(range(self.N))
        ax2.set_yticklabels(self.states)
        
        # 添加前驱状态名称
        for i in range(self.N):
            for t in range(T):
                ax2.text(t, i, self.states[psi[t, i]], ha="center", va="center", 
                        color="darkblue", fontsize=9, fontweight='bold')
        plt.colorbar(im2, ax=ax2)
        
        # 3. 最优路径可视化
        ax3 = fig.add_subplot(gs[1, :])
        best_path_indices = [self.states.index(s) for s in best_path]
        
        # 绘制所有可能的状态
        for t in range(T):
            for i in range(self.N):
                color = 'lightcoral' if i == best_path_indices[t] else 'lightgray'
                alpha = 1.0 if i == best_path_indices[t] else 0.3
                ax3.plot(t, i, 'o', markersize=20, color=color, alpha=alpha, 
                        markeredgecolor='black', markeredgewidth=2)
                # 标注状态名
                if i == best_path_indices[t]:
                    ax3.text(t, i, self.states[i], ha='center', va='center', 
                            fontsize=10, fontweight='bold', color='darkred')
        
        # 绘制最优路径连线
        for t in range(T-1):
            ax3.plot([t, t+1], [best_path_indices[t], best_path_indices[t+1]], 
                    'r-', linewidth=3, alpha=0.7, zorder=10)
            # 添加箭头
            ax3.annotate('', xy=(t+1, best_path_indices[t+1]), 
                        xytext=(t, best_path_indices[t]),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2))
        
        ax3.set_xlabel('时刻 t', fontsize=12)
        ax3.set_ylabel('状态', fontsize=12)
        ax3.set_title('最优状态路径（维特比路径）', fontsize=14, fontweight='bold')
        ax3.set_xticks(range(T))
        ax3.set_xticklabels([f't={i+1}\n观测:{obs_seq[i]}' for i in range(T)])
        ax3.set_yticks(range(self.N))
        ax3.set_yticklabels(self.states)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(-0.5, T-0.5)
        ax3.set_ylim(-0.5, self.N-0.5)
        
        # 4. δ曲线图
        ax4 = fig.add_subplot(gs[2, 0])
        for i in range(self.N):
            line_style = '-' if self.states[i] in best_path else '--'
            line_width = 3 if self.states[i] in best_path else 1
            alpha = 1.0 if self.states[i] in best_path else 0.5
            ax4.plot(range(1, T+1), delta[:, i], marker='o', label=self.states[i], 
                    linewidth=line_width, linestyle=line_style, alpha=alpha)
        
        ax4.set_xlabel('时刻 t', fontsize=11)
        ax4.set_ylabel('δₜ(i) - 局部最大概率', fontsize=11)
        ax4.set_title('各状态的局部最大概率变化', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        ax4.set_xticks(range(1, T+1))
        
        # 5. 观测序列和最优状态对应
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')
        
        # 创建表格
        table_data = [['时刻', '观测', '最优状态', 'δ值']]
        for t in range(T):
            state_idx = best_path_indices[t]
            table_data.append([
                f't={t+1}',
                obs_seq[t],
                best_path[t],
                f'{delta[t, state_idx]:.6f}'
            ])
        
        table = ax5.table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.15, 0.2, 0.3, 0.35])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # 设置表头样式
        for i in range(4):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 设置数据行样式
        for i in range(1, len(table_data)):
            for j in range(4):
                table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        ax5.set_title('最优路径详细信息', fontsize=12, fontweight='bold', pad=20)
        
        plt.savefig('hmm_viterbi.png', dpi=300, bbox_inches='tight')
        print("维特比算法可视化图已保存为: hmm_viterbi.png\n")
        plt.close()


def compare_forward_viterbi(hmm, obs_seq):
    """对比前向算法和维特比算法"""
    print("\n" + "="*70)
    print("前向算法 vs 维特比算法 - 对比分析")
    print("="*70)
    print()
    
    # 前向算法（所有路径的概率和）
    from hmm_forward import HMMForward
    hmm_forward = HMMForward(hmm.states, hmm.observations, hmm.A, hmm.B, hmm.pi)
    prob_forward, alpha = hmm_forward.forward(obs_seq)
    
    # 维特比算法（最优路径的概率）
    best_path, prob_viterbi, delta, psi = hmm.viterbi(obs_seq, verbose=False)
    
    print(f"观测序列: {obs_seq}\n")
    
    print("【前向算法】")
    print(f"  计算方式: 所有可能路径的概率和")
    print(f"  P(O|λ) = Σ P(O,Q|λ) = {prob_forward:.8f}")
    print()
    
    print("【维特比算法】")
    print(f"  计算方式: 概率最大的单一路径")
    print(f"  P* = max P(O,Q|λ) = {prob_viterbi:.8f}")
    print(f"  最优状态序列: {best_path}")
    print()
    
    print("【对比】")
    print(f"  P(所有路径) >= P(最优路径): {prob_forward:.8f} >= {prob_viterbi:.8f}")
    print(f"  关系成立: {prob_forward >= prob_viterbi}")
    print(f"  比值: P(所有路径)/P(最优路径) = {prob_forward/prob_viterbi:.2f}")
    print()
    
    print("【解释】")
    print("  - 前向算法计算的是所有路径的概率总和")
    print("  - 维特比算法只找最优的那条路径")
    print("  - 因此 P(所有路径) >= P(最优路径) 总是成立")
    print("  - 维特比算法用于解码：找出最可能的隐藏状态序列")
    print("  - 前向算法用于评估：计算观测序列在模型下的概率")
    print("="*70)
    print()


def main():
    """主函数 - 李航书例10.3"""
    
    print("\n" + "="*70)
    print("示例：盒子抽球模型（李航《统计学习方法》例10.3）")
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
    hmm = HMMViterbi(states, observations, A, B, pi)
    
    # 观测序列
    obs_seq = ['红', '白', '红']
    
    # 执行维特比算法
    best_path, max_prob, delta, psi = hmm.viterbi(obs_seq)
    
    # 可视化
    hmm.visualize_viterbi(obs_seq, best_path, delta, psi)
    
    # 对比前向算法和维特比算法
    compare_forward_viterbi(hmm, obs_seq)
    
    # 测试多个观测序列
    print("\n" + "="*70)
    print("【多序列测试】")
    print("="*70)
    print()
    
    test_sequences = [
        ['红', '红', '红'],
        ['白', '白', '白'],
        ['红', '白', '白'],
        ['白', '红', '白', '红']
    ]
    
    print(f"{'观测序列':<25} {'最优状态序列':<40} {'最大概率':<15}")
    print("-" * 80)
    
    for seq in test_sequences:
        path, prob, _, _ = hmm.viterbi(seq, verbose=False)
        print(f"{str(seq):<25} {str(path):<40} {prob:.8f}")
    
    print()
    
    # 分析
    print("【分析】")
    print("观察最优状态序列的规律：")
    print("- 观测到'红'时，倾向于选择红球概率高的盒子（盒子3, P(红)=0.7）")
    print("- 观测到'白'时，倾向于选择白球概率高的盒子（盒子2, P(白)=0.6）")
    print("- 但也要考虑状态转移概率，不会频繁跳转")
    print("- 维特比算法在观测概率和转移概率之间找到最优平衡")
    print()


if __name__ == "__main__":
    main()
