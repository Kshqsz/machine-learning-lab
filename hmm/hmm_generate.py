"""
隐马尔可夫模型 - 观测序列生成算法

问题描述：
有4个盒子，每个盒子里有红球和白球
{盒子1, 红球:5, 白球:5}
{盒子2, 红球:3, 白球:7}
{盒子3, 红球:6, 白球:4}
{盒子4, 红球:8, 白球:2}

状态转移规则：
- 盒子1 → 盒子2 (概率1.0)
- 盒子2 → 盒子1 (概率0.4) 或 盒子3 (概率0.6)
- 盒子3 → 盒子2 (概率0.4) 或 盒子4 (概率0.6)
- 盒子4 → 盒子3 (概率0.5) 或 盒子4 (概率0.5)

过程：
1. 随机选择一个盒子开始
2. 从当前盒子随机抽取一个球，记录颜色，放回
3. 按照转移规则转移到下一个盒子
4. 重复步骤2-3，生成观测序列

参考：《统计学习方法》(李航) 第10章 隐马尔可夫模型
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['Arial Unicode MS']
rcParams['axes.unicode_minus'] = False


class HMM:
    """隐马尔可夫模型"""
    
    def __init__(self, states, observations, transition_prob, emission_prob, initial_prob):
        """
        初始化HMM模型
        
        参数:
            states: 状态列表 (盒子编号)
            observations: 观测列表 (球的颜色)
            transition_prob: 状态转移概率矩阵 A[i][j] = P(q_{t+1}=j|q_t=i)
            emission_prob: 观测概率矩阵 B[i][k] = P(o_t=k|q_t=i)
            initial_prob: 初始状态概率分布 π[i] = P(q_1=i)
        """
        self.states = states  # 隐状态集合
        self.observations = observations  # 观测集合
        self.A = np.array(transition_prob)  # 状态转移概率矩阵
        self.B = np.array(emission_prob)  # 观测概率矩阵
        self.pi = np.array(initial_prob)  # 初始状态概率
        
        self.n_states = len(states)
        self.n_observations = len(observations)
        
    def generate_sequence(self, length, seed=None):
        """
        生成观测序列和对应的状态序列
        
        参数:
            length: 序列长度
            seed: 随机种子
            
        返回:
            states_seq: 状态序列 (盒子序列)
            obs_seq: 观测序列 (球颜色序列)
        """
        if seed is not None:
            np.random.seed(seed)
        
        states_seq = []  # 状态序列
        obs_seq = []  # 观测序列
        
        # 1. 根据初始状态概率选择初始状态
        current_state = np.random.choice(self.n_states, p=self.pi)
        states_seq.append(current_state)
        
        # 2. 根据当前状态的观测概率生成观测
        obs = np.random.choice(self.n_observations, p=self.B[current_state])
        obs_seq.append(obs)
        
        print(f"初始状态: {self.states[current_state]}")
        print(f"  观测概率: {dict(zip(self.observations, self.B[current_state]))}")
        print(f"  生成观测: {self.observations[obs]}\n")
        
        # 3. 迭代生成序列
        for t in range(1, length):
            # 根据当前状态的转移概率转移到下一个状态
            next_state = np.random.choice(self.n_states, p=self.A[current_state])
            states_seq.append(next_state)
            
            # 根据新状态的观测概率生成观测
            obs = np.random.choice(self.n_observations, p=self.B[next_state])
            obs_seq.append(obs)
            
            print(f"时刻 {t+1}:")
            print(f"  {self.states[current_state]} → {self.states[next_state]}")
            print(f"  转移概率: {self.A[current_state][next_state]:.2f}")
            print(f"  观测概率: {dict(zip(self.observations, self.B[next_state]))}")
            print(f"  生成观测: {self.observations[obs]}\n")
            
            current_state = next_state
        
        return states_seq, obs_seq
    
    def print_model(self):
        """打印模型参数"""
        print("=" * 60)
        print("隐马尔可夫模型参数")
        print("=" * 60)
        
        print("\n【状态集合】")
        for i, state in enumerate(self.states):
            print(f"  状态 {i}: {state}")
        
        print("\n【观测集合】")
        for i, obs in enumerate(self.observations):
            print(f"  观测 {i}: {obs}")
        
        print("\n【初始状态概率分布 π】")
        for i, prob in enumerate(self.pi):
            print(f"  P({self.states[i]}) = {prob:.2f}")
        
        print("\n【状态转移概率矩阵 A】")
        print("  从 \\ 到 ", end="")
        for state in self.states:
            print(f"{state:>8}", end="")
        print()
        for i, state_from in enumerate(self.states):
            print(f"  {state_from:>8}", end="")
            for j in range(self.n_states):
                print(f"{self.A[i][j]:>8.2f}", end="")
            print()
        
        print("\n【观测概率矩阵 B】")
        print("  状态 \\ 观测 ", end="")
        for obs in self.observations:
            print(f"{obs:>8}", end="")
        print()
        for i, state in enumerate(self.states):
            print(f"  {state:>11}", end="")
            for j in range(self.n_observations):
                print(f"{self.B[i][j]:>8.2f}", end="")
            print()
        print("=" * 60)
        print()


def visualize_sequence(hmm, states_seq, obs_seq):
    """可视化观测序列和状态序列"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    T = len(states_seq)
    time_steps = list(range(1, T + 1))
    
    # 绘制状态序列
    state_indices = states_seq
    ax1.plot(time_steps, state_indices, 'o-', markersize=12, linewidth=2, color='steelblue')
    ax1.set_yticks(range(hmm.n_states))
    ax1.set_yticklabels(hmm.states)
    ax1.set_xlabel('时刻 t', fontsize=12)
    ax1.set_ylabel('隐状态 (盒子)', fontsize=12)
    ax1.set_title('隐状态序列（盒子选择过程）', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(time_steps)
    
    # 在每个点上标注状态
    for i, (t, s) in enumerate(zip(time_steps, state_indices)):
        ax1.annotate(hmm.states[s], (t, s), textcoords="offset points", 
                    xytext=(0, 10), ha='center', fontsize=10, fontweight='bold')
    
    # 绘制观测序列
    obs_indices = obs_seq
    colors = ['red' if o == 0 else 'lightgray' for o in obs_indices]
    ax2.bar(time_steps, [1]*T, color=colors, edgecolor='black', linewidth=2, alpha=0.7)
    ax2.set_xlabel('时刻 t', fontsize=12)
    ax2.set_ylabel('观测 (球颜色)', fontsize=12)
    ax2.set_title('观测序列（抽取的球颜色）', fontsize=14, fontweight='bold')
    ax2.set_yticks([0.5])
    ax2.set_yticklabels([''])
    ax2.set_ylim(0, 1.5)
    ax2.set_xticks(time_steps)
    
    # 在每个柱子上标注颜色
    for i, (t, o) in enumerate(zip(time_steps, obs_indices)):
        color_text = hmm.observations[o]
        text_color = 'darkred' if o == 0 else 'black'
        ax2.text(t, 0.5, color_text, ha='center', va='center', 
                fontsize=11, fontweight='bold', color=text_color)
    
    plt.tight_layout()
    plt.savefig('hmm_generate.png', dpi=300, bbox_inches='tight')
    print("可视化图已保存为: hmm_generate.png")
    plt.close()


def main():
    """主函数"""
    
    # 定义状态和观测
    states = ['盒子1', '盒子2', '盒子3', '盒子4']
    observations = ['红球', '白球']
    
    # 初始状态概率（均匀分布，随机选择一个盒子开始）
    initial_prob = [0.25, 0.25, 0.25, 0.25]
    
    # 状态转移概率矩阵 A[i][j] = P(q_{t+1}=j|q_t=i)
    # 规则：
    # - 盒子1 → 盒子2 (概率1.0)
    # - 盒子2 → 盒子1 (0.4) 或 盒子3 (0.6)
    # - 盒子3 → 盒子2 (0.4) 或 盒子4 (0.6)
    # - 盒子4 → 盒子3 (0.5) 或 盒子4 (0.5)
    transition_prob = [
        [0.0, 1.0, 0.0, 0.0],  # 盒子1的转移
        [0.4, 0.0, 0.6, 0.0],  # 盒子2的转移
        [0.0, 0.4, 0.0, 0.6],  # 盒子3的转移
        [0.0, 0.0, 0.5, 0.5]   # 盒子4的转移
    ]
    
    # 观测概率矩阵 B[i][k] = P(o_t=k|q_t=i)
    # 每个盒子中红球和白球的比例
    # 盒子1: 红5, 白5 → [0.5, 0.5]
    # 盒子2: 红3, 白7 → [0.3, 0.7]
    # 盒子3: 红6, 白4 → [0.6, 0.4]
    # 盒子4: 红8, 白2 → [0.8, 0.2]
    emission_prob = [
        [0.5, 0.5],  # 盒子1
        [0.3, 0.7],  # 盒子2
        [0.6, 0.4],  # 盒子3
        [0.8, 0.2]   # 盒子4
    ]
    
    # 创建HMM模型
    hmm = HMM(states, observations, transition_prob, emission_prob, initial_prob)
    
    # 打印模型参数
    hmm.print_model()
    
    # 生成观测序列（长度为5）
    print("=" * 60)
    print("生成观测序列（序列长度 T = 5）")
    print("=" * 60)
    print()
    
    states_seq, obs_seq = hmm.generate_sequence(length=5, seed=42)
    
    # 输出结果
    print("=" * 60)
    print("生成结果")
    print("=" * 60)
    print(f"\n隐状态序列: {[states[i] for i in states_seq]}")
    print(f"观测序列:   {[observations[i] for i in obs_seq]}")
    print()
    
    # 可视化
    visualize_sequence(hmm, states_seq, obs_seq)
    
    # 再生成几个不同的序列示例
    print("\n" + "=" * 60)
    print("其他随机序列示例")
    print("=" * 60)
    
    for i in range(3):
        print(f"\n示例 {i+1}:")
        states_seq, obs_seq = hmm.generate_sequence(length=5, seed=None)
        print(f"  状态序列: {[states[s] for s in states_seq]}")
        print(f"  观测序列: {[observations[o] for o in obs_seq]}")


if __name__ == "__main__":
    main()
