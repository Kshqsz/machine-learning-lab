"""
隐马尔可夫模型 - Baum-Welch算法 (EM算法)

问题：学习问题
给定观测序列 O=(o₁,o₂,...,o_T)，估计模型参数 λ=(A,B,π)，使得 P(O|λ) 最大

Baum-Welch算法是EM算法在HMM中的应用：
- E步：利用当前参数计算期望（γ_t(i) 和 ξ_t(i,j)）
- M步：最大化期望，更新参数

算法本质：
- 基于前向-后向算法
- 迭代优化，保证似然函数单调不减
- 收敛到局部最优解

参考：《统计学习方法》(李航) 第10章 隐马尔可夫模型
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['Arial Unicode MS']
rcParams['axes.unicode_minus'] = False


class HMMBaumWelch:
    """隐马尔可夫模型 - Baum-Welch算法（EM算法）"""
    
    def __init__(self, n_states, n_observations):
        """
        初始化HMM模型
        
        参数:
            n_states: 状态数量 N
            n_observations: 观测数量 M
        """
        self.N = n_states
        self.M = n_observations
        
        # 随机初始化参数
        self.A = None  # 状态转移概率矩阵 (N×N)
        self.B = None  # 观测概率矩阵 (N×M)
        self.pi = None  # 初始状态概率向量 (N)
        
    def initialize_params(self, method='random', seed=None):
        """
        初始化模型参数
        
        参数:
            method: 'random' 随机初始化, 'uniform' 均匀初始化
            seed: 随机种子
        """
        if seed is not None:
            np.random.seed(seed)
        
        if method == 'random':
            # 随机初始化
            self.pi = np.random.rand(self.N)
            self.pi /= self.pi.sum()
            
            self.A = np.random.rand(self.N, self.N)
            self.A /= self.A.sum(axis=1, keepdims=True)
            
            self.B = np.random.rand(self.N, self.M)
            self.B /= self.B.sum(axis=1, keepdims=True)
            
        elif method == 'uniform':
            # 均匀初始化
            self.pi = np.ones(self.N) / self.N
            self.A = np.ones((self.N, self.N)) / self.N
            self.B = np.ones((self.N, self.M)) / self.M
    
    def set_params(self, A, B, pi):
        """手动设置模型参数"""
        self.A = np.array(A)
        self.B = np.array(B)
        self.pi = np.array(pi)
    
    def forward(self, obs_seq):
        """
        前向算法（使用缩放因子避免下溢）
        
        返回:
            alpha: 前向概率矩阵 (T×N)
            log_prob: 观测序列的对数似然
        """
        T = len(obs_seq)
        alpha = np.zeros((T, self.N))
        scale = np.zeros(T)  # 缩放因子
        
        # 初始化
        alpha[0] = self.pi * self.B[:, obs_seq[0]]
        scale[0] = alpha[0].sum()
        alpha[0] /= scale[0]
        
        # 递推
        for t in range(1, T):
            for i in range(self.N):
                alpha[t, i] = np.sum(alpha[t-1] * self.A[:, i]) * self.B[i, obs_seq[t]]
            scale[t] = alpha[t].sum()
            if scale[t] > 0:
                alpha[t] /= scale[t]
        
        # 计算对数似然
        log_prob = np.sum(np.log(scale + 1e-10))
        
        return alpha, log_prob
    
    def backward(self, obs_seq):
        """
        后向算法（使用与前向算法相同的缩放因子）
        
        返回:
            beta: 后向概率矩阵 (T×N)
        """
        T = len(obs_seq)
        beta = np.zeros((T, self.N))
        scale = np.zeros(T)
        
        # 初始化
        beta[T-1] = 1.0
        
        # 先计算缩放因子（通过前向算法）
        alpha_scaled = np.zeros((T, self.N))
        alpha_scaled[0] = self.pi * self.B[:, obs_seq[0]]
        scale[0] = alpha_scaled[0].sum()
        alpha_scaled[0] /= scale[0]
        
        for t in range(1, T):
            for i in range(self.N):
                alpha_scaled[t, i] = np.sum(alpha_scaled[t-1] * self.A[:, i]) * self.B[i, obs_seq[t]]
            scale[t] = alpha_scaled[t].sum()
            if scale[t] > 0:
                alpha_scaled[t] /= scale[t]
        
        # 后向递推（使用相同的缩放因子）
        for t in range(T-2, -1, -1):
            for i in range(self.N):
                beta[t, i] = np.sum(self.A[i] * self.B[:, obs_seq[t+1]] * beta[t+1])
            if scale[t] > 0:
                beta[t] /= scale[t]
        
        return beta
    
    def compute_gamma(self, alpha, beta):
        """
        计算 γ_t(i) = P(q_t=s_i|O,λ)
        在给定观测序列O和模型λ的条件下，时刻t处于状态s_i的概率
        
        参数:
            alpha: 前向概率 (T×N)
            beta: 后向概率 (T×N)
            
        返回:
            gamma: γ_t(i) (T×N)
        """
        T = alpha.shape[0]
        gamma = alpha * beta
        gamma /= (gamma.sum(axis=1, keepdims=True) + 1e-10)
        
        return gamma
    
    def compute_xi(self, alpha, beta, obs_seq):
        """
        计算 ξ_t(i,j) = P(q_t=s_i, q_{t+1}=s_j|O,λ)
        在给定观测序列O和模型λ的条件下，时刻t处于状态s_i且时刻t+1处于状态s_j的概率
        
        参数:
            alpha: 前向概率 (T×N)
            beta: 后向概率 (T×N)
            obs_seq: 观测序列
            
        返回:
            xi: ξ_t(i,j) (T-1×N×N)
        """
        T = len(obs_seq)
        xi = np.zeros((T-1, self.N, self.N))
        
        for t in range(T-1):
            denominator = 0.0
            for i in range(self.N):
                for j in range(self.N):
                    xi[t, i, j] = alpha[t, i] * self.A[i, j] * \
                                  self.B[j, obs_seq[t+1]] * beta[t+1, j]
                    denominator += xi[t, i, j]
            
            xi[t] /= (denominator + 1e-10)
        
        return xi
    
    def baum_welch(self, obs_sequences, max_iter=100, tol=1e-4, verbose=True):
        """
        Baum-Welch算法（EM算法）
        
        参数:
            obs_sequences: 观测序列列表（可以是单个序列或多个序列）
            max_iter: 最大迭代次数
            tol: 收敛阈值
            verbose: 是否输出详细信息
            
        返回:
            log_probs: 每次迭代的对数似然列表
        """
        # 确保输入是列表
        if not isinstance(obs_sequences[0], (list, np.ndarray)):
            obs_sequences = [obs_sequences]
        
        log_probs = []
        
        if verbose:
            print("=" * 70)
            print("Baum-Welch算法 - HMM参数学习")
            print("=" * 70)
            print(f"\n观测序列数量: {len(obs_sequences)}")
            print(f"状态数 N = {self.N}")
            print(f"观测数 M = {self.M}")
            print(f"最大迭代次数: {max_iter}")
            print(f"收敛阈值: {tol}")
            print()
            
            print("初始参数:")
            self._print_params()
        
        for iteration in range(max_iter):
            # 累加统计量
            pi_sum = np.zeros(self.N)
            A_num = np.zeros((self.N, self.N))
            A_den = np.zeros(self.N)
            B_num = np.zeros((self.N, self.M))
            B_den = np.zeros(self.N)
            
            total_log_prob = 0.0
            
            # E步：对每个观测序列计算期望
            for obs_seq in obs_sequences:
                T = len(obs_seq)
                
                # 前向-后向算法
                alpha, log_prob = self.forward(obs_seq)
                beta = self.backward(obs_seq)
                
                total_log_prob += log_prob
                
                # 计算 γ 和 ξ
                gamma = self.compute_gamma(alpha, beta)
                xi = self.compute_xi(alpha, beta, obs_seq)
                
                # 累加初始状态概率
                pi_sum += gamma[0]
                
                # 累加状态转移概率统计量
                A_num += xi.sum(axis=0)  # Σ_t ξ_t(i,j)
                A_den += gamma[:-1].sum(axis=0)  # Σ_t γ_t(i)
                
                # 累加观测概率统计量
                for t in range(T):
                    B_num[:, obs_seq[t]] += gamma[t]
                B_den += gamma.sum(axis=0)
            
            log_probs.append(total_log_prob)
            
            if verbose:
                print(f"迭代 {iteration+1}: 对数似然 = {total_log_prob:.6f}")
            
            # 检查收敛
            if iteration > 0 and abs(log_probs[-1] - log_probs[-2]) < tol:
                if verbose:
                    print(f"\n算法收敛！迭代次数: {iteration+1}")
                break
            
            # M步：更新参数
            # 更新初始概率
            self.pi = pi_sum / len(obs_sequences)
            
            # 更新状态转移概率
            for i in range(self.N):
                if A_den[i] > 0:
                    self.A[i] = A_num[i] / A_den[i]
                else:
                    self.A[i] = np.ones(self.N) / self.N
            
            # 更新观测概率
            for i in range(self.N):
                if B_den[i] > 0:
                    self.B[i] = B_num[i] / B_den[i]
                else:
                    self.B[i] = np.ones(self.M) / self.M
        
        if verbose:
            print("\n" + "=" * 70)
            print("学习完成！最终参数:")
            self._print_params()
        
        return log_probs
    
    def _print_params(self):
        """打印模型参数"""
        print("\n初始状态概率 π:")
        for i in range(self.N):
            print(f"  π[{i}] = {self.pi[i]:.4f}", end="  ")
        print()
        
        print("\n状态转移概率矩阵 A:")
        print("  ", end="")
        for j in range(self.N):
            print(f"状态{j:>6}", end="")
        print()
        for i in range(self.N):
            print(f"  状态{i}", end="")
            for j in range(self.N):
                print(f"{self.A[i,j]:>7.4f}", end="")
            print()
        
        print("\n观测概率矩阵 B:")
        print("  ", end="")
        for k in range(self.M):
            print(f"观测{k:>6}", end="")
        print()
        for i in range(self.N):
            print(f"  状态{i}", end="")
            for k in range(self.M):
                print(f"{self.B[i,k]:>7.4f}", end="")
            print()
        print()
    
    def predict(self, obs_seq):
        """预测观测序列的概率"""
        alpha, log_prob = self.forward(obs_seq)
        prob = np.exp(log_prob)
        return prob


def visualize_learning(log_probs, title="Baum-Welch算法学习曲线"):
    """可视化学习过程"""
    plt.figure(figsize=(10, 6))
    
    iterations = range(1, len(log_probs) + 1)
    plt.plot(iterations, log_probs, 'o-', linewidth=2, markersize=8, color='steelblue')
    
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('对数似然 Log P(O|λ)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 标注最终值
    final_val = log_probs[-1]
    plt.axhline(y=final_val, color='red', linestyle='--', alpha=0.5, 
                label=f'最终值: {final_val:.6f}')
    
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('hmm_baum_welch_learning_curve.png', dpi=300, bbox_inches='tight')
    print("学习曲线已保存为: hmm_baum_welch_learning_curve.png")
    plt.close()


def compare_params(true_params, learned_params, param_name):
    """对比真实参数和学习参数"""
    print(f"\n{param_name} 对比:")
    print(f"  真实值: {true_params}")
    print(f"  学习值: {learned_params}")
    error = np.abs(true_params - learned_params).mean()
    print(f"  平均绝对误差: {error:.6f}")
    return error


def main():
    """主函数 - 示例"""
    
    print("\n" + "="*70)
    print("示例1：已知真实参数，验证Baum-Welch算法")
    print("="*70)
    
    # 设置真实的HMM参数
    N = 3  # 3个状态
    M = 2  # 2个观测（红、白）
    
    true_pi = np.array([0.2, 0.4, 0.4])
    true_A = np.array([
        [0.5, 0.2, 0.3],
        [0.3, 0.5, 0.2],
        [0.2, 0.3, 0.5]
    ])
    true_B = np.array([
        [0.5, 0.5],  # 状态0: P(红)=0.5, P(白)=0.5
        [0.4, 0.6],  # 状态1: P(红)=0.4, P(白)=0.6
        [0.7, 0.3]   # 状态2: P(红)=0.7, P(白)=0.3
    ])
    
    print("\n真实的HMM参数:")
    print(f"π = {true_pi}")
    print(f"A =\n{true_A}")
    print(f"B =\n{true_B}")
    
    # 使用真实模型生成观测序列
    print("\n生成训练数据...")
    np.random.seed(42)
    
    def generate_sequence(pi, A, B, T):
        """生成观测序列"""
        states = []
        obs = []
        
        # 初始状态
        state = np.random.choice(N, p=pi)
        states.append(state)
        obs.append(np.random.choice(M, p=B[state]))
        
        # 后续状态
        for t in range(1, T):
            state = np.random.choice(N, p=A[state])
            states.append(state)
            obs.append(np.random.choice(M, p=B[state]))
        
        return states, obs
    
    # 生成多条观测序列
    obs_sequences = []
    for i in range(10):
        _, obs = generate_sequence(true_pi, true_A, true_B, 50)
        obs_sequences.append(obs)
    
    print(f"生成了 {len(obs_sequences)} 条观测序列，每条长度为 50")
    print(f"示例序列: {obs_sequences[0][:20]}...")
    
    # 创建HMM模型并随机初始化
    print("\n" + "="*70)
    print("开始Baum-Welch算法训练...")
    print("="*70)
    
    hmm = HMMBaumWelch(N, M)
    hmm.initialize_params(method='random', seed=123)
    
    # 运行Baum-Welch算法
    log_probs = hmm.baum_welch(obs_sequences, max_iter=50, tol=1e-6, verbose=True)
    
    # 可视化学习过程
    visualize_learning(log_probs)
    
    # 对比参数
    print("\n" + "="*70)
    print("参数对比分析")
    print("="*70)
    
    error_pi = compare_params(true_pi, hmm.pi, "初始概率 π")
    error_A = compare_params(true_A.flatten(), hmm.A.flatten(), "转移概率 A")
    error_B = compare_params(true_B.flatten(), hmm.B.flatten(), "观测概率 B")
    
    print(f"\n总体平均误差: {(error_pi + error_A + error_B) / 3:.6f}")
    
    # 测试预测
    print("\n" + "="*70)
    print("测试学习后的模型")
    print("="*70)
    
    test_seq = obs_sequences[0][:10]
    print(f"\n测试序列: {test_seq}")
    
    # 使用真实模型计算概率
    hmm_true = HMMBaumWelch(N, M)
    hmm_true.set_params(true_A, true_B, true_pi)
    prob_true = hmm_true.predict(test_seq)
    
    # 使用学习模型计算概率
    prob_learned = hmm.predict(test_seq)
    
    print(f"\n真实模型预测概率: {prob_true:.8f}")
    print(f"学习模型预测概率: {prob_learned:.8f}")
    print(f"概率相对误差: {abs(prob_true - prob_learned) / prob_true * 100:.2f}%")
    
    # 示例2：无监督学习
    print("\n\n" + "="*70)
    print("示例2：纯无监督学习（不知道真实参数）")
    print("="*70)
    
    # 假设我们只有观测序列，不知道真实参数
    # 观测序列：0=红球，1=白球
    unknown_obs = [
        [0, 1, 0, 0, 1, 1, 0, 1, 0, 0],
        [1, 0, 1, 0, 0, 1, 0, 1, 1, 0],
        [0, 0, 1, 1, 0, 1, 0, 0, 1, 1],
        [1, 1, 0, 1, 0, 0, 1, 0, 1, 0],
        [0, 1, 1, 0, 1, 0, 0, 1, 0, 1]
    ]
    
    print(f"\n观测序列数量: {len(unknown_obs)}")
    print("观测序列示例:")
    for i, seq in enumerate(unknown_obs):
        print(f"  序列{i+1}: {seq}")
    
    # 创建新模型
    hmm_unknown = HMMBaumWelch(n_states=2, n_observations=2)
    hmm_unknown.initialize_params(method='uniform', seed=456)
    
    print("\n开始无监督学习...")
    log_probs_unknown = hmm_unknown.baum_welch(
        unknown_obs, 
        max_iter=30, 
        tol=1e-5, 
        verbose=True
    )
    
    visualize_learning(log_probs_unknown, "无监督学习 - Baum-Welch算法")
    
    print("\n学习完成！模型已自动发现隐藏状态的规律。")


if __name__ == "__main__":
    main()
