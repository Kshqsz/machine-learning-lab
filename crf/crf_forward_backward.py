"""
线性链条件随机场 - 前向后向算法
Linear-chain Conditional Random Field - Forward-Backward Algorithm

实现CRF的前向-后向算法，用于计算：
1. 归一化因子 Z(x)
2. 边缘概率 P(y_i | x)
3. 特征期望值（用于梯度计算）

"""

import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False


class LinearChainCRF:
    """
    线性链条件随机场
    
    模型形式：
    P(y|x) = (1/Z(x)) * exp(Σ_i Σ_k λ_k * t_k(y_{i-1}, y_i, x, i) + Σ_i Σ_l μ_l * s_l(y_i, x, i))
    
    其中：
    - t_k: 转移特征函数
    - s_l: 状态特征函数
    - λ_k, μ_l: 特征权重
    - Z(x): 归一化因子
    """
    
    def __init__(self, states: List[str], features: List[str], verbose: bool = True):
        """
        初始化CRF模型
        
        参数:
            states: 状态（标签）集合，如 ['B', 'M', 'E', 'S']（中文分词）
            features: 特征名称列表
            verbose: 是否输出详细日志
        """
        self.states = states
        self.n_states = len(states)
        self.state_to_idx = {s: i for i, s in enumerate(states)}
        self.features = features
        self.n_features = len(features)
        self.verbose = verbose
        
        # 特征权重（需要训练）
        self.weights = None
    
    def set_weights(self, weights: np.ndarray):
        """设置特征权重"""
        self.weights = weights
    
    def compute_feature_vector(self, x: List[str], i: int, y_prev: int, y_curr: int) -> np.ndarray:
        """
        计算特征向量 f(y_{i-1}, y_i, x, i)
        
        在实际应用中，这个函数需要根据具体任务定义特征函数
        这里使用简化的特征函数作为示例
        
        参数:
            x: 观测序列
            i: 当前位置
            y_prev: 前一个状态的索引
            y_curr: 当前状态的索引
        
        返回:
            特征向量
        """
        features = np.zeros(self.n_features)
        
        # 简化示例：使用基本的转移特征和状态特征
        # 实际应用中需要根据任务定义更复杂的特征
        
        # 转移特征：y_{i-1} -> y_i
        trans_idx = y_prev * self.n_states + y_curr
        if trans_idx < self.n_features // 2:
            features[trans_idx] = 1.0
        
        # 状态特征：当前观测x[i]与状态y_i的组合
        if i < len(x):
            # 这里使用简单的哈希作为示例
            obs_idx = (hash(x[i]) % (self.n_features // 2)) + self.n_features // 2
            if obs_idx < self.n_features:
                features[obs_idx] = 1.0
        
        return features
    
    def compute_potential(self, x: List[str], i: int, y_prev: int, y_curr: int) -> float:
        """
        计算势函数 Ψ_i(y_{i-1}, y_i | x)
        
        Ψ_i(y_{i-1}, y_i | x) = exp(w^T * f(y_{i-1}, y_i, x, i))
        
        参数:
            x: 观测序列
            i: 当前位置
            y_prev: 前一个状态的索引
            y_curr: 当前状态的索引
        
        返回:
            势函数值
        """
        if self.weights is None:
            # 如果没有权重，使用均匀分布
            return 1.0
        
        features = self.compute_feature_vector(x, i, y_prev, y_curr)
        score = np.dot(self.weights, features)
        return np.exp(score)
    
    def forward(self, x: List[str]) -> Tuple[np.ndarray, float]:
        """
        前向算法：计算前向概率 α
        
        α_i(y_i) = Σ_{y_{i-1}} α_{i-1}(y_{i-1}) * Ψ_i(y_{i-1}, y_i | x)
        
        参数:
            x: 观测序列
        
        返回:
            alpha: 前向概率矩阵 [T, n_states]
            Z: 归一化因子 Z(x)
        """
        T = len(x)
        alpha = np.zeros((T, self.n_states))
        
        if self.verbose:
            print("\n" + "=" * 70)
            print("前向算法 - Forward Algorithm")
            print("=" * 70)
            print(f"观测序列: {x}")
            print(f"序列长度 T = {T}")
            print(f"状态集合: {self.states}")
        
        # 初始化 (i=0)
        if self.verbose:
            print("\n【步骤1：初始化】")
            print("计算 α_0(y_0) = Ψ_0(start, y_0 | x)")
        
        for y in range(self.n_states):
            # 从虚拟起始状态到第一个状态
            alpha[0, y] = self.compute_potential(x, 0, -1, y)
            if self.verbose:
                print(f"  α_0({self.states[y]}) = {alpha[0, y]:.6f}")
        
        # 递推 (i=1 to T-1)
        if self.verbose:
            print("\n【步骤2：递推】")
            print("计算 α_i(y_i) = Σ_{y_{i-1}} α_{i-1}(y_{i-1}) * Ψ_i(y_{i-1}, y_i | x)")
        
        for i in range(1, T):
            if self.verbose:
                print(f"\n--- 时刻 i={i}, 观测 x_{i} = '{x[i]}' ---")
            
            for y_curr in range(self.n_states):
                sum_val = 0.0
                details = []
                
                for y_prev in range(self.n_states):
                    potential = self.compute_potential(x, i, y_prev, y_curr)
                    contribution = alpha[i-1, y_prev] * potential
                    sum_val += contribution
                    details.append(f"α_{i-1}({self.states[y_prev]})×Ψ={alpha[i-1, y_prev]:.4f}×{potential:.4f}={contribution:.4f}")
                
                alpha[i, y_curr] = sum_val
                
                if self.verbose:
                    print(f"  α_{i}({self.states[y_curr]}):")
                    for detail in details:
                        print(f"    {detail}")
                    print(f"    = {sum_val:.6f}")
        
        # 终止：计算归一化因子
        Z = np.sum(alpha[-1, :])
        
        if self.verbose:
            print("\n【步骤3：终止】")
            print("计算归一化因子 Z(x) = Σ_y α_T(y)")
            for y in range(self.n_states):
                print(f"  α_{T-1}({self.states[y]}) = {alpha[-1, y]:.6f}")
            print(f"\nZ(x) = {Z:.6f}")
            print("=" * 70)
        
        return alpha, Z
    
    def backward(self, x: List[str]) -> np.ndarray:
        """
        后向算法：计算后向概率 β
        
        β_i(y_i) = Σ_{y_{i+1}} Ψ_{i+1}(y_i, y_{i+1} | x) * β_{i+1}(y_{i+1})
        
        参数:
            x: 观测序列
        
        返回:
            beta: 后向概率矩阵 [T, n_states]
        """
        T = len(x)
        beta = np.zeros((T, self.n_states))
        
        if self.verbose:
            print("\n" + "=" * 70)
            print("后向算法 - Backward Algorithm")
            print("=" * 70)
            print(f"观测序列: {x}")
        
        # 初始化 (i=T-1)
        if self.verbose:
            print("\n【步骤1：初始化】")
            print("设置 β_{T-1}(y) = 1 for all y")
        
        beta[-1, :] = 1.0
        
        for y in range(self.n_states):
            if self.verbose:
                print(f"  β_{T-1}({self.states[y]}) = {beta[-1, y]:.6f}")
        
        # 递推 (i=T-2 down to 0)
        if self.verbose:
            print("\n【步骤2：递推】")
            print("计算 β_i(y_i) = Σ_{y_{i+1}} Ψ_{i+1}(y_i, y_{i+1} | x) * β_{i+1}(y_{i+1})")
        
        for i in range(T-2, -1, -1):
            if self.verbose:
                print(f"\n--- 时刻 i={i}, 观测 x_{i+1} = '{x[i+1]}' ---")
            
            for y_curr in range(self.n_states):
                sum_val = 0.0
                details = []
                
                for y_next in range(self.n_states):
                    potential = self.compute_potential(x, i+1, y_curr, y_next)
                    contribution = potential * beta[i+1, y_next]
                    sum_val += contribution
                    details.append(f"Ψ×β_{i+1}({self.states[y_next]})={potential:.4f}×{beta[i+1, y_next]:.4f}={contribution:.4f}")
                
                beta[i, y_curr] = sum_val
                
                if self.verbose:
                    print(f"  β_{i}({self.states[y_curr]}):")
                    for detail in details:
                        print(f"    {detail}")
                    print(f"    = {sum_val:.6f}")
        
        if self.verbose:
            print("\n【步骤3：验证】")
            print("计算 Z(x) = Σ_y Ψ_0(start, y | x) * β_0(y)")
            Z_check = 0.0
            for y in range(self.n_states):
                potential = self.compute_potential(x, 0, -1, y)
                contribution = potential * beta[0, y]
                Z_check += contribution
                print(f"  Ψ_0(start, {self.states[y]}) × β_0({self.states[y]}) = {potential:.4f} × {beta[0, y]:.4f} = {contribution:.6f}")
            print(f"\nZ(x) = {Z_check:.6f}")
            print("=" * 70)
        
        return beta
    
    def compute_marginals(self, x: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算边缘概率
        
        P(y_i | x) = (α_i(y_i) * β_i(y_i)) / Z(x)
        
        参数:
            x: 观测序列
        
        返回:
            marginals: 边缘概率矩阵 [T, n_states]
            Z: 归一化因子
        """
        alpha, Z = self.forward(x)
        beta = self.backward(x)
        
        # 计算边缘概率
        marginals = (alpha * beta) / Z
        
        if self.verbose:
            print("\n" + "=" * 70)
            print("边缘概率 - Marginal Probabilities")
            print("=" * 70)
            print("计算 P(y_i | x) = (α_i(y_i) × β_i(y_i)) / Z(x)")
            print()
            
            T = len(x)
            for i in range(T):
                print(f"时刻 i={i}, 观测 x_{i} = '{x[i]}':")
                for y in range(self.n_states):
                    print(f"  P(y_{i}={self.states[y]} | x) = ({alpha[i,y]:.4f} × {beta[i,y]:.4f}) / {Z:.4f} = {marginals[i,y]:.6f}")
                print(f"  概率和: {np.sum(marginals[i, :]):.6f} (应该=1)")
                print()
            
            print("=" * 70)
        
        return marginals, Z
    
    def visualize(self, x: List[str], alpha: np.ndarray, beta: np.ndarray, 
                  marginals: np.ndarray, save_path: str = 'crf_forward_backward.png'):
        """
        可视化前向、后向和边缘概率
        
        参数:
            x: 观测序列
            alpha: 前向概率矩阵
            beta: 后向概率矩阵
            marginals: 边缘概率矩阵
            save_path: 保存路径
        """
        T = len(x)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('线性链CRF - 前向后向算法可视化', fontsize=16, fontweight='bold')
        
        # 1. 前向概率热力图
        ax1 = axes[0, 0]
        im1 = ax1.imshow(alpha.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax1.set_xlabel('时刻 t')
        ax1.set_ylabel('状态')
        ax1.set_title('前向概率 α_t(y)')
        ax1.set_xticks(range(T))
        ax1.set_xticklabels([f't={i}\n{x[i]}' for i in range(T)])
        ax1.set_yticks(range(self.n_states))
        ax1.set_yticklabels(self.states)
        plt.colorbar(im1, ax=ax1, label='概率值')
        
        # 添加数值标注
        for i in range(T):
            for j in range(self.n_states):
                text = ax1.text(i, j, f'{alpha[i, j]:.3f}',
                               ha="center", va="center", color="black", fontsize=9)
        
        # 2. 后向概率热力图
        ax2 = axes[0, 1]
        im2 = ax2.imshow(beta.T, aspect='auto', cmap='YlGnBu', interpolation='nearest')
        ax2.set_xlabel('时刻 t')
        ax2.set_ylabel('状态')
        ax2.set_title('后向概率 β_t(y)')
        ax2.set_xticks(range(T))
        ax2.set_xticklabels([f't={i}\n{x[i]}' for i in range(T)])
        ax2.set_yticks(range(self.n_states))
        ax2.set_yticklabels(self.states)
        plt.colorbar(im2, ax=ax2, label='概率值')
        
        # 添加数值标注
        for i in range(T):
            for j in range(self.n_states):
                text = ax2.text(i, j, f'{beta[i, j]:.3f}',
                               ha="center", va="center", color="black", fontsize=9)
        
        # 3. 边缘概率热力图
        ax3 = axes[1, 0]
        im3 = ax3.imshow(marginals.T, aspect='auto', cmap='Greens', interpolation='nearest')
        ax3.set_xlabel('时刻 t')
        ax3.set_ylabel('状态')
        ax3.set_title('边缘概率 P(y_t | x)')
        ax3.set_xticks(range(T))
        ax3.set_xticklabels([f't={i}\n{x[i]}' for i in range(T)])
        ax3.set_yticks(range(self.n_states))
        ax3.set_yticklabels(self.states)
        plt.colorbar(im3, ax=ax3, label='概率值')
        
        # 添加数值标注
        for i in range(T):
            for j in range(self.n_states):
                text = ax3.text(i, j, f'{marginals[i, j]:.3f}',
                               ha="center", va="center", color="black", fontsize=9)
        
        # 4. 概率曲线图
        ax4 = axes[1, 1]
        for j in range(self.n_states):
            ax4.plot(range(T), marginals[:, j], marker='o', label=f'状态 {self.states[j]}', linewidth=2)
        ax4.set_xlabel('时刻 t')
        ax4.set_ylabel('P(y_t | x)')
        ax4.set_title('边缘概率随时间变化')
        ax4.set_xticks(range(T))
        ax4.set_xticklabels([f'{x[i]}' for i in range(T)])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n可视化图已保存为: {save_path}")
        plt.close()


def example_simple_crf():
    """
    简单示例：使用预设权重的CRF
    """
    print("=" * 70)
    print("示例：线性链CRF前向后向算法")
    print("=" * 70)
    
    # 定义状态集合（例如：中文分词的BMES标注）
    states = ['B', 'M', 'E', 'S']  # Begin, Middle, End, Single
    
    # 定义特征数量（简化）
    n_features = 32
    features = [f'f_{i}' for i in range(n_features)]
    
    # 创建CRF模型
    crf = LinearChainCRF(states, features, verbose=True)
    
    # 设置随机权重（实际应用中需要通过训练获得）
    np.random.seed(42)
    weights = np.random.randn(n_features) * 0.5
    crf.set_weights(weights)
    
    # 观测序列（例如：一个短句子的字符）
    x = ['我', '爱', '北', '京', '天', '安', '门']
    
    print(f"\n观测序列: {' '.join(x)}")
    print(f"状态集合: {states}")
    print(f"特征数量: {n_features}")
    print(f"权重向量: 形状 {weights.shape}")
    
    # 计算前向概率
    alpha, Z = crf.forward(x)
    
    # 计算后向概率
    beta = crf.backward(x)
    
    # 验证：前向和后向计算的Z(x)应该相同
    Z_forward = Z
    Z_backward = np.sum([crf.compute_potential(x, 0, -1, y) * beta[0, y] for y in range(crf.n_states)])
    
    print("\n" + "=" * 70)
    print("验证结果")
    print("=" * 70)
    print(f"前向算法计算的 Z(x) = {Z_forward:.6f}")
    print(f"后向算法计算的 Z(x) = {Z_backward:.6f}")
    print(f"绝对误差: {abs(Z_forward - Z_backward):.10f}")
    if abs(Z_forward - Z_backward) < 1e-6:
        print("✓ 验证通过：前向和后向算法结果一致")
    else:
        print("✗ 警告：前向和后向算法结果不一致")
    
    # 计算边缘概率
    marginals, _ = crf.compute_marginals(x)
    
    # 可视化
    crf.visualize(x, alpha, beta, marginals)
    
    # 找出每个位置最可能的状态
    print("\n" + "=" * 70)
    print("最可能的状态序列（基于边缘概率）")
    print("=" * 70)
    for i in range(len(x)):
        best_state_idx = np.argmax(marginals[i, :])
        best_state = states[best_state_idx]
        best_prob = marginals[i, best_state_idx]
        print(f"位置 {i}: '{x[i]}' -> 状态 '{best_state}' (P={best_prob:.4f})")
    
    return crf, alpha, beta, marginals


def example_compare_with_uniform():
    """
    对比示例：不同权重设置的影响
    """
    print("\n\n" + "=" * 70)
    print("对比示例：均匀权重 vs 随机权重")
    print("=" * 70)
    
    states = ['B', 'I', 'O']  # 命名实体识别的BIO标注
    n_features = 20
    features = [f'f_{i}' for i in range(n_features)]
    x = ['张', '三', '在', '北', '京']
    
    # 情况1：均匀权重（所有权重为0）
    print("\n【情况1：均匀权重 (w=0)】")
    crf1 = LinearChainCRF(states, features, verbose=False)
    crf1.set_weights(np.zeros(n_features))
    
    alpha1, Z1 = crf1.forward(x)
    beta1 = crf1.backward(x)
    marginals1, _ = crf1.compute_marginals(x)
    
    print(f"Z(x) = {Z1:.6f}")
    print(f"边缘概率示例 (位置0):")
    for j, state in enumerate(states):
        print(f"  P(y_0={state} | x) = {marginals1[0, j]:.6f}")
    
    # 情况2：随机权重
    print("\n【情况2：随机权重】")
    crf2 = LinearChainCRF(states, features, verbose=False)
    np.random.seed(123)
    crf2.set_weights(np.random.randn(n_features))
    
    alpha2, Z2 = crf2.forward(x)
    beta2 = crf2.backward(x)
    marginals2, _ = crf2.compute_marginals(x)
    
    print(f"Z(x) = {Z2:.6f}")
    print(f"边缘概率示例 (位置0):")
    for j, state in enumerate(states):
        print(f"  P(y_0={state} | x) = {marginals2[0, j]:.6f}")
    
    print("\n观察：")
    print("- 均匀权重时，所有状态的边缘概率接近均匀分布")
    print("- 随机权重时，边缘概率会根据特征和权重产生偏好")


if __name__ == "__main__":
    # 运行示例
    example_simple_crf()
    example_compare_with_uniform()
    
    print("\n" + "=" * 70)
    print("前向后向算法演示完成！")
    print("=" * 70)
