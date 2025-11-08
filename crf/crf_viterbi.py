"""
线性链条件随机场 - Viterbi解码算法
Linear-chain Conditional Random Field - Viterbi Decoding

实现CRF的Viterbi算法，用于寻找最优标注序列：
y* = argmax_y P(y|x) = argmax_y (1/Z(x)) * exp(Σ w·f(y,x))

由于Z(x)与y无关，等价于：
y* = argmax_y Σ w·f(y,x)

参考：李航《统计学习方法》第11章
"""

import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False


class CRFViterbi:
    """
    线性链CRF的Viterbi解码器
    
    使用动态规划寻找最优标注序列
    """
    
    def __init__(self, states: List[str], features: List[str], verbose: bool = True):
        """
        初始化Viterbi解码器
        
        参数:
            states: 状态（标签）集合
            features: 特征名称列表
            verbose: 是否输出详细日志
        """
        self.states = states
        self.n_states = len(states)
        self.state_to_idx = {s: i for i, s in enumerate(states)}
        self.features = features
        self.n_features = len(features)
        self.verbose = verbose
        
        self.weights = None
    
    def set_weights(self, weights: np.ndarray):
        """设置特征权重"""
        self.weights = weights
    
    def compute_feature_vector(self, x: List[str], i: int, y_prev: int, y_curr: int) -> np.ndarray:
        """
        计算特征向量（与前向后向算法保持一致）
        """
        features = np.zeros(self.n_features)
        
        # 转移特征
        trans_idx = y_prev * self.n_states + y_curr
        if trans_idx < self.n_features // 2:
            features[trans_idx] = 1.0
        
        # 状态特征
        if i < len(x):
            obs_idx = (hash(x[i]) % (self.n_features // 2)) + self.n_features // 2
            if obs_idx < self.n_features:
                features[obs_idx] = 1.0
        
        return features
    
    def compute_score(self, x: List[str], i: int, y_prev: int, y_curr: int) -> float:
        """
        计算得分 score = w^T * f(y_{i-1}, y_i, x, i)
        
        注意：这里返回的是对数得分，不是概率
        """
        if self.weights is None:
            return 0.0
        
        features = self.compute_feature_vector(x, i, y_prev, y_curr)
        score = np.dot(self.weights, features)
        return score
    
    def viterbi(self, x: List[str]) -> Tuple[List[str], float, np.ndarray, np.ndarray]:
        """
        Viterbi算法：寻找最优标注序列
        
        δ_i(y) = max_{y_0,...,y_{i-1}} score(y_0,...,y_i | x)
        ψ_i(y) = argmax_{y_{i-1}} [δ_{i-1}(y_{i-1}) + score_i(y_{i-1}, y)]
        
        参数:
            x: 观测序列
        
        返回:
            best_path: 最优状态序列
            best_score: 最优路径得分
            delta: δ矩阵
            psi: ψ矩阵
        """
        T = len(x)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        
        if self.verbose:
            print("\n" + "=" * 70)
            print("Viterbi算法 - 最优标注序列解码")
            print("=" * 70)
            print(f"观测序列: {x}")
            print(f"序列长度 T = {T}")
            print(f"状态集合: {self.states}")
        
        # 步骤1：初始化 (i=0)
        if self.verbose:
            print("\n【步骤1：初始化】")
            print("计算 δ_0(y) = score(start, y | x)")
        
        for y in range(self.n_states):
            delta[0, y] = self.compute_score(x, 0, -1, y)
            psi[0, y] = -1  # 起始状态
            
            if self.verbose:
                print(f"  δ_0({self.states[y]}) = {delta[0, y]:.4f}")
        
        # 步骤2：递推 (i=1 to T-1)
        if self.verbose:
            print("\n【步骤2：递推】")
            print("计算 δ_i(y) = max_{y'} [δ_{i-1}(y') + score_i(y', y | x)]")
        
        for i in range(1, T):
            if self.verbose:
                print(f"\n--- 时刻 i={i}, 观测 x_{i} = '{x[i]}' ---")
            
            for y_curr in range(self.n_states):
                max_score = float('-inf')
                best_prev = -1
                scores_detail = []
                
                for y_prev in range(self.n_states):
                    trans_score = self.compute_score(x, i, y_prev, y_curr)
                    total_score = delta[i-1, y_prev] + trans_score
                    scores_detail.append(f"δ_{i-1}({self.states[y_prev]})+score={delta[i-1, y_prev]:.4f}+{trans_score:.4f}={total_score:.4f}")
                    
                    if total_score > max_score:
                        max_score = total_score
                        best_prev = y_prev
                
                delta[i, y_curr] = max_score
                psi[i, y_curr] = best_prev
                
                if self.verbose:
                    print(f"  δ_{i}({self.states[y_curr]}):")
                    for detail in scores_detail:
                        print(f"    {detail}")
                    print(f"    max = {max_score:.4f}, 最优前驱 = {self.states[best_prev]}")
        
        # 步骤3：终止
        if self.verbose:
            print("\n【步骤3：终止】")
            print("找到最优终点状态: y_T* = argmax_y δ_T(y)")
        
        best_last_state = np.argmax(delta[-1, :])
        best_score = delta[-1, best_last_state]
        
        if self.verbose:
            for y in range(self.n_states):
                print(f"  δ_{T-1}({self.states[y]}) = {delta[-1, y]:.4f}")
            print(f"\n最优终点状态: {self.states[best_last_state]}")
            print(f"最优路径得分: {best_score:.4f}")
        
        # 步骤4：路径回溯
        if self.verbose:
            print("\n【步骤4：路径回溯】")
            print("沿着 ψ 矩阵回溯最优路径")
        
        best_path = [0] * T
        best_path[-1] = best_last_state
        
        if self.verbose:
            print(f"  y_{T-1}* = {self.states[best_path[-1]]}")
        
        for i in range(T-2, -1, -1):
            best_path[i] = psi[i+1, best_path[i+1]]
            if self.verbose:
                print(f"  y_{i}* = ψ_{i+1}(y_{i+1}*) = ψ_{i+1}({self.states[best_path[i+1]]}) = {self.states[best_path[i]]}")
        
        # 转换为状态标签
        best_path_labels = [self.states[idx] for idx in best_path]
        
        if self.verbose:
            print("\n" + "=" * 70)
            print("【结果】")
            print("=" * 70)
            print(f"观测序列: {' '.join(x)}")
            print(f"最优标注: {' '.join(best_path_labels)}")
            print(f"路径得分: {best_score:.4f}")
            print("=" * 70)
        
        return best_path_labels, best_score, delta, psi
    
    def visualize(self, x: List[str], delta: np.ndarray, psi: np.ndarray, 
                  best_path: List[str], save_path: str = 'crf_viterbi.png'):
        """
        可视化Viterbi解码过程
        """
        T = len(x)
        best_path_indices = [self.state_to_idx[s] for s in best_path]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('线性链CRF - Viterbi解码算法可视化', fontsize=16, fontweight='bold')
        
        # 1. δ矩阵热力图
        ax1 = axes[0, 0]
        im1 = ax1.imshow(delta.T, aspect='auto', cmap='RdYlGn', interpolation='nearest')
        ax1.set_xlabel('时刻 t')
        ax1.set_ylabel('状态')
        ax1.set_title('δ矩阵：局部最优得分')
        ax1.set_xticks(range(T))
        ax1.set_xticklabels([f't={i}\n{x[i]}' for i in range(T)])
        ax1.set_yticks(range(self.n_states))
        ax1.set_yticklabels(self.states)
        plt.colorbar(im1, ax=ax1, label='得分值')
        
        # 标注数值和最优路径
        for i in range(T):
            for j in range(self.n_states):
                color = 'white' if j == best_path_indices[i] else 'black'
                weight = 'bold' if j == best_path_indices[i] else 'normal'
                ax1.text(i, j, f'{delta[i, j]:.2f}',
                        ha="center", va="center", color=color, fontweight=weight, fontsize=9)
        
        # 2. ψ矩阵（最优前驱）
        ax2 = axes[0, 1]
        im2 = ax2.imshow(psi.T, aspect='auto', cmap='tab10', interpolation='nearest')
        ax2.set_xlabel('时刻 t')
        ax2.set_ylabel('状态')
        ax2.set_title('ψ矩阵：最优前驱状态')
        ax2.set_xticks(range(T))
        ax2.set_xticklabels([f't={i}\n{x[i]}' for i in range(T)])
        ax2.set_yticks(range(self.n_states))
        ax2.set_yticklabels(self.states)
        plt.colorbar(im2, ax=ax2, label='前驱状态索引')
        
        # 标注前驱状态
        for i in range(T):
            for j in range(self.n_states):
                if psi[i, j] >= 0:
                    ax2.text(i, j, self.states[psi[i, j]],
                            ha="center", va="center", color="white", fontweight='bold', fontsize=10)
        
        # 3. 最优路径可视化
        ax3 = axes[1, 0]
        ax3.set_xlim(-0.5, T-0.5)
        ax3.set_ylim(-0.5, self.n_states-0.5)
        ax3.set_xlabel('时刻 t')
        ax3.set_ylabel('状态')
        ax3.set_title('最优标注路径')
        ax3.set_xticks(range(T))
        ax3.set_xticklabels([f'{x[i]}' for i in range(T)])
        ax3.set_yticks(range(self.n_states))
        ax3.set_yticklabels(self.states)
        ax3.grid(True, alpha=0.3)
        
        # 绘制所有可能的转移（淡化）
        for i in range(T-1):
            for j in range(self.n_states):
                for k in range(self.n_states):
                    ax3.plot([i, i+1], [j, k], 'gray', alpha=0.1, linewidth=0.5)
        
        # 绘制最优路径
        for i in range(T):
            ax3.plot(i, best_path_indices[i], 'ro', markersize=15, zorder=3)
            ax3.text(i, best_path_indices[i], best_path[i],
                    ha='center', va='center', color='white', fontweight='bold', fontsize=10, zorder=4)
        
        for i in range(T-1):
            ax3.annotate('', xy=(i+1, best_path_indices[i+1]), xytext=(i, best_path_indices[i]),
                        arrowprops=dict(arrowstyle='->', color='red', lw=3), zorder=2)
        
        # 4. δ值曲线
        ax4 = axes[1, 1]
        for j in range(self.n_states):
            style = '-o' if j in best_path_indices else '--'
            linewidth = 2.5 if j in best_path_indices else 1.0
            alpha = 1.0 if j in best_path_indices else 0.5
            ax4.plot(range(T), delta[:, j], style, label=f'状态 {self.states[j]}',
                    linewidth=linewidth, alpha=alpha)
        
        # 标注最优路径
        for i in range(T):
            ax4.plot(i, delta[i, best_path_indices[i]], 'r*', markersize=15, zorder=3)
        
        ax4.set_xlabel('时刻 t')
        ax4.set_ylabel('δ_t(y)')
        ax4.set_title('δ值随时间变化（★标记最优路径）')
        ax4.set_xticks(range(T))
        ax4.set_xticklabels([f'{x[i]}' for i in range(T)])
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n可视化图已保存为: {save_path}")
        plt.close()


def example_crf_viterbi():
    """
    示例：CRF Viterbi解码
    """
    print("=" * 70)
    print("示例：线性链CRF Viterbi解码算法")
    print("=" * 70)
    
    # 定义状态集合（中文分词BMES标注）
    states = ['B', 'M', 'E', 'S']
    
    # 定义特征
    n_features = 32
    features = [f'f_{i}' for i in range(n_features)]
    
    # 创建Viterbi解码器
    decoder = CRFViterbi(states, features, verbose=True)
    
    # 设置权重（模拟训练后的结果）
    np.random.seed(42)
    weights = np.random.randn(n_features) * 0.8
    decoder.set_weights(weights)
    
    # 观测序列
    x = ['我', '爱', '北', '京', '天', '安', '门']
    
    print(f"\n任务：中文分词")
    print(f"观测序列: {' '.join(x)}")
    print(f"标注方案: BMES (Begin, Middle, End, Single)")
    print(f"  B: 词首  M: 词中  E: 词尾  S: 单字词")
    
    # 执行Viterbi解码
    best_path, best_score, delta, psi = decoder.viterbi(x)
    
    # 根据标注结果进行分词
    print("\n" + "=" * 70)
    print("分词结果解释")
    print("=" * 70)
    words = []
    current_word = ""
    for i, (char, tag) in enumerate(zip(x, best_path)):
        if tag == 'B':
            if current_word:
                words.append(current_word)
            current_word = char
        elif tag == 'M':
            current_word += char
        elif tag == 'E':
            current_word += char
            words.append(current_word)
            current_word = ""
        elif tag == 'S':
            if current_word:
                words.append(current_word)
                current_word = ""
            words.append(char)
    
    if current_word:
        words.append(current_word)
    
    print(f"原始文本: {''.join(x)}")
    print(f"标注序列: {' '.join(best_path)}")
    print(f"分词结果: {' / '.join(words)}")
    
    # 可视化
    decoder.visualize(x, delta, psi, best_path)
    
    return decoder, best_path, delta, psi


def example_ner_tagging():
    """
    示例：命名实体识别（NER）
    """
    print("\n\n" + "=" * 70)
    print("示例：命名实体识别（NER）")
    print("=" * 70)
    
    # BIO标注方案
    states = ['B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'O']
    
    n_features = 64
    features = [f'f_{i}' for i in range(n_features)]
    
    decoder = CRFViterbi(states, features, verbose=False)
    
    # 设置权重（模拟）
    np.random.seed(456)
    weights = np.random.randn(n_features) * 0.5
    
    # 添加一些偏置，让某些转移更可能
    # 例如：B-PER后更可能接I-PER
    decoder.set_weights(weights)
    
    # 观测序列
    x = ['张', '三', '在', '北', '京', '工', '作']
    
    print(f"\n任务：命名实体识别")
    print(f"观测序列: {' '.join(x)}")
    print(f"标注方案: BIO")
    print(f"  B-PER: 人名开始  I-PER: 人名延续")
    print(f"  B-LOC: 地名开始  I-LOC: 地名延续")
    print(f"  B-ORG: 机构名开始  I-ORG: 机构名延续")
    print(f"  O: 其他")
    
    # 执行解码
    best_path, best_score, delta, psi = decoder.viterbi(x)
    
    print("\n" + "=" * 70)
    print("标注结果")
    print("=" * 70)
    for char, tag in zip(x, best_path):
        print(f"  '{char}' -> {tag}")
    
    # 提取实体
    print("\n提取的实体:")
    entities = []
    current_entity = ""
    current_type = ""
    
    for char, tag in zip(x, best_path):
        if tag.startswith('B-'):
            if current_entity:
                entities.append((current_entity, current_type))
            current_entity = char
            current_type = tag[2:]
        elif tag.startswith('I-'):
            current_entity += char
        else:  # 'O'
            if current_entity:
                entities.append((current_entity, current_type))
                current_entity = ""
                current_type = ""
    
    if current_entity:
        entities.append((current_entity, current_type))
    
    for entity, ent_type in entities:
        print(f"  [{entity}] - {ent_type}")
    
    # 可视化
    decoder.visualize(x, delta, psi, best_path, save_path='crf_viterbi_ner.png')


def example_compare_paths():
    """
    对比不同权重下的解码结果
    """
    print("\n\n" + "=" * 70)
    print("对比示例：不同权重的解码结果")
    print("=" * 70)
    
    states = ['B', 'I', 'O']
    n_features = 20
    features = [f'f_{i}' for i in range(n_features)]
    x = ['张', '三', '爱', '中', '国']
    
    # 情况1：均匀权重
    print("\n【情况1：均匀权重】")
    decoder1 = CRFViterbi(states, features, verbose=False)
    decoder1.set_weights(np.zeros(n_features))
    path1, score1, _, _ = decoder1.viterbi(x)
    print(f"最优标注: {' '.join(path1)}")
    print(f"路径得分: {score1:.4f}")
    
    # 情况2：随机权重
    print("\n【情况2：随机权重A】")
    decoder2 = CRFViterbi(states, features, verbose=False)
    np.random.seed(111)
    decoder2.set_weights(np.random.randn(n_features))
    path2, score2, _, _ = decoder2.viterbi(x)
    print(f"最优标注: {' '.join(path2)}")
    print(f"路径得分: {score2:.4f}")
    
    # 情况3：另一组随机权重
    print("\n【情况3：随机权重B】")
    decoder3 = CRFViterbi(states, features, verbose=False)
    np.random.seed(222)
    decoder3.set_weights(np.random.randn(n_features) * 2)
    path3, score3, _, _ = decoder3.viterbi(x)
    print(f"最优标注: {' '.join(path3)}")
    print(f"路径得分: {score3:.4f}")
    
    print("\n观察：")
    print("- 权重不同会导致不同的最优标注序列")
    print("- 这就是为什么需要通过训练数据学习合适的权重")


if __name__ == "__main__":
    # 运行示例
    example_crf_viterbi()
    example_ner_tagging()
    example_compare_paths()
    
    print("\n" + "=" * 70)
    print("Viterbi解码算法演示完成！")
    print("=" * 70)
