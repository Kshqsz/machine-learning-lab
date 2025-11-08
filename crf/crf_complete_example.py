"""
线性链条件随机场 - 完整示例
Linear-chain Conditional Random Field - Complete Example

综合演示CRF的完整流程：
1. 前向-后向算法（推断）
2. Viterbi解码（预测）
3. BFGS训练（学习）

应用场景：中文分词、命名实体识别、词性标注

参考：李航《统计学习方法》第11章
"""

import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# 导入我们实现的CRF模块
import sys
import os
sys.path.append(os.path.dirname(__file__))

# 注意：在实际运行时，这些import会使用同目录下的文件
# 这里为了演示完整性，我们重新实现一个简化的完整CRF类


class CompleteCRF:
    """
    完整的线性链CRF实现
    
    整合了前向-后向算法、Viterbi解码和BFGS训练
    """
    
    def __init__(self, states: List[str], verbose: bool = True):
        self.states = states
        self.n_states = len(states)
        self.state_to_idx = {s: i for i, s in enumerate(states)}
        self.verbose = verbose
        
        self.feature_to_idx = {}
        self.n_features = 0
        self.weights = None
        
        self._init_features()
    
    def _init_features(self):
        """初始化特征"""
        feature_id = 0
        
        # 转移特征
        for i in range(self.n_states):
            for j in range(self.n_states):
                self.feature_to_idx[f'trans_{i}_{j}'] = feature_id
                feature_id += 1
        
        # 状态特征
        for i in range(self.n_states):
            self.feature_to_idx[f'state_{i}'] = feature_id
            feature_id += 1
        
        self.n_features = feature_id
        self.weights = np.zeros(self.n_features)
    
    def _get_features(self, x: List[str], i: int, y_prev: int, y_curr: int) -> np.ndarray:
        """提取特征"""
        features = np.zeros(self.n_features)
        
        if y_prev >= 0:
            features[self.feature_to_idx[f'trans_{y_prev}_{y_curr}']] = 1.0
        
        features[self.feature_to_idx[f'state_{y_curr}']] = 1.0
        
        return features
    
    def _get_score(self, x: List[str], i: int, y_prev: int, y_curr: int) -> float:
        """计算得分"""
        features = self._get_features(x, i, y_prev, y_curr)
        return np.dot(self.weights, features)
    
    def forward_backward(self, x: List[str]) -> Tuple[np.ndarray, np.ndarray, float]:
        """前向-后向算法"""
        T = len(x)
        
        # 前向
        log_alpha = np.full((T, self.n_states), -np.inf)
        for y in range(self.n_states):
            log_alpha[0, y] = self._get_score(x, 0, -1, y)
        
        for i in range(1, T):
            for y_curr in range(self.n_states):
                log_sum = -np.inf
                for y_prev in range(self.n_states):
                    score = self._get_score(x, i, y_prev, y_curr)
                    log_val = log_alpha[i-1, y_prev] + score
                    log_sum = np.logaddexp(log_sum, log_val)
                log_alpha[i, y_curr] = log_sum
        
        log_Z = -np.inf
        for y in range(self.n_states):
            log_Z = np.logaddexp(log_Z, log_alpha[-1, y])
        
        # 后向
        log_beta = np.zeros((T, self.n_states))
        for i in range(T-2, -1, -1):
            for y_curr in range(self.n_states):
                log_sum = -np.inf
                for y_next in range(self.n_states):
                    score = self._get_score(x, i+1, y_curr, y_next)
                    log_val = score + log_beta[i+1, y_next]
                    log_sum = np.logaddexp(log_sum, log_val)
                log_beta[i, y_curr] = log_sum
        
        # 边缘概率
        marginals = np.exp(log_alpha + log_beta - log_Z)
        
        return log_alpha, log_beta, marginals
    
    def viterbi(self, x: List[str]) -> Tuple[List[str], float]:
        """Viterbi解码"""
        T = len(x)
        delta = np.full((T, self.n_states), -np.inf)
        psi = np.zeros((T, self.n_states), dtype=int)
        
        # 初始化
        for y in range(self.n_states):
            delta[0, y] = self._get_score(x, 0, -1, y)
        
        # 递推
        for i in range(1, T):
            for y_curr in range(self.n_states):
                for y_prev in range(self.n_states):
                    score = self._get_score(x, i, y_prev, y_curr)
                    total = delta[i-1, y_prev] + score
                    if total > delta[i, y_curr]:
                        delta[i, y_curr] = total
                        psi[i, y_curr] = y_prev
        
        # 回溯
        path = [0] * T
        path[-1] = np.argmax(delta[-1, :])
        best_score = delta[-1, path[-1]]
        
        for i in range(T-2, -1, -1):
            path[i] = psi[i+1, path[i+1]]
        
        path_labels = [self.states[idx] for idx in path]
        return path_labels, best_score
    
    def train(self, X_train: List[List[str]], Y_train: List[List[str]], 
             lambda_reg: float = 0.1, max_iter: int = 50):
        """BFGS训练"""
        from scipy.optimize import minimize
        
        # 计算经验计数
        empirical = np.zeros(self.n_features)
        for x, y_labels in zip(X_train, Y_train):
            # 确保x和y长度一致
            if len(x) != len(y_labels):
                continue
            y = [self.state_to_idx[label] for label in y_labels]
            for i in range(len(x)):
                y_prev = y[i-1] if i > 0 else -1
                features = self._get_features(x, i, y_prev, y[i])
                empirical += features
        
        history = {'loss': [], 'grad_norm': []}
        
        def objective(w):
            self.weights = w
            
            # 计算log Z和模型期望
            log_Z_sum = 0.0
            model_expected = np.zeros(self.n_features)
            
            for x in X_train:
                T = len(x)
                log_alpha, log_beta, _ = self.forward_backward(x)
                
                # log Z
                log_Z = -np.inf
                for y in range(self.n_states):
                    log_Z = np.logaddexp(log_Z, log_alpha[-1, y])
                log_Z_sum += log_Z
                
                # 模型期望
                for i in range(T):
                    for y_curr in range(self.n_states):
                        for y_prev in range(-1 if i == 0 else 0, self.n_states):
                            if i == 0 and y_prev != -1:
                                continue
                            
                            if i == 0:
                                log_prob = log_alpha[0, y_curr] + log_beta[0, y_curr] - log_Z
                            else:
                                score = self._get_score(x, i, y_prev, y_curr)
                                log_prob = log_alpha[i-1, y_prev] + score + log_beta[i, y_curr] - log_Z
                            
                            prob = np.exp(log_prob)
                            features = self._get_features(x, i, y_prev, y_curr)
                            model_expected += prob * features
            
            # 损失和梯度
            loss = -np.dot(w, empirical) + log_Z_sum + 0.5 * lambda_reg * np.dot(w, w)
            grad = -empirical + model_expected + lambda_reg * w
            
            history['loss'].append(loss)
            history['grad_norm'].append(np.linalg.norm(grad))
            
            return loss, grad
        
        if self.verbose:
            print("开始BFGS训练...")
        
        result = minimize(objective, self.weights, method='L-BFGS-B', jac=True,
                         options={'maxiter': max_iter, 'disp': self.verbose})
        
        self.weights = result.x
        
        if self.verbose:
            print(f"训练完成！最终损失: {result.fun:.4f}")
        
        return history


def chinese_word_segmentation_example():
    """
    应用示例1：中文分词
    """
    print("\n" + "=" * 80)
    print("应用示例1：中文分词 (Chinese Word Segmentation)")
    print("=" * 80)
    
    states = ['B', 'M', 'E', 'S']  # Begin, Middle, End, Single
    crf = CompleteCRF(states, verbose=True)
    
    # 训练数据
    X_train = [
        list('我爱中国'),
        list('北京欢迎你'),
        list('机器学习'),
        list('自然语言处理'),
        list('深度学习算法'),
        list('人工智能时代'),
        list('数据科学'),
        list('计算机视觉')
    ]
    
    Y_train = [
        ['S', 'S', 'B', 'E'],  # 我/爱/中国
        ['B', 'E', 'S', 'B', 'E'],  # 北京/欢/迎你
        ['B', 'M', 'M', 'E'],  # 机器学习
        ['S', 'B', 'M', 'M', 'E'],  # 自/然语言处理
        ['B', 'M', 'M', 'M', 'E'],  # 深度学习算法
        ['B', 'M', 'M', 'M', 'M', 'E'],  # 人工智能时代
        ['B', 'M', 'M', 'E'],  # 数据科学
        ['B', 'M', 'M', 'M', 'E']  # 计算机视觉
    ]
    
    print("\n训练数据:")
    for x, y in zip(X_train[:5], Y_train[:5]):
        print(f"  {''.join(x):12s} -> {' '.join(y)}")
    print(f"  ... 共 {len(X_train)} 个句子")
    
    # 训练模型
    print("\n" + "-" * 80)
    history = crf.train(X_train, Y_train, lambda_reg=0.1, max_iter=30)
    
    # 测试
    print("\n" + "-" * 80)
    print("测试预测")
    print("-" * 80)
    
    test_sentences = [
        '我爱北京',
        '机器学习很有趣',
        '人工智能',
        '自然语言',
        '深度神经网络'
    ]
    
    for sentence in test_sentences:
        x = list(sentence)
        y_pred, score = crf.viterbi(x)
        
        # 分词
        words = []
        current_word = ""
        for char, tag in zip(x, y_pred):
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
        
        print(f"\n原句: {sentence}")
        print(f"标注: {' '.join(y_pred)}")
        print(f"分词: {' / '.join(words)}")
        print(f"得分: {score:.4f}")
    
    # 可视化训练过程
    visualize_training_history(history, "中文分词", 'crf_chinese_segmentation.png')
    
    return crf, history


def ner_example():
    """
    应用示例2：命名实体识别
    """
    print("\n\n" + "=" * 80)
    print("应用示例2：命名实体识别 (Named Entity Recognition)")
    print("=" * 80)
    
    states = ['B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'O']  # Person, Location, Other
    crf = CompleteCRF(states, verbose=False)
    
    # 训练数据
    X_train = [
        list('张三在北京工作'),
        list('李四去了上海'),
        list('王五住在广州'),
        list('赵六来自杭州'),
        list('孙七在深圳上班')
    ]
    
    Y_train = [
        ['B-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'O', 'O'],
        ['B-PER', 'I-PER', 'O', 'O', 'B-LOC', 'I-LOC'],
        ['B-PER', 'I-PER', 'O', 'O', 'B-LOC', 'I-LOC'],
        ['B-PER', 'I-PER', 'O', 'O', 'B-LOC', 'I-LOC'],
        ['B-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'O', 'O']
    ]
    
    print("\n训练数据:")
    for x, y in zip(X_train, Y_train):
        print(f"  {''.join(x):15s} -> {' '.join(y)}")
    
    # 训练
    print("\n训练中...")
    history = crf.train(X_train, Y_train, lambda_reg=0.05, max_iter=30)
    print("训练完成！")
    
    # 测试
    print("\n" + "-" * 80)
    print("测试预测")
    print("-" * 80)
    
    test_sentences = [
        '钱八在成都',
        '周九去南京',
        '吴十住重庆'
    ]
    
    for sentence in test_sentences:
        x = list(sentence)
        y_pred, score = crf.viterbi(x)
        
        # 提取实体
        entities = {'PER': [], 'LOC': []}
        current_entity = ""
        current_type = None
        
        for char, tag in zip(x, y_pred):
            if tag.startswith('B-'):
                if current_entity and current_type:
                    entities[current_type].append(current_entity)
                current_entity = char
                current_type = tag[2:]
            elif tag.startswith('I-'):
                current_entity += char
            else:
                if current_entity and current_type:
                    entities[current_type].append(current_entity)
                current_entity = ""
                current_type = None
        
        if current_entity and current_type:
            entities[current_type].append(current_entity)
        
        print(f"\n原句: {sentence}")
        print(f"标注: {' '.join(y_pred)}")
        print(f"人名: {', '.join(entities['PER']) if entities['PER'] else '无'}")
        print(f"地名: {', '.join(entities['LOC']) if entities['LOC'] else '无'}")
    
    return crf, history


def pos_tagging_example():
    """
    应用示例3：词性标注
    """
    print("\n\n" + "=" * 80)
    print("应用示例3：词性标注 (Part-of-Speech Tagging)")
    print("=" * 80)
    
    states = ['n', 'v', 'a', 'd', 'p']  # 名词、动词、形容词、副词、介词
    crf = CompleteCRF(states, verbose=False)
    
    # 训练数据（简化）
    X_train = [
        ['我', '爱', '中', '国'],
        ['他', '很', '高', '兴'],
        ['在', '学', '校', '里']
    ]
    
    Y_train = [
        ['n', 'v', 'n', 'n'],  # 我/n 爱/v 中/n 国/n
        ['n', 'd', 'a', 'a'],  # 他/n 很/d 高/a 兴/a
        ['p', 'n', 'n', 'n']   # 在/p 学/n 校/n 里/n
    ]
    
    print("\n训练数据:")
    for x, y in zip(X_train, Y_train):
        print(f"  {'/'.join([f'{c}({t})' for c, t in zip(x, y)])}")
    
    # 训练
    print("\n训练中...")
    crf.train(X_train, Y_train, lambda_reg=0.1, max_iter=20)
    print("训练完成！")
    
    # 测试
    print("\n测试:")
    test = ['我', '在', '北', '京']
    y_pred, _ = crf.viterbi(test)
    print(f"  {'/'.join([f'{c}({t})' for c, t in zip(test, y_pred)])}")
    
    return crf


def visualize_training_history(history, title, filename):
    """可视化训练历史"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'CRF训练过程 - {title}', fontsize=16, fontweight='bold')
    
    iterations = list(range(len(history['loss'])))
    
    # 损失曲线
    ax1 = axes[0]
    ax1.plot(iterations, history['loss'], 'b-', linewidth=2, label='训练损失')
    ax1.set_xlabel('迭代次数')
    ax1.set_ylabel('负对数似然')
    ax1.set_title('损失函数收敛曲线')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 梯度范数
    ax2 = axes[1]
    ax2.semilogy(iterations, history['grad_norm'], 'r-', linewidth=2, label='梯度范数')
    ax2.set_xlabel('迭代次数')
    ax2.set_ylabel('梯度范数 (log scale)')
    ax2.set_title('梯度范数变化')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n可视化已保存: {filename}")
    plt.close()


def compare_with_hmm():
    """
    对比CRF与HMM的区别
    """
    print("\n\n" + "=" * 80)
    print("CRF vs HMM 对比分析")
    print("=" * 80)
    
    print("""
CRF（条件随机场）与 HMM（隐马尔可夫模型）的主要区别：

1. 模型类型：
   - HMM: 生成式模型，建模 P(X,Y)
   - CRF: 判别式模型，直接建模 P(Y|X)

2. 条件独立性假设：
   - HMM: 假设观测独立（给定当前状态，观测独立于其他观测）
   - CRF: 无需观测独立性假设，可以使用任意特征

3. 特征函数：
   - HMM: 使用固定的状态转移概率和发射概率
   - CRF: 灵活定义特征函数，可以包含任意观测特征

4. 参数学习：
   - HMM: EM算法（Baum-Welch）
   - CRF: 梯度下降/BFGS等优化算法

5. 优势：
   - HMM: 可以生成观测序列，计算观测概率
   - CRF: 标注准确率通常更高，特征工程更灵活

6. 应用场景：
   - HMM: 语音识别、时序数据生成
   - CRF: 序列标注（分词、NER、词性标注）

数学形式对比：

HMM:
  P(Y,X) = P(y_1) ∏ P(y_t|y_{t-1}) ∏ P(x_t|y_t)

CRF:
  P(Y|X) = (1/Z(X)) exp(Σ w·f(y_{t-1}, y_t, X, t))

其中CRF的特征函数f可以任意定义，灵活性更高。
    """)


def algorithm_summary():
    """
    算法总结
    """
    print("\n\n" + "=" * 80)
    print("CRF算法总结")
    print("=" * 80)
    
    print("""
线性链条件随机场的三个基本算法：

1. 前向-后向算法 (Forward-Backward Algorithm)
   目的：计算归一化因子Z(x)和边缘概率P(y_t|x)
   用途：推断、概率计算、梯度计算
   时间复杂度：O(T·N²)
   
   核心：
   - 前向：α_t(y) = Σ_{y'} α_{t-1}(y') · exp(score)
   - 后向：β_t(y) = Σ_{y'} exp(score) · β_{t+1}(y')
   - Z(x) = Σ_y α_T(y)

2. Viterbi解码算法 (Viterbi Decoding)
   目的：寻找最优标注序列 y* = argmax P(y|x)
   用途：序列标注、预测
   时间复杂度：O(T·N²)
   
   核心：
   - δ_t(y) = max_{y'} [δ_{t-1}(y') + score(y', y)]
   - 回溯找到最优路径

3. BFGS训练算法 (BFGS Training)
   目的：学习特征权重w
   用途：模型训练、参数估计
   时间复杂度：O(迭代次数 · 样本数 · T · N²)
   
   核心：
   - 目标函数：L(w) = -Σ w·f(x,y) + Σ log Z(x) + λ||w||²
   - 梯度：∇L = -E_data[f] + E_model[f] + λw
   - BFGS优化

特点：
- 全局归一化（相比HMM的局部归一化）
- 判别式模型（直接优化P(Y|X)）
- 特征灵活（可以使用任意观测特征）
- 训练复杂度高但预测准确率好

适用场景：
✓ 中文分词
✓ 命名实体识别
✓ 词性标注
✓ 任何序列标注任务
    """)


if __name__ == "__main__":
    print("=" * 80)
    print("条件随机场 (CRF) 完整示例")
    print("=" * 80)
    print("\n本示例展示CRF的三个核心算法及其应用：")
    print("  1. 前向-后向算法")
    print("  2. Viterbi解码算法")
    print("  3. BFGS训练算法")
    
    # 运行示例
    chinese_word_segmentation_example()
    ner_example()
    pos_tagging_example()
    
    # 对比和总结
    compare_with_hmm()
    algorithm_summary()
    
    print("\n" + "=" * 80)
    print("所有示例运行完成！")
    print("=" * 80)
    print("\n生成的可视化文件：")
    print("  - crf_chinese_segmentation.png: 中文分词训练过程")
    print("\n查看其他算法的详细实现：")
    print("  - crf_forward_backward.py: 前向-后向算法")
    print("  - crf_viterbi.py: Viterbi解码算法")
    print("  - crf_train.py: BFGS训练算法")
