"""
线性链条件随机场 - BFGS训练算法
Linear-chain Conditional Random Field - BFGS Training

实现使用BFGS拟牛顿法训练CRF模型：
1. 计算对数似然函数
2. 计算梯度
3. 使用BFGS优化权重

参考：李航《统计学习方法》第11章
"""

import numpy as np
from typing import List, Tuple, Dict
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False


class CRFTrainer:
    """
    线性链CRF训练器
    
    使用BFGS拟牛顿法训练模型参数
    """
    
    def __init__(self, states: List[str], verbose: bool = True):
        """
        初始化训练器
        
        参数:
            states: 状态（标签）集合
            verbose: 是否输出详细日志
        """
        self.states = states
        self.n_states = len(states)
        self.state_to_idx = {s: i for i, s in enumerate(states)}
        self.verbose = verbose
        
        self.feature_templates = []
        self.feature_to_idx = {}
        self.n_features = 0
        
        self.weights = None
        self.training_history = {
            'iteration': [],
            'loss': [],
            'gradient_norm': []
        }
    
    def create_feature_templates(self):
        """
        创建特征模板
        
        包括：
        1. 转移特征：t(y_{i-1}, y_i, x, i)
        2. 状态特征：s(y_i, x, i)
        """
        feature_id = 0
        
        # 转移特征：所有状态对的转移
        for y_prev in range(self.n_states):
            for y_curr in range(self.n_states):
                feature_name = f"trans_{self.states[y_prev]}_{self.states[y_curr]}"
                self.feature_to_idx[feature_name] = feature_id
                self.feature_templates.append({
                    'type': 'transition',
                    'y_prev': y_prev,
                    'y_curr': y_curr,
                    'name': feature_name
                })
                feature_id += 1
        
        # 状态特征：当前状态
        for y in range(self.n_states):
            feature_name = f"state_{self.states[y]}"
            self.feature_to_idx[feature_name] = feature_id
            self.feature_templates.append({
                'type': 'state',
                'y': y,
                'name': feature_name
            })
            feature_id += 1
        
        # 状态-观测特征（简化版本，实际应用中会更复杂）
        # 这里我们为每个状态和观测的组合创建特征
        self.n_features = feature_id
        
        if self.verbose:
            print(f"创建了 {self.n_features} 个特征模板")
            print(f"  - 转移特征: {self.n_states * self.n_states} 个")
            print(f"  - 状态特征: {self.n_states} 个")
    
    def extract_features(self, x: List[str], y: List[int], i: int) -> np.ndarray:
        """
        从位置i提取特征
        
        参数:
            x: 观测序列
            y: 标注序列（状态索引）
            i: 当前位置
        
        返回:
            特征向量
        """
        features = np.zeros(self.n_features)
        
        # 转移特征
        y_prev = y[i-1] if i > 0 else -1
        y_curr = y[i]
        
        if y_prev >= 0:
            feature_name = f"trans_{self.states[y_prev]}_{self.states[y_curr]}"
            if feature_name in self.feature_to_idx:
                features[self.feature_to_idx[feature_name]] = 1.0
        
        # 状态特征
        feature_name = f"state_{self.states[y_curr]}"
        if feature_name in self.feature_to_idx:
            features[self.feature_to_idx[feature_name]] = 1.0
        
        return features
    
    def compute_potential_score(self, w: np.ndarray, x: List[str], i: int, 
                                y_prev: int, y_curr: int) -> float:
        """
        计算势函数的得分（对数空间）
        
        score = w^T * f(y_{i-1}, y_i, x, i)
        """
        # 构造临时的y序列来提取特征
        y = [0] * len(x)
        if y_prev >= 0:
            y[i-1] = y_prev
        y[i] = y_curr
        
        features = self.extract_features(x, y, i)
        score = np.dot(w, features)
        return score
    
    def forward_algorithm(self, w: np.ndarray, x: List[str]) -> Tuple[np.ndarray, float]:
        """
        前向算法（对数空间）
        
        返回:
            alpha: 前向概率矩阵（对数）
            log_Z: log Z(x)
        """
        T = len(x)
        log_alpha = np.full((T, self.n_states), -np.inf)
        
        # 初始化
        for y in range(self.n_states):
            score = self.compute_potential_score(w, x, 0, -1, y)
            log_alpha[0, y] = score
        
        # 递推
        for i in range(1, T):
            for y_curr in range(self.n_states):
                log_sum = -np.inf
                for y_prev in range(self.n_states):
                    score = self.compute_potential_score(w, x, i, y_prev, y_curr)
                    log_val = log_alpha[i-1, y_prev] + score
                    log_sum = np.logaddexp(log_sum, log_val)
                log_alpha[i, y_curr] = log_sum
        
        # 计算log Z(x)
        log_Z = -np.inf
        for y in range(self.n_states):
            log_Z = np.logaddexp(log_Z, log_alpha[-1, y])
        
        return log_alpha, log_Z
    
    def backward_algorithm(self, w: np.ndarray, x: List[str]) -> np.ndarray:
        """
        后向算法（对数空间）
        
        返回:
            beta: 后向概率矩阵（对数）
        """
        T = len(x)
        log_beta = np.zeros((T, self.n_states))
        
        # 初始化
        log_beta[-1, :] = 0.0
        
        # 递推
        for i in range(T-2, -1, -1):
            for y_curr in range(self.n_states):
                log_sum = -np.inf
                for y_next in range(self.n_states):
                    score = self.compute_potential_score(w, x, i+1, y_curr, y_next)
                    log_val = score + log_beta[i+1, y_next]
                    log_sum = np.logaddexp(log_sum, log_val)
                log_beta[i, y_curr] = log_sum
        
        return log_beta
    
    def compute_expected_counts(self, w: np.ndarray, x: List[str]) -> np.ndarray:
        """
        计算特征的期望计数（模型期望）
        
        E_p[f] = Σ_y P(y|x) * Σ_i f(y_{i-1}, y_i, x, i)
        """
        T = len(x)
        log_alpha, log_Z = self.forward_algorithm(w, x)
        log_beta = self.backward_algorithm(w, x)
        
        expected_counts = np.zeros(self.n_features)
        
        # 对每个位置
        for i in range(T):
            # 对每个状态转移
            for y_prev in range(-1 if i == 0 else 0, self.n_states):
                for y_curr in range(self.n_states):
                    if i == 0 and y_prev != -1:
                        continue
                    
                    # 计算边缘概率 P(y_{i-1}, y_i | x)
                    if i == 0:
                        log_prob = log_alpha[0, y_curr] + log_beta[0, y_curr] - log_Z
                    else:
                        score = self.compute_potential_score(w, x, i, y_prev, y_curr)
                        log_prob = log_alpha[i-1, y_prev] + score + log_beta[i, y_curr] - log_Z
                    
                    prob = np.exp(log_prob)
                    
                    # 提取特征并累加
                    y = [0] * T
                    if i > 0:
                        y[i-1] = y_prev
                    y[i] = y_curr
                    features = self.extract_features(x, y, i)
                    expected_counts += prob * features
        
        return expected_counts
    
    def compute_empirical_counts(self, X_train: List[List[str]], 
                                 Y_train: List[List[str]]) -> np.ndarray:
        """
        计算特征的经验计数（数据期望）
        
        E_data[f] = Σ_{(x,y)} Σ_i f(y_{i-1}, y_i, x, i)
        """
        empirical_counts = np.zeros(self.n_features)
        
        for x, y_labels in zip(X_train, Y_train):
            y = [self.state_to_idx[label] for label in y_labels]
            T = len(x)
            
            for i in range(T):
                features = self.extract_features(x, y, i)
                empirical_counts += features
        
        return empirical_counts
    
    def compute_loss_and_gradient(self, w: np.ndarray, X_train: List[List[str]], 
                                  Y_train: List[List[str]], 
                                  empirical_counts: np.ndarray,
                                  lambda_reg: float = 0.01) -> Tuple[float, np.ndarray]:
        """
        计算负对数似然损失和梯度
        
        L(w) = -log P(Y|X) + λ/2 * ||w||²
             = -Σ w·f(x,y) + Σ log Z(x) + λ/2 * ||w||²
        
        ∇L(w) = -E_data[f] + E_model[f] + λ*w
        """
        n_samples = len(X_train)
        
        # 计算log Z(x)的和
        log_Z_sum = 0.0
        for x in X_train:
            _, log_Z = self.forward_algorithm(w, x)
            log_Z_sum += log_Z
        
        # 计算模型期望
        model_expected_counts = np.zeros(self.n_features)
        for x in X_train:
            model_expected_counts += self.compute_expected_counts(w, x)
        
        # 计算损失（负对数似然 + L2正则化）
        loss = -np.dot(w, empirical_counts) + log_Z_sum + 0.5 * lambda_reg * np.dot(w, w)
        
        # 计算梯度
        gradient = -empirical_counts + model_expected_counts + lambda_reg * w
        
        return loss, gradient
    
    def train(self, X_train: List[List[str]], Y_train: List[List[str]], 
             lambda_reg: float = 0.01, max_iter: int = 100) -> np.ndarray:
        """
        使用BFGS训练CRF模型
        
        参数:
            X_train: 训练观测序列列表
            Y_train: 训练标注序列列表
            lambda_reg: L2正则化系数
            max_iter: 最大迭代次数
        
        返回:
            weights: 训练后的权重
        """
        if self.verbose:
            print("\n" + "=" * 70)
            print("CRF训练 - BFGS优化算法")
            print("=" * 70)
            print(f"训练样本数: {len(X_train)}")
            print(f"状态数: {self.n_states}")
            print(f"特征数: {self.n_features}")
            print(f"正则化系数 λ: {lambda_reg}")
            print(f"最大迭代次数: {max_iter}")
        
        # 创建特征模板
        if self.n_features == 0:
            self.create_feature_templates()
        
        # 计算经验计数（只需计算一次）
        if self.verbose:
            print("\n计算经验特征计数...")
        empirical_counts = self.compute_empirical_counts(X_train, Y_train)
        
        # 初始化权重
        w0 = np.zeros(self.n_features)
        
        # 定义目标函数（用于scipy.optimize）
        def objective(w):
            loss, grad = self.compute_loss_and_gradient(w, X_train, Y_train, 
                                                        empirical_counts, lambda_reg)
            
            # 记录训练历史
            self.training_history['iteration'].append(len(self.training_history['iteration']))
            self.training_history['loss'].append(loss)
            self.training_history['gradient_norm'].append(np.linalg.norm(grad))
            
            if self.verbose and len(self.training_history['iteration']) % 10 == 1:
                print(f"  迭代 {len(self.training_history['iteration'])-1}: "
                      f"损失 = {loss:.4f}, 梯度范数 = {np.linalg.norm(grad):.4f}")
            
            return loss, grad
        
        # 使用BFGS优化
        if self.verbose:
            print("\n开始BFGS优化...")
        
        result = minimize(
            objective,
            w0,
            method='L-BFGS-B',  # 使用L-BFGS-B（带边界的L-BFGS）
            jac=True,  # objective函数同时返回梯度
            options={
                'maxiter': max_iter,
                'disp': self.verbose
            }
        )
        
        self.weights = result.x
        
        if self.verbose:
            print("\n" + "=" * 70)
            print("训练完成")
            print("=" * 70)
            print(f"最终损失: {result.fun:.4f}")
            print(f"迭代次数: {result.nit}")
            print(f"收敛状态: {result.message}")
            print("=" * 70)
        
        return self.weights
    
    def predict(self, x: List[str]) -> Tuple[List[str], float]:
        """
        使用Viterbi算法进行预测
        
        参数:
            x: 观测序列
        
        返回:
            best_path: 最优标注序列
            best_score: 最优得分
        """
        if self.weights is None:
            raise ValueError("模型尚未训练！请先调用 train() 方法")
        
        T = len(x)
        delta = np.full((T, self.n_states), -np.inf)
        psi = np.zeros((T, self.n_states), dtype=int)
        
        # 初始化
        for y in range(self.n_states):
            score = self.compute_potential_score(self.weights, x, 0, -1, y)
            delta[0, y] = score
        
        # 递推
        for i in range(1, T):
            for y_curr in range(self.n_states):
                for y_prev in range(self.n_states):
                    score = self.compute_potential_score(self.weights, x, i, y_prev, y_curr)
                    total_score = delta[i-1, y_prev] + score
                    
                    if total_score > delta[i, y_curr]:
                        delta[i, y_curr] = total_score
                        psi[i, y_curr] = y_prev
        
        # 回溯
        best_last = np.argmax(delta[-1, :])
        best_score = delta[-1, best_last]
        
        path = [0] * T
        path[-1] = best_last
        for i in range(T-2, -1, -1):
            path[i] = psi[i+1, path[i+1]]
        
        best_path = [self.states[idx] for idx in path]
        
        return best_path, best_score
    
    def evaluate(self, X_test: List[List[str]], Y_test: List[List[str]]) -> float:
        """
        评估模型准确率
        
        参数:
            X_test: 测试观测序列
            Y_test: 测试标注序列
        
        返回:
            accuracy: 准确率
        """
        correct = 0
        total = 0
        
        for x, y_true in zip(X_test, Y_test):
            y_pred, _ = self.predict(x)
            
            for yt, yp in zip(y_true, y_pred):
                if yt == yp:
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy
    
    def visualize_training(self, save_path: str = 'crf_training.png'):
        """
        可视化训练过程
        """
        if len(self.training_history['iteration']) == 0:
            print("没有训练历史数据可供可视化")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('CRF训练过程 (BFGS)', fontsize=16, fontweight='bold')
        
        iterations = self.training_history['iteration']
        losses = self.training_history['loss']
        grad_norms = self.training_history['gradient_norm']
        
        # 1. 损失曲线
        ax1 = axes[0]
        ax1.plot(iterations, losses, 'b-', linewidth=2, label='训练损失')
        ax1.set_xlabel('迭代次数')
        ax1.set_ylabel('负对数似然损失')
        ax1.set_title('损失函数收敛曲线')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. 梯度范数
        ax2 = axes[1]
        ax2.semilogy(iterations, grad_norms, 'r-', linewidth=2, label='梯度范数')
        ax2.set_xlabel('迭代次数')
        ax2.set_ylabel('梯度范数 (log scale)')
        ax2.set_title('梯度范数变化')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n训练过程可视化已保存为: {save_path}")
        plt.close()


def example_train_crf():
    """
    示例：训练CRF模型
    """
    print("=" * 70)
    print("示例：训练线性链CRF模型")
    print("=" * 70)
    
    # 定义状态（中文分词的BMES标注）
    states = ['B', 'M', 'E', 'S']
    
    # 创建训练数据（简化示例）
    X_train = [
        ['我', '爱', '中', '国'],
        ['北', '京', '欢', '迎', '你'],
        ['机', '器', '学', '习'],
        ['自', '然', '语', '言', '处', '理'],
        ['深', '度', '学', '习']
    ]
    
    Y_train = [
        ['S', 'S', 'B', 'E'],  # 我/爱/中国
        ['B', 'E', 'S', 'B', 'E'],  # 北京/欢/迎你
        ['B', 'M', 'M', 'E'],  # 机器学习
        ['S', 'B', 'M', 'M', 'M', 'E'],  # 自/然语言处理
        ['B', 'M', 'M', 'E']  # 深度学习
    ]
    
    print(f"\n训练数据:")
    for x, y in zip(X_train, Y_train):
        print(f"  {''.join(x):10s} -> {' '.join(y)}")
    
    # 创建训练器
    trainer = CRFTrainer(states, verbose=True)
    
    # 训练模型
    weights = trainer.train(X_train, Y_train, lambda_reg=0.1, max_iter=50)
    
    # 可视化训练过程
    trainer.visualize_training()
    
    # 测试
    print("\n" + "=" * 70)
    print("测试预测")
    print("=" * 70)
    
    X_test = [
        ['我', '爱', '北', '京'],
        ['机', '器', '学', '习', '很', '有', '趣']
    ]
    
    for x in X_test:
        y_pred, score = trainer.predict(x)
        print(f"\n观测序列: {''.join(x)}")
        print(f"预测标注: {' '.join(y_pred)}")
        print(f"路径得分: {score:.4f}")
        
        # 分词结果
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
        
        print(f"分词结果: {' / '.join(words)}")
    
    # 评估
    if len(Y_train) > 0:
        train_accuracy = trainer.evaluate(X_train, Y_train)
        print(f"\n训练集准确率: {train_accuracy*100:.2f}%")
    
    return trainer


def example_compare_regularization():
    """
    对比不同正则化系数的效果
    """
    print("\n\n" + "=" * 70)
    print("对比示例：不同正则化系数的影响")
    print("=" * 70)
    
    states = ['B', 'I', 'O']
    
    X_train = [
        ['张', '三', '在', '北', '京'],
        ['李', '四', '去', '上', '海'],
        ['王', '五', '住', '广', '州']
    ]
    
    Y_train = [
        ['B', 'I', 'O', 'B', 'I'],
        ['B', 'I', 'O', 'B', 'I'],
        ['B', 'I', 'O', 'B', 'I']
    ]
    
    lambda_values = [0.0, 0.1, 1.0]
    
    for lambda_reg in lambda_values:
        print(f"\n【λ = {lambda_reg}】")
        trainer = CRFTrainer(states, verbose=False)
        weights = trainer.train(X_train, Y_train, lambda_reg=lambda_reg, max_iter=30)
        
        # 权重统计
        print(f"权重范数: {np.linalg.norm(weights):.4f}")
        print(f"最终损失: {trainer.training_history['loss'][-1]:.4f}")
        
        # 测试
        x_test = ['赵', '六', '在', '深', '圳']
        y_pred, score = trainer.predict(x_test)
        print(f"预测标注: {' '.join(y_pred)}")


if __name__ == "__main__":
    # 运行示例
    example_train_crf()
    example_compare_regularization()
    
    print("\n" + "=" * 70)
    print("CRF训练算法演示完成！")
    print("=" * 70)
