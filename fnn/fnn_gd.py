"""
前馈神经网络 - 批量梯度下降法 (Feedforward Neural Network with Batch Gradient Descent)
====================================================================================

前馈神经网络（Feedforward Neural Network, FNN）是最基础的神经网络架构，
信息单向从输入层流向输出层，不形成循环。

本实现使用批量梯度下降法（Batch Gradient Descent）训练网络。

理论基础
--------

1. **网络结构**:
   
   输入层 -> 隐藏层1 -> 隐藏层2 -> ... -> 输出层
   
   每层之间通过权重矩阵和偏置向量连接。

2. **前向传播**:
   
   对于第 l 层：
   $$z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}$$
   $$a^{[l]} = g^{[l]}(z^{[l]})$$
   
   其中：
   - $W^{[l]}$: 第 l 层的权重矩阵
   - $b^{[l]}$: 第 l 层的偏置向量
   - $g^{[l]}$: 第 l 层的激活函数
   - $a^{[l]}$: 第 l 层的激活值
   - $z^{[l]}$: 第 l 层的线性组合

3. **激活函数**:
   
   **Sigmoid**: $\\sigma(z) = \\frac{1}{1 + e^{-z}}$
   
   导数: $\\sigma'(z) = \\sigma(z)(1 - \\sigma(z))$
   
   **ReLU**: $\\text{ReLU}(z) = \\max(0, z)$
   
   导数: $\\text{ReLU}'(z) = \\begin{cases} 1 & \\text{if } z > 0 \\\\ 0 & \\text{otherwise} \\end{cases}$
   
   **Tanh**: $\\tanh(z) = \\frac{e^z - e^{-z}}{e^z + e^{-z}}$
   
   导数: $\\tanh'(z) = 1 - \\tanh^2(z)$

4. **损失函数**:
   
   **均方误差 (MSE)** - 回归:
   $$J = \frac{1}{2m} \sum_{i=1}^m ||y^{(i)} - \hat{y}^{(i)}||^2$$
   
   **交叉熵 (Cross-Entropy)** - 分类:
   $$J = -\frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K y_k^{(i)} \log(\hat{y}_k^{(i)})$$

5. **反向传播**:
   
   输出层误差:
   $$\delta^{[L]} = \frac{\partial J}{\partial z^{[L]}} = (a^{[L]} - y) \odot g'^{[L]}(z^{[L]})$$
   
   隐藏层误差（从后向前）:
   $$\delta^{[l]} = (W^{[l+1]})^T \delta^{[l+1]} \odot g'^{[l]}(z^{[l]})$$
   
   梯度:
   $$\frac{\partial J}{\partial W^{[l]}} = \frac{1}{m} \delta^{[l]} (a^{[l-1]})^T$$
   $$\frac{\partial J}{\partial b^{[l]}} = \frac{1}{m} \sum_i \delta^{[l],(i)}$$

6. **批量梯度下降 (Batch GD)**:
   
   每次迭代使用全部训练样本计算梯度：
   
   $$W^{[l]} := W^{[l]} - \alpha \frac{\partial J}{\partial W^{[l]}}$$
   $$b^{[l]} := b^{[l]} - \alpha \frac{\partial J}{\partial b^{[l]}}$$
   
   其中 $\alpha$ 是学习率。
   
   **特点**：
   - 梯度计算准确、稳定
   - 收敛路径平滑
   - 计算量大，速度慢
   - 适合小数据集

算法步骤
--------

1. 初始化所有权重和偏置（小随机数）
2. 对于每次迭代：
   a. 前向传播：计算所有层的激活值
   b. 计算损失
   c. 反向传播：计算所有层的梯度
   d. 使用全部样本的平均梯度更新参数
   e. 记录损失和准确率
3. 直到收敛或达到最大迭代次数

日期: 2025-01-19
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Callable
import time

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class FeedforwardNeuralNetwork:
    """
    前馈神经网络 - 批量梯度下降法
    
    参数
    ----
    layer_sizes : List[int]
        每层的神经元数量，例如 [784, 128, 64, 10] 表示：
        - 输入层: 784个神经元
        - 隐藏层1: 128个神经元
        - 隐藏层2: 64个神经元
        - 输出层: 10个神经元
    
    activation : str
        隐藏层激活函数: 'sigmoid', 'relu', 'tanh'
    
    output_activation : str
        输出层激活函数: 'sigmoid', 'softmax', 'linear'
    
    loss : str
        损失函数: 'mse' (均方误差), 'cross_entropy' (交叉熵)
    
    learning_rate : float
        学习率
    
    max_epochs : int
        最大训练轮数
    
    tol : float
        收敛容差（损失变化）
    
    random_state : int
        随机种子
    
    verbose : bool
        是否显示训练信息
    """
    
    def __init__(
        self,
        layer_sizes: List[int],
        activation: str = 'relu',
        output_activation: str = 'softmax',
        loss: str = 'cross_entropy',
        learning_rate: float = 0.01,
        max_epochs: int = 1000,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        verbose: bool = False
    ):
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)
        self.activation = activation
        self.output_activation = output_activation
        self.loss = loss
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        
        # 模型参数
        self.weights = []  # 权重矩阵列表
        self.biases = []   # 偏置向量列表
        
        # 训练历史
        self.train_loss_history = []
        self.train_acc_history = []
        self.n_epochs_ = 0
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _initialize_parameters(self):
        """
        初始化权重和偏置
        
        使用Xavier初始化（适合sigmoid/tanh）或He初始化（适合ReLU）
        """
        self.weights = []
        self.biases = []
        
        for i in range(self.n_layers - 1):
            n_in = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]
            
            # 根据激活函数选择初始化方法
            if self.activation == 'relu':
                # He初始化
                scale = np.sqrt(2.0 / n_in)
            else:
                # Xavier初始化
                scale = np.sqrt(1.0 / n_in)
            
            W = np.random.randn(n_out, n_in) * scale
            b = np.zeros((n_out, 1))
            
            self.weights.append(W)
            self.biases.append(b)
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid激活函数"""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    
    def _sigmoid_derivative(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid导数"""
        s = self._sigmoid(z)
        return s * (1 - s)
    
    def _relu(self, z: np.ndarray) -> np.ndarray:
        """ReLU激活函数"""
        return np.maximum(0, z)
    
    def _relu_derivative(self, z: np.ndarray) -> np.ndarray:
        """ReLU导数"""
        return (z > 0).astype(float)
    
    def _tanh(self, z: np.ndarray) -> np.ndarray:
        """Tanh激活函数"""
        return np.tanh(z)
    
    def _tanh_derivative(self, z: np.ndarray) -> np.ndarray:
        """Tanh导数"""
        t = np.tanh(z)
        return 1 - t ** 2
    
    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """
        Softmax激活函数
        
        数值稳定版本：减去最大值
        """
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)
    
    def _apply_activation(self, z: np.ndarray, activation: str) -> np.ndarray:
        """应用激活函数"""
        if activation == 'sigmoid':
            return self._sigmoid(z)
        elif activation == 'relu':
            return self._relu(z)
        elif activation == 'tanh':
            return self._tanh(z)
        elif activation == 'softmax':
            return self._softmax(z)
        elif activation == 'linear':
            return z
        else:
            raise ValueError(f"未知的激活函数: {activation}")
    
    def _apply_activation_derivative(self, z: np.ndarray, activation: str) -> np.ndarray:
        """应用激活函数的导数"""
        if activation == 'sigmoid':
            return self._sigmoid_derivative(z)
        elif activation == 'relu':
            return self._relu_derivative(z)
        elif activation == 'tanh':
            return self._tanh_derivative(z)
        elif activation == 'linear':
            return np.ones_like(z)
        else:
            raise ValueError(f"未知的激活函数: {activation}")
    
    def _forward_propagation(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        前向传播
        
        参数
        ----
        X : np.ndarray
            输入数据，形状 (n_features, m)
        
        返回
        ----
        Z_cache : List[np.ndarray]
            每层的线性组合 z
        A_cache : List[np.ndarray]
            每层的激活值 a
        """
        A = X
        Z_cache = []
        A_cache = [A]
        
        # 前向传播到隐藏层
        for i in range(self.n_layers - 2):
            Z = self.weights[i] @ A + self.biases[i]
            A = self._apply_activation(Z, self.activation)
            Z_cache.append(Z)
            A_cache.append(A)
        
        # 输出层
        Z = self.weights[-1] @ A + self.biases[-1]
        A = self._apply_activation(Z, self.output_activation)
        Z_cache.append(Z)
        A_cache.append(A)
        
        return Z_cache, A_cache
    
    def _compute_loss(self, Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
        """
        计算损失
        
        参数
        ----
        Y_true : np.ndarray
            真实标签，形状 (n_classes, m)
        Y_pred : np.ndarray
            预测值，形状 (n_classes, m)
        
        返回
        ----
        loss : float
            损失值
        """
        m = Y_true.shape[1]
        
        if self.loss == 'mse':
            # 均方误差
            loss = np.sum((Y_true - Y_pred) ** 2) / (2 * m)
        elif self.loss == 'cross_entropy':
            # 交叉熵（数值稳定版本）
            Y_pred = np.clip(Y_pred, 1e-10, 1 - 1e-10)
            loss = -np.sum(Y_true * np.log(Y_pred)) / m
        else:
            raise ValueError(f"未知的损失函数: {self.loss}")
        
        return loss
    
    def _backward_propagation(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z_cache: List[np.ndarray],
        A_cache: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        反向传播
        
        参数
        ----
        X : np.ndarray
            输入数据，形状 (n_features, m)
        Y : np.ndarray
            真实标签，形状 (n_classes, m)
        Z_cache : List[np.ndarray]
            前向传播的线性组合
        A_cache : List[np.ndarray]
            前向传播的激活值
        
        返回
        ----
        dW : List[np.ndarray]
            权重梯度
        db : List[np.ndarray]
            偏置梯度
        """
        m = X.shape[1]
        dW = [None] * (self.n_layers - 1)
        db = [None] * (self.n_layers - 1)
        
        # 输出层误差
        if self.loss == 'cross_entropy' and self.output_activation == 'softmax':
            # 交叉熵+Softmax的特殊情况：简化为 a - y
            delta = A_cache[-1] - Y
        else:
            # 一般情况
            dA = A_cache[-1] - Y  # 损失对激活值的导数
            dZ = self._apply_activation_derivative(Z_cache[-1], self.output_activation)
            delta = dA * dZ
        
        # 输出层梯度
        dW[-1] = (delta @ A_cache[-2].T) / m
        db[-1] = np.sum(delta, axis=1, keepdims=True) / m
        
        # 反向传播到隐藏层
        for l in range(self.n_layers - 3, -1, -1):
            delta = (self.weights[l + 1].T @ delta) * \
                    self._apply_activation_derivative(Z_cache[l], self.activation)
            dW[l] = (delta @ A_cache[l].T) / m
            db[l] = np.sum(delta, axis=1, keepdims=True) / m
        
        return dW, db
    
    def _update_parameters(self, dW: List[np.ndarray], db: List[np.ndarray]):
        """
        批量梯度下降更新参数
        
        使用全部样本的平均梯度一次性更新所有参数
        """
        for i in range(self.n_layers - 1):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'FeedforwardNeuralNetwork':
        """
        训练神经网络
        
        参数
        ----
        X : np.ndarray
            训练数据，形状 (n_samples, n_features)
        y : np.ndarray
            训练标签，形状 (n_samples,) 或 (n_samples, n_classes)
        
        返回
        ----
        self
        """
        # 数据预处理
        X = X.T  # 转置为 (n_features, m)
        
        # 处理标签
        if y.ndim == 1:
            n_classes = self.layer_sizes[-1]
            if n_classes == 1:
                # 二分类，单个输出神经元
                Y = y.reshape(1, -1)
            else:
                # 多分类，转换为one-hot编码
                Y = np.zeros((n_classes, len(y)))
                Y[y, np.arange(len(y))] = 1
        else:
            Y = y.T
        
        m = X.shape[1]
        
        if self.verbose:
            print(f"批量梯度下降训练开始:")
            print(f"  样本数: {m}")
            print(f"  网络结构: {' -> '.join(map(str, self.layer_sizes))}")
            print(f"  隐藏层激活函数: {self.activation}")
            print(f"  输出层激活函数: {self.output_activation}")
            print(f"  损失函数: {self.loss}")
            print(f"  学习率: {self.learning_rate}")
            print(f"  优化方法: 批量梯度下降 (使用全部{m}个样本)")
        
        # 初始化参数
        self._initialize_parameters()
        
        start_time = time.time()
        
        # 训练循环
        for epoch in range(self.max_epochs):
            # 前向传播（使用全部样本）
            Z_cache, A_cache = self._forward_propagation(X)
            Y_pred = A_cache[-1]
            
            # 计算损失
            loss = self._compute_loss(Y, Y_pred)
            self.train_loss_history.append(loss)
            
            # 计算准确率
            if self.output_activation == 'softmax':
                predictions = np.argmax(Y_pred, axis=0)
                true_labels = np.argmax(Y, axis=0)
                accuracy = np.mean(predictions == true_labels)
            else:
                accuracy = np.mean((Y_pred > 0.5) == Y)
            
            self.train_acc_history.append(accuracy)
            
            # 反向传播（计算全部样本的平均梯度）
            dW, db = self._backward_propagation(X, Y, Z_cache, A_cache)
            
            # 更新参数（一次性使用平均梯度更新）
            self._update_parameters(dW, db)
            
            # 显示进度
            if self.verbose and (epoch + 1) % 100 == 0:
                elapsed = time.time() - start_time
                print(f"  Epoch {epoch + 1}/{self.max_epochs}: "
                      f"Loss={loss:.4f}, Acc={accuracy:.4f}, "
                      f"Time={elapsed:.2f}s")
            
            # 检查收敛
            if epoch > 0 and abs(self.train_loss_history[-1] - self.train_loss_history[-2]) < self.tol:
                if self.verbose:
                    print(f"  Epoch {epoch + 1}: 收敛！")
                break
        
        self.n_epochs_ = epoch + 1
        
        if self.verbose:
            elapsed = time.time() - start_time
            print(f"\n训练完成:")
            print(f"  训练轮数: {self.n_epochs_}")
            print(f"  最终损失: {self.train_loss_history[-1]:.4f}")
            print(f"  最终准确率: {self.train_acc_history[-1]:.4f}")
            print(f"  总耗时: {elapsed:.2f}s")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        参数
        ----
        X : np.ndarray
            测试数据，形状 (n_samples, n_features)
        
        返回
        ----
        predictions : np.ndarray
            预测结果
        """
        X = X.T
        _, A_cache = self._forward_propagation(X)
        Y_pred = A_cache[-1]
        
        if self.output_activation == 'softmax':
            return np.argmax(Y_pred, axis=0)
        else:
            return (Y_pred > 0.5).astype(int).ravel()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率
        
        参数
        ----
        X : np.ndarray
            测试数据，形状 (n_samples, n_features)
        
        返回
        ----
        probabilities : np.ndarray
            预测概率
        """
        X = X.T
        _, A_cache = self._forward_propagation(X)
        return A_cache[-1].T


# ============================================================================
# 示例应用
# ============================================================================

def demo_1_xor_problem():
    """
    示例1: XOR问题（非线性分类）
    """
    print("=" * 70)
    print("示例1: XOR问题 - 批量梯度下降")
    print("=" * 70)
    
    # XOR数据
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([0, 1, 1, 0])
    
    print(f"\nXOR问题数据:")
    for i in range(len(X)):
        print(f"  输入: {X[i]}, 输出: {y[i]}")
    
    # 训练网络
    nn = FeedforwardNeuralNetwork(
        layer_sizes=[2, 4, 1],  # 2输入 -> 4隐藏 -> 1输出
        activation='tanh',
        output_activation='sigmoid',
        loss='mse',
        learning_rate=0.5,
        max_epochs=2000,
        random_state=42,
        verbose=True
    )
    
    nn.fit(X, y)
    
    # 测试
    predictions = nn.predict(X)
    proba = nn.predict_proba(X)
    
    print(f"\n预测结果:")
    for i in range(len(X)):
        print(f"  输入: {X[i]}, 真实: {y[i]}, "
              f"预测: {predictions[i]}, 概率: {proba[i, 0]:.4f}")
    
    # 可视化训练曲线
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(nn.train_loss_history, 'b-', linewidth=2)
    axes[0].set_xlabel('训练轮数')
    axes[0].set_ylabel('损失 (MSE)')
    axes[0].set_title('批量梯度下降 - 损失曲线')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(nn.train_acc_history, 'g-', linewidth=2)
    axes[1].set_xlabel('训练轮数')
    axes[1].set_ylabel('准确率')
    axes[1].set_title('批量梯度下降 - 准确率曲线')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fnn/fnn_gd_xor.png', dpi=150, bbox_inches='tight')
    print("\n图片已保存: fnn/fnn_gd_xor.png")
    plt.close()


def demo_2_iris_classification():
    """
    示例2: 鸢尾花分类（多分类）
    """
    print("\n" + "=" * 70)
    print("示例2: 鸢尾花分类 - 批量梯度下降")
    print("=" * 70)
    
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # 加载数据
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"\n数据集信息:")
    print(f"  训练集: {X_train.shape[0]} 个样本")
    print(f"  测试集: {X_test.shape[0]} 个样本")
    print(f"  特征数: {X_train.shape[1]}")
    print(f"  类别数: {len(np.unique(y))}")
    
    # 训练网络
    nn = FeedforwardNeuralNetwork(
        layer_sizes=[4, 10, 3],  # 4特征 -> 10隐藏 -> 3类
        activation='relu',
        output_activation='softmax',
        loss='cross_entropy',
        learning_rate=0.1,
        max_epochs=1000,
        random_state=42,
        verbose=True
    )
    
    nn.fit(X_train, y_train)
    
    # 评估
    train_pred = nn.predict(X_train)
    test_pred = nn.predict(X_test)
    
    train_acc = np.mean(train_pred == y_train)
    test_acc = np.mean(test_pred == y_test)
    
    print(f"\n模型评估:")
    print(f"  训练集准确率: {train_acc:.4f}")
    print(f"  测试集准确率: {test_acc:.4f}")
    
    # 混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, test_pred)
    
    print(f"\n混淆矩阵:")
    print(cm)
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(nn.train_loss_history, 'b-', linewidth=2, label='训练损失')
    axes[0].set_xlabel('训练轮数')
    axes[0].set_ylabel('交叉熵损失')
    axes[0].set_title('批量梯度下降 - 损失曲线')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(nn.train_acc_history, 'g-', linewidth=2, label='训练准确率')
    axes[1].axhline(y=test_acc, color='r', linestyle='--', linewidth=2, label=f'测试准确率 ({test_acc:.3f})')
    axes[1].set_xlabel('训练轮数')
    axes[1].set_ylabel('准确率')
    axes[1].set_title('批量梯度下降 - 准确率曲线')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fnn/fnn_gd_iris.png', dpi=150, bbox_inches='tight')
    print("\n图片已保存: fnn/fnn_gd_iris.png")
    plt.close()


def demo_3_regression():
    """
    示例3: 回归问题
    """
    print("\n" + "=" * 70)
    print("示例3: 非线性回归 - 批量梯度下降")
    print("=" * 70)
    
    # 生成非线性数据
    np.random.seed(42)
    X = np.linspace(-3, 3, 100).reshape(-1, 1)
    y = np.sin(X).ravel() + np.random.normal(0, 0.1, 100)
    
    print(f"\n数据集信息:")
    print(f"  样本数: {len(X)}")
    print(f"  任务: 拟合 y = sin(x) + noise")
    
    # 训练网络
    nn = FeedforwardNeuralNetwork(
        layer_sizes=[1, 20, 20, 1],  # 1输入 -> 20 -> 20 -> 1输出
        activation='tanh',
        output_activation='linear',
        loss='mse',
        learning_rate=0.01,
        max_epochs=2000,
        random_state=42,
        verbose=True
    )
    
    nn.fit(X, y.reshape(-1, 1))
    
    # 预测
    y_pred = nn.predict_proba(X).ravel()
    mse = np.mean((y - y_pred) ** 2)
    
    print(f"\n模型评估:")
    print(f"  均方误差 (MSE): {mse:.6f}")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 拟合结果
    axes[0].scatter(X, y, alpha=0.5, s=20, label='真实数据')
    axes[0].plot(X, y_pred, 'r-', linewidth=2, label='神经网络拟合')
    axes[0].plot(X, np.sin(X), 'g--', linewidth=2, label='真实函数 sin(x)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title('批量梯度下降 - 非线性回归')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 损失曲线
    axes[1].plot(nn.train_loss_history, 'b-', linewidth=2)
    axes[1].set_xlabel('训练轮数')
    axes[1].set_ylabel('MSE损失')
    axes[1].set_title('批量梯度下降 - 损失曲线')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('fnn/fnn_gd_regression.png', dpi=150, bbox_inches='tight')
    print("\n图片已保存: fnn/fnn_gd_regression.png")
    plt.close()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("前馈神经网络 - 批量梯度下降法")
    print("=" * 70)
    
    # 运行所有示例
    demo_1_xor_problem()
    demo_2_iris_classification()
    demo_3_regression()
    
    print("\n" + "=" * 70)
    print("所有示例运行完成!")
    print("=" * 70)
    print("\n批量梯度下降特点:")
    print("1. 使用全部训练样本计算梯度")
    print("2. 梯度准确，收敛路径平滑")
    print("3. 每轮迭代计算量大")
    print("4. 适合小数据集")
    print("5. 可以并行计算梯度")
