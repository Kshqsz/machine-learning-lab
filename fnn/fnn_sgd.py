"""
前馈神经网络 - 随机梯度下降法 (Feedforward Neural Network with Stochastic Gradient Descent)
==========================================================================================

随机梯度下降（Stochastic Gradient Descent, SGD）是一种优化算法，
每次迭代只使用一个样本（或小批量样本）计算梯度并更新参数。

相比批量梯度下降，SGD训练速度更快，但收敛路径更加曲折。

理论基础
--------

1. **随机梯度下降 vs 批量梯度下降**:

   **批量梯度下降 (Batch GD)**:
   - 每次迭代使用全部 m 个样本
   - 梯度：$\nabla J = \frac{1}{m} \sum_{i=1}^m \nabla J^{(i)}$
   - 特点：准确、稳定、慢
   
   **随机梯度下降 (SGD)**:
   - 每次迭代使用1个样本
   - 梯度：$\nabla J \approx \nabla J^{(i)}$ （单个样本）
   - 特点：快速、噪声大、可能跳出局部最优
   
   **小批量梯度下降 (Mini-batch GD)**:
   - 每次迭代使用 batch_size 个样本
   - 梯度：$\nabla J \approx \frac{1}{b} \sum_{j=1}^b \nabla J^{(j)}$
   - 特点：平衡速度和稳定性

2. **SGD更新规则**:

   对于每个训练样本 $(x^{(i)}, y^{(i)})$：
   
   $$W^{[l]} := W^{[l]} - \alpha \nabla_{W^{[l]}} J^{(i)}$$
   $$b^{[l]} := b^{[l]} - \alpha \nabla_{b^{[l]}} J^{(i)}$$
   
   其中 $J^{(i)}$ 是第 i 个样本的损失。

3. **学习率衰减**:

   为了提高收敛性，学习率通常随训练进行而衰减：
   
   **时间衰减**: $\alpha_t = \frac{\alpha_0}{1 + decay \times t}$
   
   **指数衰减**: $\alpha_t = \alpha_0 \times decay^{epoch}$
   
   **步进衰减**: $\alpha_t = \alpha_0 \times factor^{\lfloor epoch / step \rfloor}$

4. **Epoch vs Iteration**:

   - **1 Epoch**: 遍历全部训练数据一次
   - **1 Iteration**: 更新一次参数
   
   对于 m 个样本：
   - 批量GD: 1 epoch = 1 iteration
   - SGD: 1 epoch = m iterations
   - Mini-batch GD: 1 epoch = m / batch_size iterations

5. **数据洗牌 (Shuffling)**:

   每个epoch开始前随机打乱数据顺序，有助于：
   - 避免顺序偏差
   - 提高收敛速度
   - 增加梯度方向的随机性

6. **SGD的优势**:

   - **速度快**: 不需要等待全部样本计算完成
   - **在线学习**: 可以处理流式数据
   - **内存友好**: 一次只处理少量样本
   - **逃离局部最优**: 梯度噪声有助于跳出鞍点
   - **更好的泛化**: 噪声起到正则化作用

7. **SGD的劣势**:

   - **收敛不稳定**: 梯度噪声大，损失震荡
   - **超参数敏感**: 学习率需要仔细调整
   - **难以并行**: 串行更新参数
   - **收敛速度**: 后期震荡，不易精确收敛

算法步骤
--------

```
输入：训练数据 (X, y)，学习率 α，batch_size
输出：训练好的网络参数

1. 初始化权重和偏置

2. for epoch in range(max_epochs):
   
   a. 随机打乱训练数据
   
   b. 将数据分成多个mini-batch
   
   c. for each mini-batch:
      
      i.   前向传播（使用mini-batch）
      ii.  计算损失
      iii. 反向传播（计算mini-batch的平均梯度）
      iv.  立即更新参数
   
   d. 记录epoch损失和准确率
   
   e. 应用学习率衰减（可选）
   
   f. 检查收敛条件

3. 返回训练好的模型
```

日期: 2025-01-19
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import time

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class FeedforwardNeuralNetworkSGD:
    """
    前馈神经网络 - 随机梯度下降法
    
    参数
    ----
    layer_sizes : List[int]
        每层的神经元数量
    
    activation : str
        隐藏层激活函数: 'sigmoid', 'relu', 'tanh'
    
    output_activation : str
        输出层激活函数: 'sigmoid', 'softmax', 'linear'
    
    loss : str
        损失函数: 'mse', 'cross_entropy'
    
    learning_rate : float
        初始学习率
    
    batch_size : int
        小批量大小
        - 1: 纯SGD
        - m: 批量GD
        - 通常取 16, 32, 64, 128, 256
    
    max_epochs : int
        最大训练轮数
    
    learning_rate_decay : float
        学习率衰减率（每个epoch后）
    
    shuffle : bool
        是否在每个epoch前打乱数据
    
    tol : float
        收敛容差（epoch损失变化）
    
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
        batch_size: int = 32,
        max_epochs: int = 100,
        learning_rate_decay: float = 0.0,
        shuffle: bool = True,
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
        self.initial_learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.learning_rate_decay = learning_rate_decay
        self.shuffle = shuffle
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        
        # 模型参数
        self.weights = []
        self.biases = []
        
        # 训练历史
        self.train_loss_history = []  # 每个epoch的平均损失
        self.train_acc_history = []   # 每个epoch的准确率
        self.batch_loss_history = []  # 每个batch的损失（用于观察震荡）
        self.learning_rate_history = []
        self.n_epochs_ = 0
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _initialize_parameters(self):
        """初始化权重和偏置"""
        self.weights = []
        self.biases = []
        
        for i in range(self.n_layers - 1):
            n_in = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]
            
            if self.activation == 'relu':
                scale = np.sqrt(2.0 / n_in)  # He初始化
            else:
                scale = np.sqrt(1.0 / n_in)  # Xavier初始化
            
            W = np.random.randn(n_out, n_in) * scale
            b = np.zeros((n_out, 1))
            
            self.weights.append(W)
            self.biases.append(b)
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    
    def _sigmoid_derivative(self, z: np.ndarray) -> np.ndarray:
        s = self._sigmoid(z)
        return s * (1 - s)
    
    def _relu(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(0, z)
    
    def _relu_derivative(self, z: np.ndarray) -> np.ndarray:
        return (z > 0).astype(float)
    
    def _tanh(self, z: np.ndarray) -> np.ndarray:
        return np.tanh(z)
    
    def _tanh_derivative(self, z: np.ndarray) -> np.ndarray:
        t = np.tanh(z)
        return 1 - t ** 2
    
    def _softmax(self, z: np.ndarray) -> np.ndarray:
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)
    
    def _apply_activation(self, z: np.ndarray, activation: str) -> np.ndarray:
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
        """前向传播"""
        A = X
        Z_cache = []
        A_cache = [A]
        
        for i in range(self.n_layers - 2):
            Z = self.weights[i] @ A + self.biases[i]
            A = self._apply_activation(Z, self.activation)
            Z_cache.append(Z)
            A_cache.append(A)
        
        Z = self.weights[-1] @ A + self.biases[-1]
        A = self._apply_activation(Z, self.output_activation)
        Z_cache.append(Z)
        A_cache.append(A)
        
        return Z_cache, A_cache
    
    def _compute_loss(self, Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
        """计算损失"""
        m = Y_true.shape[1]
        
        if self.loss == 'mse':
            loss = np.sum((Y_true - Y_pred) ** 2) / (2 * m)
        elif self.loss == 'cross_entropy':
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
        """反向传播"""
        m = X.shape[1]
        dW = [None] * (self.n_layers - 1)
        db = [None] * (self.n_layers - 1)
        
        # 输出层误差
        if self.loss == 'cross_entropy' and self.output_activation == 'softmax':
            delta = A_cache[-1] - Y
        else:
            dA = A_cache[-1] - Y
            dZ = self._apply_activation_derivative(Z_cache[-1], self.output_activation)
            delta = dA * dZ
        
        # 输出层梯度
        dW[-1] = (delta @ A_cache[-2].T) / m
        db[-1] = np.sum(delta, axis=1, keepdims=True) / m
        
        # 隐藏层梯度
        for l in range(self.n_layers - 3, -1, -1):
            delta = (self.weights[l + 1].T @ delta) * \
                    self._apply_activation_derivative(Z_cache[l], self.activation)
            dW[l] = (delta @ A_cache[l].T) / m
            db[l] = np.sum(delta, axis=1, keepdims=True) / m
        
        return dW, db
    
    def _update_parameters(self, dW: List[np.ndarray], db: List[np.ndarray]):
        """SGD参数更新"""
        for i in range(self.n_layers - 1):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]
    
    def _create_mini_batches(
        self,
        X: np.ndarray,
        Y: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        创建mini-batches
        
        返回
        ----
        mini_batches : List[Tuple]
            [(X_batch, Y_batch), ...]
        """
        m = X.shape[1]
        mini_batches = []
        
        # 打乱数据
        if self.shuffle:
            permutation = np.random.permutation(m)
            X_shuffled = X[:, permutation]
            Y_shuffled = Y[:, permutation]
        else:
            X_shuffled = X
            Y_shuffled = Y
        
        # 分割成batches
        n_complete_batches = m // self.batch_size
        
        for k in range(n_complete_batches):
            start = k * self.batch_size
            end = (k + 1) * self.batch_size
            X_batch = X_shuffled[:, start:end]
            Y_batch = Y_shuffled[:, start:end]
            mini_batches.append((X_batch, Y_batch))
        
        # 处理剩余样本
        if m % self.batch_size != 0:
            start = n_complete_batches * self.batch_size
            X_batch = X_shuffled[:, start:]
            Y_batch = Y_shuffled[:, start:]
            mini_batches.append((X_batch, Y_batch))
        
        return mini_batches
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'FeedforwardNeuralNetworkSGD':
        """
        训练神经网络（随机梯度下降）
        
        参数
        ----
        X : np.ndarray
            训练数据，形状 (n_samples, n_features)
        y : np.ndarray
            训练标签
        
        返回
        ----
        self
        """
        # 数据预处理
        X = X.T  # (n_features, m)
        
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
        n_batches = int(np.ceil(m / self.batch_size))
        
        if self.verbose:
            print(f"随机梯度下降训练开始:")
            print(f"  样本数: {m}")
            print(f"  网络结构: {' -> '.join(map(str, self.layer_sizes))}")
            print(f"  批量大小: {self.batch_size}")
            print(f"  每轮迭代次数: {n_batches}")
            if self.batch_size == 1:
                print(f"  优化方法: 纯SGD（每次1个样本）")
            elif self.batch_size == m:
                print(f"  优化方法: 批量梯度下降（全部样本）")
            else:
                print(f"  优化方法: Mini-batch SGD")
            print(f"  初始学习率: {self.learning_rate}")
            if self.learning_rate_decay > 0:
                print(f"  学习率衰减: {self.learning_rate_decay}")
            print(f"  数据洗牌: {self.shuffle}")
        
        # 初始化参数
        self._initialize_parameters()
        
        start_time = time.time()
        
        # 训练循环
        for epoch in range(self.max_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            # 创建mini-batches
            mini_batches = self._create_mini_batches(X, Y)
            
            # 对每个mini-batch进行训练
            for X_batch, Y_batch in mini_batches:
                # 前向传播
                Z_cache, A_cache = self._forward_propagation(X_batch)
                Y_pred = A_cache[-1]
                
                # 计算损失
                batch_loss = self._compute_loss(Y_batch, Y_pred)
                epoch_loss += batch_loss * Y_batch.shape[1]  # 累积损失
                
                # 记录batch损失（用于观察震荡）
                self.batch_loss_history.append(batch_loss)
                
                # 计算准确率
                if self.output_activation == 'softmax':
                    predictions = np.argmax(Y_pred, axis=0)
                    true_labels = np.argmax(Y_batch, axis=0)
                    epoch_correct += np.sum(predictions == true_labels)
                else:
                    epoch_correct += np.sum((Y_pred > 0.5) == Y_batch)
                
                epoch_total += Y_batch.shape[1]
                
                # 反向传播
                dW, db = self._backward_propagation(X_batch, Y_batch, Z_cache, A_cache)
                
                # 立即更新参数（SGD的核心）
                self._update_parameters(dW, db)
            
            # Epoch统计
            epoch_loss /= m
            epoch_acc = epoch_correct / epoch_total
            
            self.train_loss_history.append(epoch_loss)
            self.train_acc_history.append(epoch_acc)
            self.learning_rate_history.append(self.learning_rate)
            
            # 学习率衰减
            if self.learning_rate_decay > 0:
                self.learning_rate = self.initial_learning_rate / (1 + self.learning_rate_decay * (epoch + 1))
            
            # 显示进度
            if self.verbose and (epoch + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"  Epoch {epoch + 1}/{self.max_epochs}: "
                      f"Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}, "
                      f"LR={self.learning_rate:.6f}, Time={elapsed:.2f}s")
            
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
            print(f"  总迭代次数: {self.n_epochs_ * n_batches}")
            print(f"  最终损失: {self.train_loss_history[-1]:.4f}")
            print(f"  最终准确率: {self.train_acc_history[-1]:.4f}")
            print(f"  总耗时: {elapsed:.2f}s")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        X = X.T
        _, A_cache = self._forward_propagation(X)
        Y_pred = A_cache[-1]
        
        if self.output_activation == 'softmax':
            return np.argmax(Y_pred, axis=0)
        else:
            return (Y_pred > 0.5).astype(int).ravel()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        X = X.T
        _, A_cache = self._forward_propagation(X)
        return A_cache[-1].T


# ============================================================================
# 示例应用
# ============================================================================

def demo_1_xor_problem():
    """
    示例1: XOR问题 - SGD vs Batch GD对比
    """
    print("=" * 70)
    print("示例1: XOR问题 - SGD vs Batch GD对比")
    print("=" * 70)
    
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    
    print(f"\nXOR问题数据: {len(X)} 个样本")
    
    # 训练多个模型进行对比
    configs = [
        {'batch_size': 4, 'name': '批量GD'},
        {'batch_size': 2, 'name': 'Mini-batch (batch=2)'},
        {'batch_size': 1, 'name': '纯SGD (batch=1)'},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n训练 {config['name']}...")
        nn = FeedforwardNeuralNetworkSGD(
            layer_sizes=[2, 4, 1],
            activation='tanh',
            output_activation='sigmoid',
            loss='mse',
            learning_rate=0.5,
            batch_size=config['batch_size'],
            max_epochs=500,
            shuffle=True,
            random_state=42,
            verbose=False
        )
        
        nn.fit(X, y)
        predictions = nn.predict(X)
        accuracy = np.mean(predictions == y)
        
        results.append({
            'name': config['name'],
            'model': nn,
            'accuracy': accuracy
        })
        
        print(f"  最终准确率: {accuracy:.4f}")
    
    # 可视化对比
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 损失曲线对比（每个epoch）
    for r in results:
        axes[0, 0].plot(r['model'].train_loss_history, 
                        label=r['name'], linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('损失 (MSE)')
    axes[0, 0].set_title('损失曲线对比（每个Epoch）')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Batch损失曲线（展示震荡）
    colors = ['blue', 'orange', 'green']
    for i, r in enumerate(results):
        n_batches = len(r['model'].batch_loss_history)
        axes[0, 1].plot(range(n_batches), r['model'].batch_loss_history, 
                        label=r['name'], linewidth=1, alpha=0.7, color=colors[i])
    axes[0, 1].set_xlabel('Iteration (batch)')
    axes[0, 1].set_ylabel('损失')
    axes[0, 1].set_title('每个Batch的损失（展示震荡）')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 准确率曲线
    for r in results:
        axes[1, 0].plot(r['model'].train_acc_history, 
                        label=r['name'], linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('准确率')
    axes[1, 0].set_title('准确率曲线对比')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 最终对比
    names = [r['name'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    axes[1, 1].bar(names, accuracies, color=['blue', 'orange', 'green'])
    axes[1, 1].set_ylabel('最终准确率')
    axes[1, 1].set_title('最终准确率对比')
    axes[1, 1].set_ylim([0, 1.1])
    for i, v in enumerate(accuracies):
        axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('fnn/fnn_sgd_comparison.png', dpi=150, bbox_inches='tight')
    print("\n图片已保存: fnn/fnn_sgd_comparison.png")
    plt.close()


def demo_2_iris_with_sgd():
    """
    示例2: 鸢尾花分类 - SGD训练
    """
    print("\n" + "=" * 70)
    print("示例2: 鸢尾花分类 - Mini-batch SGD")
    print("=" * 70)
    
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    iris = load_iris()
    X, y = iris.data, iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"\n数据集: 训练{X_train.shape[0]}个, 测试{X_test.shape[0]}个")
    
    # 训练SGD模型
    nn = FeedforwardNeuralNetworkSGD(
        layer_sizes=[4, 10, 3],
        activation='relu',
        output_activation='softmax',
        loss='cross_entropy',
        learning_rate=0.1,
        batch_size=16,  # Mini-batch
        max_epochs=200,
        learning_rate_decay=0.01,  # 学习率衰减
        shuffle=True,
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
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 损失曲线
    axes[0, 0].plot(nn.train_loss_history, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('交叉熵损失')
    axes[0, 0].set_title('Mini-batch SGD - 损失曲线')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Batch损失（展示震荡）
    axes[0, 1].plot(nn.batch_loss_history, 'r-', linewidth=0.5, alpha=0.5)
    axes[0, 1].set_xlabel('Iteration (batch)')
    axes[0, 1].set_ylabel('损失')
    axes[0, 1].set_title('每个Batch的损失（震荡）')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 准确率曲线
    axes[1, 0].plot(nn.train_acc_history, 'g-', linewidth=2, label='训练准确率')
    axes[1, 0].axhline(y=test_acc, color='r', linestyle='--', 
                       linewidth=2, label=f'测试准确率 ({test_acc:.3f})')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('准确率')
    axes[1, 0].set_title('Mini-batch SGD - 准确率曲线')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 学习率衰减
    axes[1, 1].plot(nn.learning_rate_history, 'purple', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('学习率')
    axes[1, 1].set_title('学习率衰减曲线')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fnn/fnn_sgd_iris.png', dpi=150, bbox_inches='tight')
    print("\n图片已保存: fnn/fnn_sgd_iris.png")
    plt.close()


def demo_3_batch_size_comparison():
    """
    示例3: 不同batch_size的对比
    """
    print("\n" + "=" * 70)
    print("示例3: 不同Batch Size的性能对比")
    print("=" * 70)
    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # 生成分类数据
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15,
        n_redundant=5, n_classes=3, n_clusters_per_class=1,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n数据集: {X_train.shape[0]} 训练样本, {X_test.shape[0]} 测试样本")
    
    # 测试不同的batch_size
    batch_sizes = [1, 8, 32, 128, len(X_train)]
    batch_names = ['SGD(1)', 'Mini(8)', 'Mini(32)', 'Mini(128)', f'Batch({len(X_train)})']
    
    results = []
    
    for batch_size, name in zip(batch_sizes, batch_names):
        print(f"\n训练 batch_size={batch_size} ({name})...")
        
        start_time = time.time()
        
        nn = FeedforwardNeuralNetworkSGD(
            layer_sizes=[20, 50, 3],
            activation='relu',
            output_activation='softmax',
            loss='cross_entropy',
            learning_rate=0.01,
            batch_size=batch_size,
            max_epochs=50,
            shuffle=True,
            random_state=42,
            verbose=False
        )
        
        nn.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        test_acc = np.mean(nn.predict(X_test) == y_test)
        
        results.append({
            'name': name,
            'batch_size': batch_size,
            'model': nn,
            'test_acc': test_acc,
            'train_time': train_time,
            'n_iterations': len(nn.batch_loss_history)
        })
        
        print(f"  测试准确率: {test_acc:.4f}")
        print(f"  训练时间: {train_time:.2f}s")
        print(f"  总迭代次数: {results[-1]['n_iterations']}")
    
    # 可视化对比
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 损失曲线
    for r in results:
        axes[0, 0].plot(r['model'].train_loss_history, 
                        label=r['name'], linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('损失')
    axes[0, 0].set_title('不同Batch Size的损失曲线')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 准确率对比
    names = [r['name'] for r in results]
    accs = [r['test_acc'] for r in results]
    axes[0, 1].bar(range(len(names)), accs, color='skyblue')
    axes[0, 1].set_xticks(range(len(names)))
    axes[0, 1].set_xticklabels(names, rotation=45)
    axes[0, 1].set_ylabel('测试准确率')
    axes[0, 1].set_title('不同Batch Size的准确率')
    axes[0, 1].set_ylim([0, 1.0])
    for i, v in enumerate(accs):
        axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    # 训练时间对比
    times = [r['train_time'] for r in results]
    axes[1, 0].bar(range(len(names)), times, color='orange')
    axes[1, 0].set_xticks(range(len(names)))
    axes[1, 0].set_xticklabels(names, rotation=45)
    axes[1, 0].set_ylabel('训练时间 (秒)')
    axes[1, 0].set_title('不同Batch Size的训练时间')
    for i, v in enumerate(times):
        axes[1, 0].text(i, v + 0.5, f'{v:.1f}s', ha='center')
    
    # 迭代次数对比
    iterations = [r['n_iterations'] for r in results]
    axes[1, 1].bar(range(len(names)), iterations, color='green')
    axes[1, 1].set_xticks(range(len(names)))
    axes[1, 1].set_xticklabels(names, rotation=45)
    axes[1, 1].set_ylabel('总迭代次数')
    axes[1, 1].set_title('不同Batch Size的迭代次数')
    for i, v in enumerate(iterations):
        axes[1, 1].text(i, v + 100, f'{v}', ha='center')
    
    plt.tight_layout()
    plt.savefig('fnn/fnn_sgd_batch_comparison.png', dpi=150, bbox_inches='tight')
    print("\n图片已保存: fnn/fnn_sgd_batch_comparison.png")
    plt.close()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("前馈神经网络 - 随机梯度下降法")
    print("=" * 70)
    
    # 运行所有示例
    demo_1_xor_problem()
    demo_2_iris_with_sgd()
    demo_3_batch_size_comparison()
    
    print("\n" + "=" * 70)
    print("所有示例运行完成!")
    print("=" * 70)
    print("\nSGD vs Batch GD 总结:")
    print("\n批量梯度下降 (Batch GD):")
    print("  ✓ 梯度准确，收敛稳定")
    print("  ✓ 适合小数据集")
    print("  ✗ 速度慢，内存占用大")
    print("  ✗ 易陷入局部最优")
    
    print("\n随机梯度下降 (SGD):")
    print("  ✓ 速度快，内存友好")
    print("  ✓ 可处理大规模数据")
    print("  ✓ 噪声有助于逃离局部最优")
    print("  ✗ 收敛不稳定，震荡明显")
    print("  ✗ 需要仔细调整学习率")
    
    print("\nMini-batch SGD (推荐):")
    print("  ✓ 平衡速度和稳定性")
    print("  ✓ 可以利用GPU并行计算")
    print("  ✓ 常用batch_size: 32, 64, 128, 256")
