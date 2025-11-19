"""
前馈神经网络 - 早停法 (Early Stopping)
==========================================

早停法（Early Stopping）是一种有效的正则化技术，通过监控验证集性能来防止过拟合。
当验证集性能不再提升时，提前终止训练。

理论基础
--------

1. **过拟合问题**:

   随着训练进行：
   - 训练误差持续下降
   - 验证误差先下降后上升
   
   当验证误差开始上升时，模型开始过拟合。

2. **早停策略**:

   在每个epoch后：
   - 评估验证集性能
   - 如果性能提升，保存模型
   - 如果连续 patience 个epoch没有提升，停止训练
   - 恢复到验证性能最佳的模型

3. **监控指标**:

   - **损失（Loss）**: 适合回归问题
   - **准确率（Accuracy）**: 适合分类问题
   - **其他指标**: F1-score, AUC等

4. **性能提升判断**:

   相对改进：
   $$\\text{improvement} = \\frac{\\text{best} - \\text{current}}{\\text{best}}$$
   
   绝对改进：
   $$\\text{improvement} = \\text{best} - \\text{current}$$
   
   如果 improvement > min_delta，认为有提升。

5. **早停法的优势**:

   - **防止过拟合**: 在最佳泛化点停止
   - **节省时间**: 不需要训练完全部epoch
   - **自动化**: 无需手动判断停止时机
   - **简单有效**: 无额外超参数调整负担

6. **与其他正则化方法对比**:

   | 方法 | 原理 | 优点 | 缺点 |
   |------|------|------|------|
   | L1/L2正则 | 惩罚权重大小 | 理论完善 | 需调参λ |
   | Dropout | 随机失活神经元 | 效果好 | 增加计算 |
   | Early Stopping | 提前终止训练 | 简单高效 | 需验证集 |
   | Data Augmentation | 增加数据多样性 | 提升泛化 | 需领域知识 |

7. **超参数**:

   - **patience**: 容忍多少个epoch无改进（通常5-20）
   - **min_delta**: 最小改进阈值（通常1e-4）
   - **monitor**: 监控指标（'val_loss' 或 'val_acc'）
   - **mode**: 'min'（最小化）或 'max'（最大化）
   - **restore_best**: 是否恢复最佳模型

8. **训练曲线分析**:

   典型的训练/验证曲线：
   ```
   Loss
   │
   │  训练集 ↘↘↘↘↘↘↘↘↘
   │          
   │  验证集 ↘↘↘↗↗↗
   │              ↑
   │           早停点
   └─────────────────> Epoch
   ```

算法步骤
--------

```
输入：训练集、验证集、patience、min_delta
输出：最佳模型

1. 初始化:
   best_score = ∞ (for loss) 或 -∞ (for accuracy)
   patience_counter = 0
   best_weights = None

2. for epoch in range(max_epochs):
   
   a. 训练一个epoch
   
   b. 评估验证集:
      val_score = evaluate(validation_set)
   
   c. 判断是否改进:
      if has_improved(val_score, best_score, min_delta):
          best_score = val_score
          best_weights = copy_weights()
          patience_counter = 0
      else:
          patience_counter += 1
   
   d. 早停判断:
      if patience_counter >= patience:
          print("Early stopping triggered")
          restore_weights(best_weights)
          break

3. 返回最佳模型
```

日期: 2025-01-19
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict
import copy
import time

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class EarlyStoppingNN:
    """
    前馈神经网络 - 早停法
    
    参数
    ----
    layer_sizes : List[int]
        每层神经元数量
    
    activation : str
        隐藏层激活函数
    
    output_activation : str
        输出层激活函数
    
    loss : str
        损失函数
    
    learning_rate : float
        学习率
    
    batch_size : int
        批量大小
    
    max_epochs : int
        最大训练轮数
    
    patience : int
        早停容忍度（连续多少个epoch无改进）
    
    min_delta : float
        最小改进阈值
    
    monitor : str
        监控指标: 'val_loss' 或 'val_acc'
    
    mode : str
        'min'（最小化）或 'max'（最大化）
    
    restore_best_weights : bool
        是否恢复到最佳权重
    
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
        max_epochs: int = 1000,
        patience: int = 10,
        min_delta: float = 1e-4,
        monitor: str = 'val_loss',
        mode: str = 'min',
        restore_best_weights: bool = True,
        random_state: Optional[int] = None,
        verbose: bool = False
    ):
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)
        self.activation = activation
        self.output_activation = output_activation
        self.loss = loss
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        # 模型参数
        self.weights = []
        self.biases = []
        
        # 早停相关
        self.best_weights = None
        self.best_biases = None
        self.best_score = np.inf if mode == 'min' else -np.inf
        self.best_epoch = 0
        self.patience_counter = 0
        self.stopped_epoch = 0
        
        # 训练历史
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []
        self.n_epochs_ = 0
        
        if random_state is not None:
            np.random.seed(random_state)
    
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
            raise ValueError(f"未知激活函数: {activation}")
    
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
            raise ValueError(f"未知激活函数: {activation}")
    
    def _initialize_parameters(self):
        """初始化权重和偏置"""
        self.weights = []
        self.biases = []
        
        for i in range(self.n_layers - 1):
            n_in = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]
            
            if self.activation == 'relu':
                scale = np.sqrt(2.0 / n_in)
            else:
                scale = np.sqrt(1.0 / n_in)
            
            W = np.random.randn(n_out, n_in) * scale
            b = np.zeros((n_out, 1))
            
            self.weights.append(W)
            self.biases.append(b)
    
    def _save_weights(self):
        """保存当前权重"""
        self.best_weights = [W.copy() for W in self.weights]
        self.best_biases = [b.copy() for b in self.biases]
    
    def _restore_weights(self):
        """恢复最佳权重"""
        if self.best_weights is not None:
            self.weights = [W.copy() for W in self.best_weights]
            self.biases = [b.copy() for b in self.best_biases]
    
    def _forward_propagation(self, X: np.ndarray) -> Tuple[List, List]:
        """前向传播"""
        A = X
        Z_list = []
        A_list = [A]
        
        for i in range(self.n_layers - 2):
            Z = self.weights[i] @ A + self.biases[i]
            A = self._apply_activation(Z, self.activation)
            Z_list.append(Z)
            A_list.append(A)
        
        Z = self.weights[-1] @ A + self.biases[-1]
        A = self._apply_activation(Z, self.output_activation)
        Z_list.append(Z)
        A_list.append(A)
        
        return Z_list, A_list
    
    def _compute_loss(self, Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
        """计算损失"""
        m = Y_true.shape[1]
        
        if self.loss == 'mse':
            loss = np.sum((Y_true - Y_pred) ** 2) / (2 * m)
        elif self.loss == 'cross_entropy':
            Y_pred = np.clip(Y_pred, 1e-10, 1 - 1e-10)
            loss = -np.sum(Y_true * np.log(Y_pred)) / m
        else:
            raise ValueError(f"未知损失函数: {self.loss}")
        
        return loss
    
    def _backward_propagation(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z_list: List[np.ndarray],
        A_list: List[np.ndarray]
    ) -> Tuple[List, List]:
        """反向传播"""
        m = X.shape[1]
        dW_list = [None] * (self.n_layers - 1)
        db_list = [None] * (self.n_layers - 1)
        
        # 输出层误差
        if self.loss == 'cross_entropy' and self.output_activation == 'softmax':
            delta = A_list[-1] - Y
        else:
            dA = A_list[-1] - Y
            dZ = self._apply_activation_derivative(Z_list[-1], self.output_activation)
            delta = dA * dZ
        
        dW_list[-1] = (delta @ A_list[-2].T) / m
        db_list[-1] = np.sum(delta, axis=1, keepdims=True) / m
        
        # 隐藏层
        for l in range(self.n_layers - 3, -1, -1):
            delta = (self.weights[l + 1].T @ delta) * \
                    self._apply_activation_derivative(Z_list[l], self.activation)
            dW_list[l] = (delta @ A_list[l].T) / m
            db_list[l] = np.sum(delta, axis=1, keepdims=True) / m
        
        return dW_list, db_list
    
    def _update_parameters(self, dW_list: List, db_list: List):
        """更新参数"""
        for i in range(self.n_layers - 1):
            self.weights[i] -= self.learning_rate * dW_list[i]
            self.biases[i] -= self.learning_rate * db_list[i]
    
    def _create_mini_batches(self, X: np.ndarray, Y: np.ndarray) -> List[Tuple]:
        """创建mini-batches"""
        m = X.shape[1]
        mini_batches = []
        
        permutation = np.random.permutation(m)
        X_shuffled = X[:, permutation]
        Y_shuffled = Y[:, permutation]
        
        n_complete_batches = m // self.batch_size
        
        for k in range(n_complete_batches):
            start = k * self.batch_size
            end = (k + 1) * self.batch_size
            X_batch = X_shuffled[:, start:end]
            Y_batch = Y_shuffled[:, start:end]
            mini_batches.append((X_batch, Y_batch))
        
        if m % self.batch_size != 0:
            start = n_complete_batches * self.batch_size
            X_batch = X_shuffled[:, start:]
            Y_batch = Y_shuffled[:, start:]
            mini_batches.append((X_batch, Y_batch))
        
        return mini_batches
    
    def _evaluate(self, X: np.ndarray, Y: np.ndarray) -> Tuple[float, float]:
        """
        评估数据集
        
        返回
        ----
        loss : float
            损失值
        accuracy : float
            准确率
        """
        _, A_list = self._forward_propagation(X)
        Y_pred = A_list[-1]
        
        loss = self._compute_loss(Y, Y_pred)
        
        if self.output_activation == 'softmax' or Y.shape[0] > 1:
            predictions = np.argmax(Y_pred, axis=0)
            true_labels = np.argmax(Y, axis=0)
            accuracy = np.mean(predictions == true_labels)
        else:
            accuracy = np.mean((Y_pred > 0.5) == Y)
        
        return loss, accuracy
    
    def _has_improved(self, current_score: float) -> bool:
        """
        判断是否有改进
        
        参数
        ----
        current_score : float
            当前得分
        
        返回
        ----
        improved : bool
            是否改进
        """
        if self.mode == 'min':
            # 最小化：current < best - min_delta
            return current_score < self.best_score - self.min_delta
        else:
            # 最大化：current > best + min_delta
            return current_score > self.best_score + self.min_delta
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> 'EarlyStoppingNN':
        """
        训练神经网络（带早停）
        
        参数
        ----
        X_train : np.ndarray
            训练数据 (n_samples, n_features)
        y_train : np.ndarray
            训练标签
        X_val : np.ndarray
            验证数据
        y_val : np.ndarray
            验证标签
        
        返回
        ----
        self
        """
        # 数据预处理
        X_train = X_train.T
        X_val = X_val.T
        
        # 处理标签
        def process_labels(y):
            if y.ndim == 1:
                n_classes = self.layer_sizes[-1]
                if n_classes == 1:
                    return y.reshape(1, -1)
                else:
                    Y = np.zeros((n_classes, len(y)))
                    Y[y, np.arange(len(y))] = 1
                    return Y
            else:
                return y.T
        
        Y_train = process_labels(y_train)
        Y_val = process_labels(y_val)
        
        m_train = X_train.shape[1]
        m_val = X_val.shape[1]
        
        if self.verbose:
            print(f"早停法训练开始:")
            print(f"  训练集: {m_train} 个样本")
            print(f"  验证集: {m_val} 个样本")
            print(f"  网络结构: {' -> '.join(map(str, self.layer_sizes))}")
            print(f"  批量大小: {self.batch_size}")
            print(f"  早停参数:")
            print(f"    - patience: {self.patience}")
            print(f"    - min_delta: {self.min_delta}")
            print(f"    - monitor: {self.monitor}")
            print(f"    - mode: {self.mode}")
        
        # 初始化参数
        self._initialize_parameters()
        
        # 初始化早停
        self.best_score = np.inf if self.mode == 'min' else -np.inf
        self.patience_counter = 0
        self.stopped_epoch = 0
        
        start_time = time.time()
        
        # 训练循环
        for epoch in range(self.max_epochs):
            # 训练一个epoch
            epoch_loss = 0.0
            mini_batches = self._create_mini_batches(X_train, Y_train)
            
            for X_batch, Y_batch in mini_batches:
                Z_list, A_list = self._forward_propagation(X_batch)
                batch_loss = self._compute_loss(Y_batch, A_list[-1])
                epoch_loss += batch_loss * Y_batch.shape[1]
                
                dW_list, db_list = self._backward_propagation(X_batch, Y_batch, Z_list, A_list)
                self._update_parameters(dW_list, db_list)
            
            # 评估训练集
            train_loss, train_acc = self._evaluate(X_train, Y_train)
            self.train_loss_history.append(train_loss)
            self.train_acc_history.append(train_acc)
            
            # 评估验证集
            val_loss, val_acc = self._evaluate(X_val, Y_val)
            self.val_loss_history.append(val_loss)
            self.val_acc_history.append(val_acc)
            
            # 获取监控指标
            if self.monitor == 'val_loss':
                current_score = val_loss
            elif self.monitor == 'val_acc':
                current_score = val_acc
            else:
                raise ValueError(f"未知监控指标: {self.monitor}")
            
            # 早停判断
            if self._has_improved(current_score):
                # 性能提升
                self.best_score = current_score
                self.best_epoch = epoch
                self.patience_counter = 0
                
                if self.restore_best_weights:
                    self._save_weights()
                
                if self.verbose:
                    print(f"  Epoch {epoch + 1}/{self.max_epochs}: "
                          f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                          f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f} "
                          f"✓ (best {self.monitor}={current_score:.4f})")
            else:
                # 性能未提升
                self.patience_counter += 1
                
                if self.verbose and (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch + 1}/{self.max_epochs}: "
                          f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                          f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f} "
                          f"(no improvement for {self.patience_counter} epochs)")
                
                # 触发早停
                if self.patience_counter >= self.patience:
                    self.stopped_epoch = epoch
                    if self.verbose:
                        print(f"\\n早停触发！")
                        print(f"  在第 {epoch + 1} 轮停止训练")
                        print(f"  最佳 {self.monitor} = {self.best_score:.4f} (第 {self.best_epoch + 1} 轮)")
                    
                    if self.restore_best_weights:
                        self._restore_weights()
                        if self.verbose:
                            print(f"  已恢复到第 {self.best_epoch + 1} 轮的最佳权重")
                    
                    break
        
        self.n_epochs_ = epoch + 1
        
        if self.verbose:
            elapsed = time.time() - start_time
            print(f"\\n训练完成:")
            print(f"  实际训练轮数: {self.n_epochs_}")
            if self.stopped_epoch > 0:
                print(f"  节省轮数: {self.max_epochs - self.n_epochs_}")
            print(f"  最佳 {self.monitor}: {self.best_score:.4f}")
            print(f"  最终训练损失: {self.train_loss_history[-1]:.4f}")
            print(f"  最终验证损失: {self.val_loss_history[-1]:.4f}")
            print(f"  总耗时: {elapsed:.2f}s")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        X = X.T
        _, A_list = self._forward_propagation(X)
        Y_pred = A_list[-1]
        
        if self.layer_sizes[-1] > 1:
            return np.argmax(Y_pred, axis=0)
        else:
            return (Y_pred > 0.5).astype(int).ravel()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        X = X.T
        _, A_list = self._forward_propagation(X)
        return A_list[-1].T


# ============================================================================
# 示例应用
# ============================================================================

def demo_1_early_stopping_basic():
    """
    示例1: 早停法基础演示
    """
    print("=" * 70)
    print("示例1: 早停法基础演示")
    print("=" * 70)
    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # 生成数据
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42
    )
    
    # 划分数据集
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    print(f"\\n数据集划分:")
    print(f"  训练集: {X_train.shape[0]} 个样本")
    print(f"  验证集: {X_val.shape[0]} 个样本")
    print(f"  测试集: {X_test.shape[0]} 个样本")
    
    # 训练带早停的模型
    nn = EarlyStoppingNN(
        layer_sizes=[20, 50, 50, 3],
        activation='relu',
        output_activation='softmax',
        loss='cross_entropy',
        learning_rate=0.01,
        batch_size=32,
        max_epochs=500,
        patience=20,
        min_delta=1e-4,
        monitor='val_loss',
        mode='min',
        restore_best_weights=True,
        random_state=42,
        verbose=True
    )
    
    nn.fit(X_train, y_train, X_val, y_val)
    
    # 测试集评估
    test_pred = nn.predict(X_test)
    test_acc = np.mean(test_pred == y_test)
    
    print(f"\\n测试集准确率: {test_acc:.4f}")
    
    # 可视化训练过程
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(nn.train_loss_history) + 1)
    
    # 损失曲线
    axes[0].plot(epochs, nn.train_loss_history, 'b-', linewidth=2, label='训练损失')
    axes[0].plot(epochs, nn.val_loss_history, 'r-', linewidth=2, label='验证损失')
    axes[0].axvline(x=nn.best_epoch + 1, color='g', linestyle='--', 
                    linewidth=2, label=f'最佳epoch ({nn.best_epoch + 1})')
    if nn.stopped_epoch > 0:
        axes[0].axvline(x=nn.stopped_epoch + 1, color='orange', linestyle='--',
                        linewidth=2, label=f'早停epoch ({nn.stopped_epoch + 1})')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('损失')
    axes[0].set_title('早停法 - 损失曲线')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 准确率曲线
    axes[1].plot(epochs, nn.train_acc_history, 'b-', linewidth=2, label='训练准确率')
    axes[1].plot(epochs, nn.val_acc_history, 'r-', linewidth=2, label='验证准确率')
    axes[1].axvline(x=nn.best_epoch + 1, color='g', linestyle='--',
                    linewidth=2, label=f'最佳epoch ({nn.best_epoch + 1})')
    axes[1].axhline(y=test_acc, color='purple', linestyle=':',
                    linewidth=2, label=f'测试准确率 ({test_acc:.3f})')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('准确率')
    axes[1].set_title('早停法 - 准确率曲线')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fnn/early_stopping_basic.png', dpi=150, bbox_inches='tight')
    print("\\n图片已保存: fnn/early_stopping_basic.png")
    plt.close()


def demo_2_patience_comparison():
    """
    示例2: 不同patience值的对比
    """
    print("\\n" + "=" * 70)
    print("示例2: 不同Patience值的对比")
    print("=" * 70)
    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    X, y = make_classification(
        n_samples=800,
        n_features=20,
        n_informative=12,
        n_classes=2,
        random_state=42
    )
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    patience_values = [5, 10, 20, 50]
    results = []
    
    for patience in patience_values:
        print(f"\\n训练 patience={patience}...")
        
        nn = EarlyStoppingNN(
            layer_sizes=[20, 30, 2],
            activation='relu',
            output_activation='softmax',
            loss='cross_entropy',
            learning_rate=0.01,
            batch_size=32,
            max_epochs=200,
            patience=patience,
            monitor='val_loss',
            random_state=42,
            verbose=False
        )
        
        nn.fit(X_train, y_train, X_val, y_val)
        
        test_acc = np.mean(nn.predict(X_test) == y_test)
        
        results.append({
            'patience': patience,
            'model': nn,
            'test_acc': test_acc,
            'n_epochs': nn.n_epochs_,
            'best_epoch': nn.best_epoch + 1
        })
        
        print(f"  停止于第 {nn.n_epochs_} 轮")
        print(f"  最佳epoch: {nn.best_epoch + 1}")
        print(f"  测试准确率: {test_acc:.4f}")
    
    # 可视化对比
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = ['blue', 'orange', 'green', 'red']
    
    # 验证损失曲线
    for i, r in enumerate(results):
        epochs = range(1, len(r['model'].val_loss_history) + 1)
        axes[0, 0].plot(epochs, r['model'].val_loss_history,
                        color=colors[i], linewidth=2,
                        label=f"patience={r['patience']}")
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('验证损失')
    axes[0, 0].set_title('不同Patience的验证损失')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 训练轮数对比
    patience_list = [r['patience'] for r in results]
    n_epochs_list = [r['n_epochs'] for r in results]
    axes[0, 1].bar(range(len(patience_list)), n_epochs_list, color=colors)
    axes[0, 1].set_xticks(range(len(patience_list)))
    axes[0, 1].set_xticklabels([f"p={p}" for p in patience_list])
    axes[0, 1].set_ylabel('训练轮数')
    axes[0, 1].set_title('不同Patience的训练轮数')
    for i, v in enumerate(n_epochs_list):
        axes[0, 1].text(i, v + 2, str(v), ha='center')
    
    # 测试准确率对比
    test_accs = [r['test_acc'] for r in results]
    axes[1, 0].bar(range(len(patience_list)), test_accs, color=colors)
    axes[1, 0].set_xticks(range(len(patience_list)))
    axes[1, 0].set_xticklabels([f"p={p}" for p in patience_list])
    axes[1, 0].set_ylabel('测试准确率')
    axes[1, 0].set_title('不同Patience的测试准确率')
    axes[1, 0].set_ylim([0.7, 1.0])
    for i, v in enumerate(test_accs):
        axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    # 节省的训练轮数
    saved_epochs = [200 - r['n_epochs'] for r in results]
    axes[1, 1].bar(range(len(patience_list)), saved_epochs, color=colors)
    axes[1, 1].set_xticks(range(len(patience_list)))
    axes[1, 1].set_xticklabels([f"p={p}" for p in patience_list])
    axes[1, 1].set_ylabel('节省的轮数')
    axes[1, 1].set_title('早停节省的训练轮数')
    for i, v in enumerate(saved_epochs):
        axes[1, 1].text(i, v + 2, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig('fnn/early_stopping_patience.png', dpi=150, bbox_inches='tight')
    print("\\n图片已保存: fnn/early_stopping_patience.png")
    plt.close()


def demo_3_overfitting_prevention():
    """
    示例3: 早停法防止过拟合
    """
    print("\\n" + "=" * 70)
    print("示例3: 早停法防止过拟合")
    print("=" * 70)
    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # 生成容易过拟合的数据（样本少、特征多）
    X, y = make_classification(
        n_samples=300,
        n_features=50,
        n_informative=20,
        n_redundant=30,
        n_classes=3,
        random_state=42
    )
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    print(f"\\n数据集（容易过拟合）:")
    print(f"  训练集: {X_train.shape[0]} 个样本, {X_train.shape[1]} 个特征")
    print(f"  验证集: {X_val.shape[0]} 个样本")
    print(f"  测试集: {X_test.shape[0]} 个样本")
    
    # 无早停：训练到最大轮数
    print("\\n训练模型（无早停）...")
    nn_no_es = EarlyStoppingNN(
        layer_sizes=[50, 100, 50, 3],
        activation='relu',
        output_activation='softmax',
        learning_rate=0.01,
        batch_size=16,
        max_epochs=300,
        patience=1000,  # 设置很大，相当于不早停
        random_state=42,
        verbose=False
    )
    nn_no_es.fit(X_train, y_train, X_val, y_val)
    
    # 有早停
    print("训练模型（有早停）...")
    nn_with_es = EarlyStoppingNN(
        layer_sizes=[50, 100, 50, 3],
        activation='relu',
        output_activation='softmax',
        learning_rate=0.01,
        batch_size=16,
        max_epochs=300,
        patience=15,
        monitor='val_loss',
        random_state=42,
        verbose=True
    )
    nn_with_es.fit(X_train, y_train, X_val, y_val)
    
    # 评估
    test_acc_no_es = np.mean(nn_no_es.predict(X_test) == y_test)
    test_acc_with_es = np.mean(nn_with_es.predict(X_test) == y_test)
    
    print(f"\\n对比结果:")
    print(f"  无早停:")
    print(f"    训练轮数: {nn_no_es.n_epochs_}")
    print(f"    最终训练准确率: {nn_no_es.train_acc_history[-1]:.4f}")
    print(f"    最终验证准确率: {nn_no_es.val_acc_history[-1]:.4f}")
    print(f"    测试准确率: {test_acc_no_es:.4f}")
    print(f"\\n  有早停:")
    print(f"    训练轮数: {nn_with_es.n_epochs_}")
    print(f"    最终训练准确率: {nn_with_es.train_acc_history[-1]:.4f}")
    print(f"    最终验证准确率: {nn_with_es.val_acc_history[-1]:.4f}")
    print(f"    测试准确率: {test_acc_with_es:.4f}")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 无早停
    epochs_no_es = range(1, len(nn_no_es.train_loss_history) + 1)
    axes[0].plot(epochs_no_es, nn_no_es.train_loss_history,
                 'b-', linewidth=2, label='训练损失')
    axes[0].plot(epochs_no_es, nn_no_es.val_loss_history,
                 'r-', linewidth=2, label='验证损失')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('损失')
    axes[0].set_title(f'无早停 - 过拟合\\n(测试准确率: {test_acc_no_es:.3f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 有早停
    epochs_with_es = range(1, len(nn_with_es.train_loss_history) + 1)
    axes[1].plot(epochs_with_es, nn_with_es.train_loss_history,
                 'b-', linewidth=2, label='训练损失')
    axes[1].plot(epochs_with_es, nn_with_es.val_loss_history,
                 'r-', linewidth=2, label='验证损失')
    axes[1].axvline(x=nn_with_es.best_epoch + 1, color='g',
                    linestyle='--', linewidth=2, label=f'最佳epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('损失')
    axes[1].set_title(f'有早停 - 防止过拟合\\n(测试准确率: {test_acc_with_es:.3f})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fnn/early_stopping_overfitting.png', dpi=150, bbox_inches='tight')
    print("\\n图片已保存: fnn/early_stopping_overfitting.png")
    plt.close()


if __name__ == "__main__":
    print("\\n" + "=" * 70)
    print("前馈神经网络 - 早停法 (Early Stopping)")
    print("=" * 70)
    
    # 运行所有示例
    demo_1_early_stopping_basic()
    demo_2_patience_comparison()
    demo_3_overfitting_prevention()
    
    print("\\n" + "=" * 70)
    print("所有示例运行完成!")
    print("=" * 70)
    print("\\n早停法总结:")
    print("\\n核心思想:")
    print("  监控验证集性能，当性能不再提升时提前停止训练")
    
    print("\\n关键参数:")
    print("  - patience: 容忍多少个epoch无改进（通常5-20）")
    print("  - min_delta: 最小改进阈值（通常1e-4）")
    print("  - monitor: 监控指标（val_loss 或 val_acc）")
    print("  - restore_best: 是否恢复最佳权重")
    
    print("\\n优势:")
    print("  ✓ 有效防止过拟合")
    print("  ✓ 节省训练时间")
    print("  ✓ 自动确定训练轮数")
    print("  ✓ 简单易用")
    
    print("\\n最佳实践:")
    print("  1. 始终使用独立的验证集")
    print("  2. patience不要太小（避免提前停止）")
    print("  3. 恢复到最佳权重（restore_best_weights=True）")
    print("  4. 监控val_loss比val_acc更稳定")
    print("  5. 结合其他正则化技术（Dropout等）")
