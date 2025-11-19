"""
前馈神经网络 - 反向传播算法详解 (Backpropagation Algorithm)
================================================================

反向传播（Backpropagation, BP）算法是训练神经网络的核心算法，
它通过链式法则高效计算损失函数对每个参数的梯度。

本实现详细展示反向传播的每一步计算过程和数学推导。

理论基础
--------

1. **前向传播（Forward Propagation）**:

   从输入层到输出层，逐层计算：
   
   第 l 层的计算：
   $$z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}$$
   $$a^{[l]} = g^{[l]}(z^{[l]})$$
   
   其中：
   - $a^{[0]} = X$ （输入）
   - $a^{[L]}$ 是最终输出
   - $g^{[l]}$ 是激活函数

2. **损失函数（Loss Function）**:

   **回归 - 均方误差**:
   $$J = \\frac{1}{2m} \\sum_{i=1}^m ||a^{[L],(i)} - y^{(i)}||^2$$
   
   **分类 - 交叉熵**:
   $$J = -\\frac{1}{m} \\sum_{i=1}^m \\sum_{k=1}^K y_k^{(i)} \\log(a_k^{[L],(i)})$$

3. **反向传播核心思想**:

   使用链式法则，从输出层向输入层反向计算梯度：
   
   $$\\frac{\\partial J}{\\partial W^{[l]}} = \\frac{\\partial J}{\\partial z^{[l]}} \\frac{\\partial z^{[l]}}{\\partial W^{[l]}}$$

4. **误差项 $\\delta^{[l]}$**:

   定义误差项（error term）：
   $$\\delta^{[l]} = \\frac{\\partial J}{\\partial z^{[l]}}$$
   
   这是损失函数对第 l 层线性组合的偏导数。

5. **输出层误差 $\\delta^{[L]}$**:

   **一般情况**:
   $$\\delta^{[L]} = \\frac{\\partial J}{\\partial a^{[L]}} \\odot g'^{[L]}(z^{[L]})$$
   
   **交叉熵 + Softmax（特殊简化）**:
   $$\\delta^{[L]} = a^{[L]} - y$$
   
   **MSE + Sigmoid**:
   $$\\delta^{[L]} = (a^{[L]} - y) \\odot \\sigma'(z^{[L]})$$

6. **隐藏层误差递推**:

   从后向前传播误差：
   $$\\delta^{[l]} = (W^{[l+1]})^T \\delta^{[l+1]} \\odot g'^{[l]}(z^{[l]})$$
   
   解释：
   - $(W^{[l+1]})^T \\delta^{[l+1]}$: 从下一层反向传播误差
   - $\\odot g'^{[l]}(z^{[l]})$: 乘以激活函数的导数

7. **梯度计算**:

   有了误差项，可以直接计算梯度：
   
   **权重梯度**:
   $$\\frac{\\partial J}{\\partial W^{[l]}} = \\frac{1}{m} \\delta^{[l]} (a^{[l-1]})^T$$
   
   **偏置梯度**:
   $$\\frac{\\partial J}{\\partial b^{[l]}} = \\frac{1}{m} \\sum_{i=1}^m \\delta^{[l],(i)}$$

8. **链式法则推导**:

   完整的链式法则展开：
   
   $$\\frac{\\partial J}{\\partial W^{[l]}} = \\frac{\\partial J}{\\partial a^{[L]}} \\frac{\\partial a^{[L]}}{\\partial z^{[L]}} 
   \\frac{\\partial z^{[L]}}{\\partial a^{[L-1]}} \\cdots \\frac{\\partial a^{[l]}}{\\partial z^{[l]}} \\frac{\\partial z^{[l]}}{\\partial W^{[l]}}$$

9. **矩阵维度**:

   关键维度关系：
   - $W^{[l]}$: $(n^{[l]}, n^{[l-1]})$
   - $b^{[l]}$: $(n^{[l]}, 1)$
   - $z^{[l]}$: $(n^{[l]}, m)$
   - $a^{[l]}$: $(n^{[l]}, m)$
   - $\\delta^{[l]}$: $(n^{[l]}, m)$
   - $\\frac{\\partial J}{\\partial W^{[l]}}$: $(n^{[l]}, n^{[l-1]})$

算法步骤
--------

```
输入：训练数据 (X, y)，网络结构
输出：所有层的梯度 dW, db

【前向传播】
1. a^[0] = X
2. for l = 1 to L:
       z^[l] = W^[l] * a^[l-1] + b^[l]
       a^[l] = g^[l](z^[l])
       保存 z^[l], a^[l] 到缓存

【计算损失】
3. J = Loss(a^[L], y)

【反向传播】
4. 计算输出层误差:
   δ^[L] = (a^[L] - y) ⊙ g'^[L](z^[L])

5. for l = L-1 down to 1:
       δ^[l] = (W^[l+1])^T * δ^[l+1] ⊙ g'^[l](z^[l])

【计算梯度】
6. for l = 1 to L:
       dW^[l] = (1/m) * δ^[l] * (a^[l-1])^T
       db^[l] = (1/m) * sum(δ^[l], axis=1)

7. 返回 dW, db
```

日期: 2025-01-19
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import time

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class BackpropagationNN:
    """
    前馈神经网络 - 详细的反向传播实现
    
    专注于展示反向传播的每一步计算过程
    
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
    
    debug : bool
        是否显示详细的反向传播计算过程
    """
    
    def __init__(
        self,
        layer_sizes: List[int],
        activation: str = 'sigmoid',
        output_activation: str = 'sigmoid',
        loss: str = 'mse',
        learning_rate: float = 0.1,
        max_epochs: int = 1000,
        random_state: Optional[int] = None,
        debug: bool = False,
        verbose: bool = False
    ):
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)
        self.activation = activation
        self.output_activation = output_activation
        self.loss = loss
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.debug = debug
        self.verbose = verbose
        
        self.weights = []
        self.biases = []
        
        # 用于调试的详细信息
        self.forward_cache = {}
        self.backward_cache = {}
        
        # 训练历史
        self.train_loss_history = []
        self.gradient_norms = []  # 梯度范数历史
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid: σ(z) = 1 / (1 + e^(-z))"""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    
    def _sigmoid_derivative(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid导数: σ'(z) = σ(z)(1 - σ(z))"""
        s = self._sigmoid(z)
        return s * (1 - s)
    
    def _relu(self, z: np.ndarray) -> np.ndarray:
        """ReLU: max(0, z)"""
        return np.maximum(0, z)
    
    def _relu_derivative(self, z: np.ndarray) -> np.ndarray:
        """ReLU导数"""
        return (z > 0).astype(float)
    
    def _tanh(self, z: np.ndarray) -> np.ndarray:
        """Tanh"""
        return np.tanh(z)
    
    def _tanh_derivative(self, z: np.ndarray) -> np.ndarray:
        """Tanh导数: 1 - tanh^2(z)"""
        t = np.tanh(z)
        return 1 - t ** 2
    
    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """Softmax（数值稳定版本）"""
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
            raise ValueError(f"未知激活函数: {activation}")
    
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
            raise ValueError(f"未知激活函数: {activation}")
    
    def _initialize_parameters(self):
        """初始化权重和偏置"""
        self.weights = []
        self.biases = []
        
        for i in range(self.n_layers - 1):
            n_in = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]
            
            # Xavier初始化
            scale = np.sqrt(1.0 / n_in)
            W = np.random.randn(n_out, n_in) * scale
            b = np.zeros((n_out, 1))
            
            self.weights.append(W)
            self.biases.append(b)
    
    def forward_propagation(self, X: np.ndarray, save_cache: bool = True) -> Tuple[List, List]:
        """
        前向传播（详细版本）
        
        参数
        ----
        X : np.ndarray
            输入数据 (n_features, m)
        save_cache : bool
            是否保存中间结果
        
        返回
        ----
        Z_list : List[np.ndarray]
            每层的线性组合 z^[l]
        A_list : List[np.ndarray]
            每层的激活值 a^[l]
        """
        A = X
        Z_list = []
        A_list = [A]  # a^[0] = X
        
        if self.debug and save_cache:
            print("\\n【前向传播开始】")
            print(f"输入 a^[0] 形状: {A.shape}")
        
        # 隐藏层
        for l in range(self.n_layers - 2):
            # 线性组合: z^[l] = W^[l] * a^[l-1] + b^[l]
            Z = self.weights[l] @ A + self.biases[l]
            
            # 激活: a^[l] = g(z^[l])
            A = self._apply_activation(Z, self.activation)
            
            Z_list.append(Z)
            A_list.append(A)
            
            if self.debug and save_cache:
                print(f"\\n第 {l+1} 层（隐藏层）:")
                print(f"  W^[{l+1}] 形状: {self.weights[l].shape}")
                print(f"  b^[{l+1}] 形状: {self.biases[l].shape}")
                print(f"  z^[{l+1}] 形状: {Z.shape}")
                print(f"  z^[{l+1}] 范围: [{Z.min():.4f}, {Z.max():.4f}]")
                print(f"  a^[{l+1}] 形状: {A.shape}")
                print(f"  a^[{l+1}] 范围: [{A.min():.4f}, {A.max():.4f}]")
        
        # 输出层
        Z = self.weights[-1] @ A + self.biases[-1]
        A = self._apply_activation(Z, self.output_activation)
        
        Z_list.append(Z)
        A_list.append(A)
        
        if self.debug and save_cache:
            print(f"\\n第 {self.n_layers} 层（输出层）:")
            print(f"  W^[{self.n_layers}] 形状: {self.weights[-1].shape}")
            print(f"  z^[{self.n_layers}] 形状: {Z.shape}")
            print(f"  a^[{self.n_layers}] (输出) 形状: {A.shape}")
            print(f"  输出范围: [{A.min():.4f}, {A.max():.4f}]")
        
        return Z_list, A_list
    
    def compute_loss(self, Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
        """
        计算损失函数
        
        参数
        ----
        Y_true : np.ndarray
            真实标签 (n_output, m)
        Y_pred : np.ndarray
            预测值 (n_output, m)
        
        返回
        ----
        loss : float
            损失值
        """
        m = Y_true.shape[1]
        
        if self.loss == 'mse':
            # 均方误差: J = (1/2m) * Σ||y - ŷ||^2
            loss = np.sum((Y_true - Y_pred) ** 2) / (2 * m)
        elif self.loss == 'cross_entropy':
            # 交叉熵: J = -(1/m) * Σ y*log(ŷ)
            Y_pred = np.clip(Y_pred, 1e-10, 1 - 1e-10)
            loss = -np.sum(Y_true * np.log(Y_pred)) / m
        else:
            raise ValueError(f"未知损失函数: {self.loss}")
        
        return loss
    
    def backward_propagation(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z_list: List[np.ndarray],
        A_list: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        反向传播（详细版本）
        
        参数
        ----
        X : np.ndarray
            输入数据
        Y : np.ndarray
            真实标签
        Z_list : List[np.ndarray]
            前向传播的线性组合
        A_list : List[np.ndarray]
            前向传播的激活值
        
        返回
        ----
        dW_list : List[np.ndarray]
            权重梯度
        db_list : List[np.ndarray]
            偏置梯度
        """
        m = X.shape[1]
        L = self.n_layers - 1  # 层数（不包括输入层）
        
        dW_list = [None] * L
        db_list = [None] * L
        delta_list = [None] * L  # 保存每层的误差项
        
        if self.debug:
            print("\\n" + "=" * 70)
            print("【反向传播开始】")
            print("=" * 70)
        
        # ====================================================================
        # 步骤1: 计算输出层误差 δ^[L]
        # ====================================================================
        
        if self.debug:
            print("\\n步骤1: 计算输出层误差 δ^[L]")
            print("-" * 70)
        
        if self.loss == 'cross_entropy' and self.output_activation == 'softmax':
            # 特殊情况：交叉熵 + Softmax
            # δ^[L] = a^[L] - y
            delta = A_list[-1] - Y
            
            if self.debug:
                print("使用简化公式（交叉熵 + Softmax）:")
                print("  δ^[L] = a^[L] - y")
        else:
            # 一般情况：δ^[L] = (a^[L] - y) ⊙ g'(z^[L])
            dA = A_list[-1] - Y  # ∂J/∂a^[L]
            dZ = self._apply_activation_derivative(Z_list[-1], self.output_activation)
            delta = dA * dZ
            
            if self.debug:
                print("使用一般公式:")
                print("  ∂J/∂a^[L] = a^[L] - y")
                print(f"  ∂J/∂a^[L] 形状: {dA.shape}, 范围: [{dA.min():.6f}, {dA.max():.6f}]")
                print(f"  g'(z^[L]) 形状: {dZ.shape}, 范围: [{dZ.min():.6f}, {dZ.max():.6f}]")
                print("  δ^[L] = (∂J/∂a^[L]) ⊙ g'(z^[L])")
        
        if self.debug:
            print(f"  δ^[L] 形状: {delta.shape}")
            print(f"  δ^[L] 范围: [{delta.min():.6f}, {delta.max():.6f}]")
            print(f"  δ^[L] 范数: {np.linalg.norm(delta):.6f}")
        
        delta_list[-1] = delta
        
        # ====================================================================
        # 步骤2: 计算输出层梯度 dW^[L], db^[L]
        # ====================================================================
        
        if self.debug:
            print("\\n步骤2: 计算输出层梯度")
            print("-" * 70)
        
        # dW^[L] = (1/m) * δ^[L] * (a^[L-1])^T
        dW_list[-1] = (delta @ A_list[-2].T) / m
        
        # db^[L] = (1/m) * Σ δ^[L]
        db_list[-1] = np.sum(delta, axis=1, keepdims=True) / m
        
        if self.debug:
            print(f"  dW^[L] = (1/m) * δ^[L] @ (a^[L-1])^T")
            print(f"  dW^[L] 形状: {dW_list[-1].shape}")
            print(f"  dW^[L] 范围: [{dW_list[-1].min():.6f}, {dW_list[-1].max():.6f}]")
            print(f"  dW^[L] 范数: {np.linalg.norm(dW_list[-1]):.6f}")
            print(f"\\n  db^[L] = (1/m) * sum(δ^[L])")
            print(f"  db^[L] 形状: {db_list[-1].shape}")
            print(f"  db^[L] 范围: [{db_list[-1].min():.6f}, {db_list[-1].max():.6f}]")
        
        # ====================================================================
        # 步骤3: 反向传播误差到隐藏层
        # ====================================================================
        
        if self.debug:
            print("\\n步骤3: 反向传播到隐藏层")
            print("-" * 70)
        
        for l in range(L - 2, -1, -1):
            if self.debug:
                print(f"\\n处理第 {l+1} 层:")
            
            # δ^[l] = (W^[l+1])^T * δ^[l+1] ⊙ g'(z^[l])
            
            # 第一项：从下一层反向传播
            delta_from_next = self.weights[l + 1].T @ delta
            
            # 第二项：激活函数导数
            activation_derivative = self._apply_activation_derivative(
                Z_list[l], self.activation
            )
            
            # 误差项
            delta = delta_from_next * activation_derivative
            
            if self.debug:
                print(f"  (W^[{l+2}])^T 形状: {self.weights[l + 1].T.shape}")
                print(f"  δ^[{l+2}] 形状: {delta_list[l+1].shape}")
                print(f"  (W^[{l+2}])^T @ δ^[{l+2}] 形状: {delta_from_next.shape}")
                print(f"  (W^[{l+2}])^T @ δ^[{l+2}] 范围: [{delta_from_next.min():.6f}, {delta_from_next.max():.6f}]")
                print(f"\\n  g'(z^[{l+1}]) 形状: {activation_derivative.shape}")
                print(f"  g'(z^[{l+1}]) 范围: [{activation_derivative.min():.6f}, {activation_derivative.max():.6f}]")
                print(f"\\n  δ^[{l+1}] = (W^[{l+2}])^T @ δ^[{l+2}] ⊙ g'(z^[{l+1}])")
                print(f"  δ^[{l+1}] 形状: {delta.shape}")
                print(f"  δ^[{l+1}] 范围: [{delta.min():.6f}, {delta.max():.6f}]")
                print(f"  δ^[{l+1}] 范数: {np.linalg.norm(delta):.6f}")
            
            delta_list[l] = delta
            
            # 计算梯度
            dW_list[l] = (delta @ A_list[l].T) / m
            db_list[l] = np.sum(delta, axis=1, keepdims=True) / m
            
            if self.debug:
                print(f"\\n  dW^[{l+1}] 形状: {dW_list[l].shape}")
                print(f"  dW^[{l+1}] 范数: {np.linalg.norm(dW_list[l]):.6f}")
                print(f"  db^[{l+1}] 形状: {db_list[l].shape}")
        
        # ====================================================================
        # 步骤4: 梯度汇总
        # ====================================================================
        
        if self.debug:
            print("\\n" + "=" * 70)
            print("梯度汇总:")
            print("=" * 70)
            total_gradient_norm = 0
            for l in range(L):
                dW_norm = np.linalg.norm(dW_list[l])
                db_norm = np.linalg.norm(db_list[l])
                total_gradient_norm += dW_norm ** 2 + db_norm ** 2
                print(f"第 {l+1} 层:")
                print(f"  ||dW^[{l+1}]||: {dW_norm:.6f}")
                print(f"  ||db^[{l+1}]||: {db_norm:.6f}")
            
            total_gradient_norm = np.sqrt(total_gradient_norm)
            print(f"\\n总梯度范数: {total_gradient_norm:.6f}")
            print("=" * 70)
        
        return dW_list, db_list
    
    def update_parameters(self, dW_list: List[np.ndarray], db_list: List[np.ndarray]):
        """
        更新参数
        
        W^[l] := W^[l] - α * dW^[l]
        b^[l] := b^[l] - α * db^[l]
        """
        for l in range(self.n_layers - 1):
            self.weights[l] -= self.learning_rate * dW_list[l]
            self.biases[l] -= self.learning_rate * db_list[l]
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BackpropagationNN':
        """
        训练网络
        
        参数
        ----
        X : np.ndarray
            训练数据 (n_samples, n_features)
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
                Y = y.reshape(1, -1)
            else:
                Y = np.zeros((n_classes, len(y)))
                Y[y, np.arange(len(y))] = 1
        else:
            Y = y.T
        
        m = X.shape[1]
        
        if self.verbose:
            print(f"\\n反向传播训练开始:")
            print(f"  样本数: {m}")
            print(f"  网络结构: {' -> '.join(map(str, self.layer_sizes))}")
            print(f"  学习率: {self.learning_rate}")
            print(f"  调试模式: {self.debug}")
        
        # 初始化参数
        self._initialize_parameters()
        
        # 训练循环
        for epoch in range(self.max_epochs):
            # 前向传播
            Z_list, A_list = self.forward_propagation(X, save_cache=(epoch == 0 and self.debug))
            Y_pred = A_list[-1]
            
            # 计算损失
            loss = self.compute_loss(Y, Y_pred)
            self.train_loss_history.append(loss)
            
            # 反向传播
            dW_list, db_list = self.backward_propagation(X, Y, Z_list, A_list)
            
            # 计算总梯度范数
            gradient_norm = sum(np.linalg.norm(dW) ** 2 + np.linalg.norm(db) ** 2 
                               for dW, db in zip(dW_list, db_list))
            gradient_norm = np.sqrt(gradient_norm)
            self.gradient_norms.append(gradient_norm)
            
            # 更新参数
            self.update_parameters(dW_list, db_list)
            
            # 只在第一轮显示详细的反向传播过程
            if epoch == 0 and self.debug:
                self.debug = False
            
            # 显示进度
            if self.verbose and (epoch + 1) % 100 == 0:
                acc = np.mean(np.argmax(Y_pred, axis=0) == np.argmax(Y, axis=0)) if Y.shape[0] > 1 else \
                      np.mean((Y_pred > 0.5) == Y)
                print(f"  Epoch {epoch + 1}/{self.max_epochs}: "
                      f"Loss={loss:.6f}, Acc={acc:.4f}, "
                      f"||∇||={gradient_norm:.6f}")
        
        if self.verbose:
            print(f"\\n训练完成!")
            print(f"  最终损失: {self.train_loss_history[-1]:.6f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        X = X.T
        _, A_list = self.forward_propagation(X, save_cache=False)
        Y_pred = A_list[-1]
        
        if self.layer_sizes[-1] > 1:
            return np.argmax(Y_pred, axis=0)
        else:
            return (Y_pred > 0.5).astype(int).ravel()


# ============================================================================
# 示例应用
# ============================================================================

def demo_1_simple_network():
    """
    示例1: 简单网络的反向传播详解
    """
    print("=" * 70)
    print("示例1: 简单网络的反向传播详解")
    print("=" * 70)
    
    # 简单数据：2个样本
    X = np.array([[0, 1],
                  [1, 0]]).T  # 2个特征，2个样本
    y = np.array([1, 0])
    
    print(f"\\n训练数据:")
    print(f"  X 形状: {X.shape}")
    print(f"  X:\\n{X.T}")
    print(f"  y: {y}")
    
    # 小型网络: 2 -> 3 -> 1
    print(f"\\n网络结构: 2 (输入) -> 3 (隐藏) -> 1 (输出)")
    
    nn = BackpropagationNN(
        layer_sizes=[2, 3, 1],
        activation='sigmoid',
        output_activation='sigmoid',
        loss='mse',
        learning_rate=0.5,
        max_epochs=1,  # 只训练1轮以展示详细过程
        random_state=42,
        debug=True,  # 开启调试模式
        verbose=False
    )
    
    nn.fit(X.T, y)
    
    print("\\n提示: 以上展示了第一轮训练的完整反向传播过程")


def demo_2_xor_backprop():
    """
    示例2: XOR问题的反向传播训练
    """
    print("\\n" + "=" * 70)
    print("示例2: XOR问题 - 反向传播训练过程")
    print("=" * 70)
    
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    
    print(f"\\nXOR问题:")
    for i in range(len(X)):
        print(f"  输入: {X[i]}, 输出: {y[i]}")
    
    # 训练网络
    nn = BackpropagationNN(
        layer_sizes=[2, 4, 1],
        activation='tanh',
        output_activation='sigmoid',
        loss='mse',
        learning_rate=0.5,
        max_epochs=2000,
        random_state=42,
        debug=False,
        verbose=True
    )
    
    nn.fit(X, y)
    
    # 测试
    predictions = nn.predict(X)
    
    print(f"\\n预测结果:")
    for i in range(len(X)):
        print(f"  输入: {X[i]}, 真实: {y[i]}, 预测: {predictions[i]}")
    
    # 可视化训练过程
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 损失曲线
    axes[0].plot(nn.train_loss_history, 'b-', linewidth=2)
    axes[0].set_xlabel('训练轮数')
    axes[0].set_ylabel('MSE损失')
    axes[0].set_title('反向传播 - 损失下降曲线')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    
    # 梯度范数
    axes[1].plot(nn.gradient_norms, 'r-', linewidth=2)
    axes[1].set_xlabel('训练轮数')
    axes[1].set_ylabel('梯度范数 ||∇J||')
    axes[1].set_title('反向传播 - 梯度范数变化')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fnn/backpropagation_xor.png', dpi=150, bbox_inches='tight')
    print("\\n图片已保存: fnn/backpropagation_xor.png")
    plt.close()


def demo_3_gradient_checking():
    """
    示例3: 梯度检查（验证反向传播正确性）
    """
    print("\\n" + "=" * 70)
    print("示例3: 梯度检查 - 验证反向传播")
    print("=" * 70)
    
    # 小数据集
    np.random.seed(42)
    X = np.random.randn(5, 2)  # 5个样本，2个特征
    y = np.array([0, 1, 0, 1, 0])
    
    nn = BackpropagationNN(
        layer_sizes=[2, 3, 2],
        activation='sigmoid',
        output_activation='softmax',
        loss='cross_entropy',
        learning_rate=0.1,
        max_epochs=1,
        random_state=42,
        debug=False,
        verbose=False
    )
    
    # 初始化参数
    nn._initialize_parameters()
    
    # 准备数据
    X_t = X.T
    Y = np.zeros((2, len(y)))
    Y[y, np.arange(len(y))] = 1
    
    # 前向传播
    Z_list, A_list = nn.forward_propagation(X_t, save_cache=False)
    
    # 反向传播计算梯度
    dW_list, db_list = nn.backward_propagation(X_t, Y, Z_list, A_list)
    
    # 数值梯度检查
    print("\\n梯度检查（数值梯度 vs 反向传播梯度）:")
    print("-" * 70)
    
    epsilon = 1e-7
    
    # 检查第一层权重的几个梯度
    layer_idx = 0
    print(f"\\n检查第 {layer_idx + 1} 层权重梯度:")
    
    for i in range(min(3, nn.weights[layer_idx].shape[0])):
        for j in range(min(3, nn.weights[layer_idx].shape[1])):
            # 保存原始值
            original = nn.weights[layer_idx][i, j]
            
            # J(W + ε)
            nn.weights[layer_idx][i, j] = original + epsilon
            _, A_plus = nn.forward_propagation(X_t, save_cache=False)
            loss_plus = nn.compute_loss(Y, A_plus[-1])
            
            # J(W - ε)
            nn.weights[layer_idx][i, j] = original - epsilon
            _, A_minus = nn.forward_propagation(X_t, save_cache=False)
            loss_minus = nn.compute_loss(Y, A_minus[-1])
            
            # 数值梯度
            numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)
            
            # 反向传播梯度
            backprop_grad = dW_list[layer_idx][i, j]
            
            # 相对误差
            relative_error = abs(numerical_grad - backprop_grad) / \
                           (abs(numerical_grad) + abs(backprop_grad) + 1e-10)
            
            # 恢复原始值
            nn.weights[layer_idx][i, j] = original
            
            status = "✓" if relative_error < 1e-5 else "✗"
            print(f"  W[{i},{j}]: 数值={numerical_grad:.8f}, "
                  f"反向传播={backprop_grad:.8f}, "
                  f"相对误差={relative_error:.2e} {status}")
    
    print("\\n说明: 相对误差 < 1e-5 表示梯度计算正确")


def demo_4_vanishing_gradient():
    """
    示例4: 梯度消失问题演示
    """
    print("\\n" + "=" * 70)
    print("示例4: 梯度消失/爆炸问题")
    print("=" * 70)
    
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=100, n_features=10, 
                               n_informative=8, n_redundant=2,
                               n_classes=2, random_state=42)
    
    configs = [
        {'activation': 'sigmoid', 'layers': [10, 50, 50, 2], 'name': 'Sigmoid (深层)'},
        {'activation': 'tanh', 'layers': [10, 50, 50, 2], 'name': 'Tanh (深层)'},
        {'activation': 'relu', 'layers': [10, 50, 50, 2], 'name': 'ReLU (深层)'},
    ]
    
    results = []
    
    for config in configs:
        print(f"\\n训练 {config['name']}...")
        
        nn = BackpropagationNN(
            layer_sizes=config['layers'],
            activation=config['activation'],
            output_activation='softmax',
            loss='cross_entropy',
            learning_rate=0.01,
            max_epochs=200,
            random_state=42,
            debug=False,
            verbose=False
        )
        
        nn.fit(X, y)
        
        results.append({
            'name': config['name'],
            'model': nn
        })
    
    # 可视化梯度范数
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for r in results:
        axes[0].plot(r['model'].train_loss_history, label=r['name'], linewidth=2)
    axes[0].set_xlabel('训练轮数')
    axes[0].set_ylabel('损失')
    axes[0].set_title('不同激活函数的损失曲线')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    for r in results:
        axes[1].plot(r['model'].gradient_norms, label=r['name'], linewidth=2, alpha=0.7)
    axes[1].set_xlabel('训练轮数')
    axes[1].set_ylabel('梯度范数')
    axes[1].set_title('不同激活函数的梯度范数')
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fnn/backpropagation_gradient_problem.png', dpi=150, bbox_inches='tight')
    print("\\n图片已保存: fnn/backpropagation_gradient_problem.png")
    plt.close()
    
    print("\\n结论:")
    print("  - Sigmoid: 深层网络容易梯度消失")
    print("  - Tanh: 比Sigmoid好，但仍有梯度消失")
    print("  - ReLU: 有效缓解梯度消失问题")


if __name__ == "__main__":
    print("\\n" + "=" * 70)
    print("前馈神经网络 - 反向传播算法详解")
    print("=" * 70)
    
    # 运行所有示例
    demo_1_simple_network()
    demo_2_xor_backprop()
    demo_3_gradient_checking()
    demo_4_vanishing_gradient()
    
    print("\\n" + "=" * 70)
    print("所有示例运行完成!")
    print("=" * 70)
    print("\\n反向传播核心步骤:")
    print("1. 前向传播: 计算所有层的 z^[l] 和 a^[l]")
    print("2. 输出层误差: δ^[L] = (a^[L] - y) ⊙ g'(z^[L])")
    print("3. 反向传播误差: δ^[l] = (W^[l+1])^T δ^[l+1] ⊙ g'(z^[l])")
    print("4. 计算梯度: dW^[l] = (1/m) δ^[l] (a^[l-1])^T")
    print("5. 更新参数: W^[l] := W^[l] - α dW^[l]")
    print("\\n关键技巧:")
    print("✓ 使用链式法则高效计算梯度")
    print("✓ 缓存前向传播的中间结果")
    print("✓ 从后向前逐层传播误差")
    print("✓ 使用数值梯度检查验证正确性")
    print("✓ 注意梯度消失/爆炸问题")
