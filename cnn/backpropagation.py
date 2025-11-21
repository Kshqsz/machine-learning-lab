"""
CNN - 反向传播实现 (NumPy)
============================

这是一个教学用途的卷积神经网络（CNN）实现，包含：
- 卷积层（forward/backward）
- ReLU 激活（forward/backward）
- 最大池化（forward/backward）
- 展平与全连接层（forward/backward）
- 交叉熵损失 + softmax
- 数值梯度检查
- 用 `sklearn.datasets.load_digits` 做小样例演示

注意：这是纯 NumPy 教学实现，训练效率远低于深度学习框架。
"""

import numpy as np
from typing import Tuple, Dict

# ----------------------------- 基本操作 -----------------------------

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_loss(probs: np.ndarray, y_true: np.ndarray) -> float:
    m = probs.shape[0]
    eps = 1e-12
    clipped = np.clip(probs, eps, 1 - eps)
    log_likelihood = -np.log(clipped[np.arange(m), y_true])
    return np.sum(log_likelihood) / m


def to_one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    m = y.shape[0]
    one = np.zeros((m, num_classes))
    one[np.arange(m), y] = 1
    return one


# ----------------------------- 卷积层 (简单实现) -----------------------------
# Inputs are in shape (N, C, H, W)
# Filters shape: (F, C, HH, WW)

class ConvLayer:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # Xavier init
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.b = np.zeros((out_channels,))
        # grads
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        # cache
        self.cache = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        N, C, H, W = X.shape
        F, _, HH, WW = self.W.shape
        pad = self.padding
        stride = self.stride
        H_out = 1 + (H + 2 * pad - HH) // stride
        W_out = 1 + (W + 2 * pad - WW) // stride
        out = np.zeros((N, F, H_out, W_out))

        X_padded = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')

        for n in range(N):
            for f in range(F):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * stride
                        w_start = j * stride
                        window = X_padded[n, :, h_start:h_start + HH, w_start:w_start + WW]
                        out[n, f, i, j] = np.sum(window * self.W[f]) + self.b[f]

        self.cache = (X, X_padded, out.shape)
        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        X, X_padded, out_shape = self.cache
        N, C, H, W = X.shape
        F, _, HH, WW = self.W.shape
        pad = self.padding
        stride = self.stride
        _, _, H_out, W_out = d_out.shape

        dX_padded = np.zeros_like(X_padded)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        for n in range(N):
            for f in range(F):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * stride
                        w_start = j * stride
                        window = X_padded[n, :, h_start:h_start + HH, w_start:w_start + WW]
                        self.dW[f] += d_out[n, f, i, j] * window
                        dX_padded[n, :, h_start:h_start + HH, w_start:w_start + WW] += d_out[n, f, i, j] * self.W[f]
                self.db[f] += np.sum(d_out[n, f])

        # remove padding
        if pad > 0:
            dX = dX_padded[:, :, pad:-pad, pad:-pad]
        else:
            dX = dX_padded

        # normalize gradients by batch size
        self.dW /= N
        self.db /= N
        return dX


# ----------------------------- ReLU -----------------------------

class ReLU:
    def __init__(self):
        self.cache = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        out = np.maximum(0, X)
        self.cache = X
        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        X = self.cache
        dX = d_out * (X > 0)
        return dX


# ----------------------------- 最大池化 -----------------------------

class MaxPool:
    def __init__(self, pool_size: int = 2, stride: int = 2):
        self.pool_size = pool_size
        self.stride = stride
        self.cache = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        N, C, H, W = X.shape
        PH = self.pool_size
        PW = self.pool_size
        stride = self.stride
        H_out = 1 + (H - PH) // stride
        W_out = 1 + (W - PW) // stride

        out = np.zeros((N, C, H_out, W_out))
        self.cache = {'mask': [], 'X_shape': X.shape}

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * stride
                        w_start = j * stride
                        window = X[n, c, h_start:h_start + PH, w_start:w_start + PW]
                        out[n, c, i, j] = np.max(window)
                        # create mask for backward
                        mask = (window == np.max(window)).astype(float)
                        self.cache['mask'].append((n, c, i, j, mask, (h_start, w_start)))
        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        N, C, H, W = self.cache['X_shape']
        PH = self.pool_size
        PW = self.pool_size
        stride = self.stride
        _, _, H_out, W_out = d_out.shape

        dX = np.zeros(self.cache['X_shape'])
        idx = 0
        for (n, c, i, j, mask, (h_start, w_start)) in self.cache['mask']:
            dX[n, c, h_start:h_start + PH, w_start:w_start + PW] += mask * d_out[n, c, i, j]
            idx += 1
        return dX


# ----------------------------- 全连接层 -----------------------------

class Dense:
    def __init__(self, in_dim: int, out_dim: int):
        scale = np.sqrt(2.0 / in_dim)
        self.W = np.random.randn(in_dim, out_dim) * scale
        self.b = np.zeros((1, out_dim))
        self.cache = None
        self.dW = None
        self.db = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        # X shape (N, D)
        out = X @ self.W + self.b
        self.cache = X
        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        X = self.cache
        N = X.shape[0]
        self.dW = X.T @ d_out / N
        self.db = np.sum(d_out, axis=0, keepdims=True) / N
        dX = d_out @ self.W.T
        return dX


# ----------------------------- CNN 模型 -----------------------------

class SimpleCNN:
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int, lr: float = 0.01):
        # input_shape = (C, H, W)
        C, H, W = input_shape
        # A minimal CNN: Conv -> ReLU -> Pool -> Flatten -> Dense
        self.conv = ConvLayer(in_channels=C, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.relu = ReLU()
        self.pool = MaxPool(pool_size=2, stride=2)
        # compute flattened size
        H2 = (H + 2 * self.conv.padding - self.conv.kernel_size) // self.conv.stride + 1
        H2 = H2 // 2  # after pool
        W2 = (W + 2 * self.conv.padding - self.conv.kernel_size) // self.conv.stride + 1
        W2 = W2 // 2
        D = 8 * H2 * W2
        self.fc = Dense(D, num_classes)
        self.lr = lr

    def forward(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        # X shape (N, C, H, W)
        cache = {}
        z1 = self.conv.forward(X)
        a1 = self.relu.forward(z1)
        p1 = self.pool.forward(a1)
        N = X.shape[0]
        flat = p1.reshape(N, -1)
        logits = self.fc.forward(flat)
        probs = softmax(logits)
        cache['X'] = X
        cache['z1'] = z1
        cache['a1'] = a1
        cache['p1'] = p1
        cache['flat'] = flat
        cache['logits'] = logits
        cache['probs'] = probs
        return cache

    def backward(self, cache: Dict, y: np.ndarray) -> None:
        # y: shape (N,) integer labels
        N = y.shape[0]
        probs = cache['probs']
        probs[np.arange(N), y] -= 1
        dlogits = probs / N  # gradient of loss wrt logits

        dflat = self.fc.backward(dlogits)
        dp1 = dflat.reshape(cache['p1'].shape)
        da1 = self.pool.backward(dp1)
        dz1 = self.relu.backward(da1)
        dX = self.conv.backward(dz1)

        # update parameters (SGD)
        # conv grads: self.conv.dW, self.conv.db
        self.conv.W -= self.lr * self.conv.dW
        self.conv.b -= self.lr * self.conv.db
        # fc grads
        self.fc.W -= self.lr * self.fc.dW
        self.fc.b -= self.lr * self.fc.db

    def predict(self, X: np.ndarray) -> np.ndarray:
        cache = self.forward(X)
        logits = cache['logits']
        return np.argmax(logits, axis=1)

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        cache = self.forward(X)
        probs = cache['probs']
        return cross_entropy_loss(probs, y)


# ----------------------------- 数值梯度检查 -----------------------------

def numeric_gradient_check(model: SimpleCNN, X: np.ndarray, y: np.ndarray, eps: float = 1e-5) -> None:
    """对部分参数进行数值梯度检查（Conv layer weights 和 FC weights）"""
    # check conv.W
    analytic_W = model.conv.dW.copy()
    approx_W = np.zeros_like(model.conv.W)

    print("开始数值梯度检查（卷积层）...")
    it = np.nditer(model.conv.W, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        orig = model.conv.W[idx]
        model.conv.W[idx] = orig + eps
        loss_plus = model.compute_loss(X, y)
        model.conv.W[idx] = orig - eps
        loss_minus = model.compute_loss(X, y)
        model.conv.W[idx] = orig
        approx_W[idx] = (loss_plus - loss_minus) / (2 * eps)
        it.iternext()

    # compare a few elements
    diff = np.linalg.norm(analytic_W - approx_W) / (np.linalg.norm(analytic_W) + np.linalg.norm(approx_W) + 1e-12)
    print(f"卷积层 dW 相对误差: {diff:.6e}")

    # check fc.W
    analytic_fc = model.fc.dW.copy()
    approx_fc = np.zeros_like(model.fc.W)
    print("开始数值梯度检查（全连接层）...")
    it = np.nditer(model.fc.W, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        orig = model.fc.W[idx]
        model.fc.W[idx] = orig + eps
        loss_plus = model.compute_loss(X, y)
        model.fc.W[idx] = orig - eps
        loss_minus = model.compute_loss(X, y)
        model.fc.W[idx] = orig
        approx_fc[idx] = (loss_plus - loss_minus) / (2 * eps)
        it.iternext()

    diff_fc = np.linalg.norm(analytic_fc - approx_fc) / (np.linalg.norm(analytic_fc) + np.linalg.norm(approx_fc) + 1e-12)
    print(f"全连接层 dW 相对误差: {diff_fc:.6e}")


# ----------------------------- 示例训练 -----------------------------

def demo_train_digits(epochs: int = 20, lr: float = 0.01, batch_size: int = 32):
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    digits = load_digits()
    X = digits.images  # shape (1797, 8, 8)
    y = digits.target

    # normalize [0,16] -> [0,1]
    X = X / 16.0
    N, H, W = X.shape
    X = X.reshape(N, 1, H, W).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = SimpleCNN(input_shape=(1, H, W), num_classes=10, lr=lr)

    # small training loop
    for epoch in range(1, epochs + 1):
        # shuffle
        perm = np.random.permutation(X_train.shape[0])
        X_train = X_train[perm]
        y_train = y_train[perm]

        losses = []
        accs = []
        for i in range(0, X_train.shape[0], batch_size):
            Xb = X_train[i:i + batch_size]
            yb = y_train[i:i + batch_size]
            cache = model.forward(Xb)
            loss = cross_entropy_loss(cache['probs'], yb)
            model.backward(cache, yb)
            pred = np.argmax(cache['probs'], axis=1)
            acc = np.mean(pred == yb)
            losses.append(loss)
            accs.append(acc)

        train_loss = np.mean(losses)
        train_acc = np.mean(accs)
        test_loss = model.compute_loss(X_test, y_test)
        test_pred = model.predict(X_test)
        test_acc = np.mean(test_pred == y_test)
        print(f"Epoch {epoch:2d}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")

        # do a quick gradient check on first epoch
        if epoch == 1:
            # compute analytic grads by running backward on a tiny batch
            Xb = X_train[:8]
            yb = y_train[:8]
            cache = model.forward(Xb)
            model.backward(cache, yb)
            numeric_gradient_check(model, Xb, yb)

    print("Training finished.")


if __name__ == '__main__':
    print('\nCNN 反向传播示例 (NumPy)')
    demo_train_digits(epochs=12, lr=0.05, batch_size=32)
