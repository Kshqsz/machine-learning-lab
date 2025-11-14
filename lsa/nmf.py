"""
非负矩阵分解 (Non-negative Matrix Factorization, NMF)
====================================================

NMF是一种矩阵分解技术，将非负矩阵V分解为两个非负矩阵W和H的乘积。
广泛应用于主题模型、图像处理、推荐系统等领域。

理论基础
--------

1. **基本问题**:
   给定非负矩阵 V ∈ R^(m×n)，找到非负矩阵 W ∈ R^(m×k) 和 H ∈ R^(k×n)，使得：
   $$V \approx W \times H$$
   
   其中：
   - V: 原始数据矩阵（如词-文档矩阵）
   - W: 基矩阵（如词-主题矩阵）
   - H: 系数矩阵（如主题-文档矩阵）
   - k: 潜在因子数（k << min(m, n)）

2. **目标函数**:
   
   **Frobenius范数（欧氏距离）**:
   $$\min_{W,H} ||V - WH||_F^2 = \min_{W,H} \sum_{ij} (V_{ij} - (WH)_{ij})^2$$
   约束条件: W ≥ 0, H ≥ 0
   
   **KL散度（Kullback-Leibler divergence）**:
   $$\min_{W,H} D_{KL}(V||WH) = \sum_{ij} \left(V_{ij}\log\frac{V_{ij}}{(WH)_{ij}} - V_{ij} + (WH)_{ij}\right)$$
   约束条件: W ≥ 0, H ≥ 0

3. **乘法更新规则**:
   
   **Frobenius范数的更新规则**（Lee & Seung, 2001）:
   $$H_{kn} \leftarrow H_{kn} \frac{(W^T V)_{kn}}{(W^T W H)_{kn}}$$
   $$W_{mk} \leftarrow W_{mk} \frac{(V H^T)_{mk}}{(W H H^T)_{mk}}$$
   
   **KL散度的更新规则**:
   $$H_{kn} \leftarrow H_{kn} \frac{\sum_m W_{mk} V_{mn} / (WH)_{mn}}{\sum_m W_{mk}}$$
   $$W_{mk} \leftarrow W_{mk} \frac{\sum_n H_{kn} V_{mn} / (WH)_{mn}}{\sum_n H_{kn}}$$

4. **收敛性质**:
   - 乘法更新规则保证目标函数单调递减
   - W和H始终保持非负
   - 通常收敛到局部最优解

5. **NMF vs LSA (SVD)**:
   
   | 特性 | NMF | LSA/SVD |
   |------|-----|---------|
   | 因子符号 | 非负 | 可正可负 |
   | 可解释性 | 强（部分表示） | 弱（整体模式） |
   | 稀疏性 | 通常稀疏 | 通常稠密 |
   | 唯一性 | 不唯一 | 唯一（到符号） |
   | 计算 | 迭代算法 | 直接分解 |

算法步骤
--------

1. 初始化W和H为非负随机值
2. 迭代更新：
   a. 更新H: H ← H ⊙ (W^T V) / (W^T W H + ε)
   b. 更新W: W ← W ⊙ (V H^T) / (W H H^T + ε)
   （⊙ 表示逐元素乘法，ε防止除零）
3. 重复步骤2直到收敛或达到最大迭代次数

应用场景
--------
- 文本挖掘：主题模型、文档聚类
- 图像处理：人脸识别、图像压缩
- 音频处理：源分离、特征提取
- 推荐系统：协同过滤
- 生物信息学：基因表达分析
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Literal
import warnings
import time

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class NMF:
    """
    非负矩阵分解 (Non-negative Matrix Factorization)
    
    参数
    ----
    n_components : int
        潜在因子的数量（k）
    init : str
        初始化方法：'random', 'nndsvd', 'nndsvda', 'nndsvdar'
    solver : str
        求解器：'mu' (乘法更新), 'cd' (坐标下降)
    beta_loss : float or str
        损失函数：'frobenius' (欧氏距离), 'kullback-leibler' (KL散度)
        或者beta值（beta=2对应frobenius，beta=1对应KL）
    max_iter : int
        最大迭代次数
    tol : float
        收敛容差
    random_state : int
        随机种子
    verbose : bool
        是否显示训练信息
    """
    
    def __init__(
        self,
        n_components: int = 10,
        init: Literal['random', 'nndsvd', 'nndsvda', 'nndsvdar'] = 'random',
        solver: Literal['mu', 'cd'] = 'mu',
        beta_loss: Literal['frobenius', 'kullback-leibler'] = 'frobenius',
        max_iter: int = 200,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        verbose: bool = False
    ):
        self.n_components = n_components
        self.init = init
        self.solver = solver
        self.beta_loss = beta_loss
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        
        # 训练后设置
        self.W_ = None  # 基矩阵
        self.H_ = None  # 系数矩阵
        self.reconstruction_err_ = None  # 重构误差
        self.n_iter_ = 0  # 实际迭代次数
        self.loss_history_ = []  # 损失历史
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _initialize_wh(self, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        初始化W和H矩阵
        
        参数
        ----
        V : np.ndarray
            输入矩阵 (m, n)
            
        返回
        ----
        W : np.ndarray
            基矩阵 (m, k)
        H : np.ndarray
            系数矩阵 (k, n)
        """
        m, n = V.shape
        k = self.n_components
        
        if self.init == 'random':
            # 随机初始化
            W = np.abs(np.random.randn(m, k))
            H = np.abs(np.random.randn(k, n))
            
        elif self.init == 'nndsvd':
            # NNDSVD初始化（Non-Negative Double SVD）
            W, H = self._nndsvd(V, k)
            
        elif self.init == 'nndsvda':
            # NNDSVD with zeros filled with average
            W, H = self._nndsvd(V, k)
            avg = V.mean()
            W[W == 0] = avg
            H[H == 0] = avg
            
        elif self.init == 'nndsvdar':
            # NNDSVD with zeros filled with small random values
            W, H = self._nndsvd(V, k)
            avg = V.mean()
            W[W == 0] = avg * np.abs(np.random.randn(np.sum(W == 0))) * 0.01
            H[H == 0] = avg * np.abs(np.random.randn(np.sum(H == 0))) * 0.01
            
        else:
            raise ValueError(f"Unknown init method: {self.init}")
        
        return W, H
    
    def _nndsvd(self, V: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        NNDSVD初始化方法
        基于SVD的非负初始化策略
        """
        m, n = V.shape
        
        # SVD分解
        U, S, Vt = np.linalg.svd(V, full_matrices=False)
        
        # 初始化W和H
        W = np.zeros((m, k))
        H = np.zeros((k, n))
        
        # 第一个成分直接从SVD获取
        W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
        H[0, :] = np.sqrt(S[0]) * np.abs(Vt[0, :])
        
        # 后续成分
        for j in range(1, k):
            u = U[:, j]
            v = Vt[j, :]
            
            # 分离正负部分
            u_pos = np.maximum(u, 0)
            u_neg = np.abs(np.minimum(u, 0))
            v_pos = np.maximum(v, 0)
            v_neg = np.abs(np.minimum(v, 0))
            
            # 计算norm
            u_pos_norm = np.linalg.norm(u_pos)
            u_neg_norm = np.linalg.norm(u_neg)
            v_pos_norm = np.linalg.norm(v_pos)
            v_neg_norm = np.linalg.norm(v_neg)
            
            # 选择更大的分量
            norm_pos = u_pos_norm * v_pos_norm
            norm_neg = u_neg_norm * v_neg_norm
            
            if norm_pos >= norm_neg:
                W[:, j] = np.sqrt(S[j] * norm_pos) / u_pos_norm * u_pos
                H[j, :] = np.sqrt(S[j] * norm_pos) / v_pos_norm * v_pos
            else:
                W[:, j] = np.sqrt(S[j] * norm_neg) / u_neg_norm * u_neg
                H[j, :] = np.sqrt(S[j] * norm_neg) / v_neg_norm * v_neg
        
        return W, H
    
    def _compute_loss(self, V: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
        """计算损失函数"""
        WH = W @ H
        
        if self.beta_loss == 'frobenius':
            # Frobenius范数（欧氏距离）
            loss = 0.5 * np.sum((V - WH) ** 2)
        elif self.beta_loss == 'kullback-leibler':
            # KL散度
            # D_KL(V||WH) = sum(V * log(V/WH) - V + WH)
            # 添加小常数避免log(0)
            epsilon = 1e-10
            loss = np.sum(V * np.log((V + epsilon) / (WH + epsilon)) - V + WH)
        else:
            raise ValueError(f"Unknown beta_loss: {self.beta_loss}")
        
        return loss
    
    def _update_mu_frobenius(
        self,
        V: np.ndarray,
        W: np.ndarray,
        H: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Frobenius范数的乘法更新规则
        """
        epsilon = 1e-10
        
        # 更新H: H ← H ⊙ (W^T V) / (W^T W H + ε)
        WtV = W.T @ V
        WtWH = W.T @ W @ H
        H = H * WtV / (WtWH + epsilon)
        
        # 更新W: W ← W ⊙ (V H^T) / (W H H^T + ε)
        VHt = V @ H.T
        WHHt = W @ H @ H.T
        W = W * VHt / (WHHt + epsilon)
        
        return W, H
    
    def _update_mu_kl(
        self,
        V: np.ndarray,
        W: np.ndarray,
        H: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        KL散度的乘法更新规则
        """
        epsilon = 1e-10
        
        # 更新H
        WH = W @ H + epsilon
        numerator_H = W.T @ (V / WH)
        denominator_H = W.sum(axis=0, keepdims=True).T
        H = H * numerator_H / (denominator_H + epsilon)
        
        # 更新W
        WH = W @ H + epsilon
        numerator_W = (V / WH) @ H.T
        denominator_W = H.sum(axis=1, keepdims=True).T
        W = W * numerator_W / (denominator_W + epsilon)
        
        return W, H
    
    def fit(self, V: np.ndarray) -> 'NMF':
        """
        在数据矩阵V上训练NMF模型
        
        参数
        ----
        V : np.ndarray
            输入非负矩阵 (m, n)
            
        返回
        ----
        self : NMF
        """
        # 检查输入
        if np.any(V < 0):
            raise ValueError("输入矩阵包含负值，NMF要求非负矩阵")
        
        m, n = V.shape
        
        # 初始化W和H
        W, H = self._initialize_wh(V)
        
        # 记录初始损失
        self.loss_history_ = []
        initial_loss = self._compute_loss(V, W, H)
        self.loss_history_.append(initial_loss)
        
        if self.verbose:
            print(f"NMF训练开始:")
            print(f"  矩阵形状: V={V.shape}, W={W.shape}, H={H.shape}")
            print(f"  损失函数: {self.beta_loss}")
            print(f"  初始损失: {initial_loss:.6f}")
        
        # 迭代更新
        start_time = time.time()
        
        for iter_num in range(self.max_iter):
            # 选择更新规则
            if self.beta_loss == 'frobenius':
                W, H = self._update_mu_frobenius(V, W, H)
            elif self.beta_loss == 'kullback-leibler':
                W, H = self._update_mu_kl(V, W, H)
            
            # 计算损失
            loss = self._compute_loss(V, W, H)
            self.loss_history_.append(loss)
            
            # 检查收敛
            if iter_num > 0:
                loss_change = abs(self.loss_history_[-2] - loss)
                if loss_change < self.tol:
                    if self.verbose:
                        print(f"  迭代 {iter_num + 1}: 损失={loss:.6f}, 变化={loss_change:.6e} < tol={self.tol}")
                        print(f"  提前收敛!")
                    break
            
            # 显示进度
            if self.verbose and (iter_num + 1) % 20 == 0:
                elapsed = time.time() - start_time
                print(f"  迭代 {iter_num + 1}/{self.max_iter}: 损失={loss:.6f}, 用时={elapsed:.2f}s")
        
        self.n_iter_ = iter_num + 1
        self.W_ = W
        self.H_ = H
        self.reconstruction_err_ = loss
        
        if self.verbose:
            total_time = time.time() - start_time
            print(f"\nNMF训练完成:")
            print(f"  迭代次数: {self.n_iter_}")
            print(f"  最终损失: {self.reconstruction_err_:.6f}")
            print(f"  总用时: {total_time:.2f}s")
        
        return self
    
    def transform(self, V: np.ndarray) -> np.ndarray:
        """
        将新数据投影到H空间（固定W，求解H）
        
        参数
        ----
        V : np.ndarray
            输入矩阵 (m, n_new)
            
        返回
        ----
        H : np.ndarray
            系数矩阵 (k, n_new)
        """
        if self.W_ is None:
            raise ValueError("模型未训练，请先调用fit()")
        
        m, n_new = V.shape
        k = self.n_components
        
        # 初始化H
        H = np.abs(np.random.randn(k, n_new))
        
        # 固定W，更新H
        epsilon = 1e-10
        for _ in range(100):  # 固定迭代次数
            if self.beta_loss == 'frobenius':
                WtV = self.W_.T @ V
                WtWH = self.W_.T @ self.W_ @ H
                H = H * WtV / (WtWH + epsilon)
            elif self.beta_loss == 'kullback-leibler':
                WH = self.W_ @ H + epsilon
                numerator = self.W_.T @ (V / WH)
                denominator = self.W_.sum(axis=0, keepdims=True).T
                H = H * numerator / (denominator + epsilon)
        
        return H
    
    def fit_transform(self, V: np.ndarray) -> np.ndarray:
        """
        训练模型并返回H矩阵
        
        参数
        ----
        V : np.ndarray
            输入矩阵 (m, n)
            
        返回
        ----
        H : np.ndarray
            系数矩阵 (k, n)
        """
        self.fit(V)
        return self.H_
    
    def inverse_transform(self, H: np.ndarray) -> np.ndarray:
        """
        从H重构V: V ≈ W @ H
        
        参数
        ----
        H : np.ndarray
            系数矩阵 (k, n)
            
        返回
        ----
        V_reconstructed : np.ndarray
            重构的矩阵 (m, n)
        """
        if self.W_ is None:
            raise ValueError("模型未训练")
        
        return self.W_ @ H
    
    def get_components(self) -> np.ndarray:
        """获取基矩阵W（主题/组件）"""
        if self.W_ is None:
            raise ValueError("模型未训练")
        return self.W_
    
    def get_feature_names(self, feature_idx: int, top_n: int = 10) -> np.ndarray:
        """
        获取某个组件的top特征
        
        参数
        ----
        feature_idx : int
            组件索引
        top_n : int
            返回前n个特征
            
        返回
        ----
        top_features : np.ndarray
            特征索引数组
        """
        if self.W_ is None:
            raise ValueError("模型未训练")
        
        component = self.W_[:, feature_idx]
        top_indices = np.argsort(component)[::-1][:top_n]
        
        return top_indices


# ============================================================================
# 示例应用
# ============================================================================

def demo_1_basic_nmf():
    """
    示例1: 基本的NMF分解
    """
    print("=" * 70)
    print("示例1: 基本的NMF分解")
    print("=" * 70)
    
    # 创建一个简单的非负矩阵
    np.random.seed(42)
    m, n, k = 10, 8, 3
    
    # 生成真实的W和H
    W_true = np.abs(np.random.randn(m, k))
    H_true = np.abs(np.random.randn(k, n))
    V = W_true @ H_true
    
    print(f"\n生成的矩阵: V = W_true @ H_true")
    print(f"  V.shape = {V.shape}")
    print(f"  W_true.shape = {W_true.shape}")
    print(f"  H_true.shape = {H_true.shape}")
    print(f"\nV的前5行前5列:")
    print(V[:5, :5])
    
    # 使用NMF分解
    nmf = NMF(n_components=k, init='random', max_iter=200, verbose=True, random_state=42)
    H = nmf.fit_transform(V)
    W = nmf.W_
    
    # 重构误差
    V_reconstructed = W @ H
    reconstruction_error = np.linalg.norm(V - V_reconstructed, 'fro')
    relative_error = reconstruction_error / np.linalg.norm(V, 'fro')
    
    print(f"\n重构结果:")
    print(f"  重构误差（Frobenius范数）: {reconstruction_error:.6f}")
    print(f"  相对误差: {relative_error:.6%}")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 原始矩阵
    im1 = axes[0, 0].imshow(V, cmap='YlOrRd', aspect='auto')
    axes[0, 0].set_title('原始矩阵 V')
    axes[0, 0].set_xlabel('列')
    axes[0, 0].set_ylabel('行')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 重构矩阵
    im2 = axes[0, 1].imshow(V_reconstructed, cmap='YlOrRd', aspect='auto')
    axes[0, 1].set_title(f'重构矩阵 W @ H\n(误差: {relative_error:.2%})')
    axes[0, 1].set_xlabel('列')
    axes[0, 1].set_ylabel('行')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # W矩阵
    im3 = axes[1, 0].imshow(W, cmap='Blues', aspect='auto')
    axes[1, 0].set_title(f'基矩阵 W ({W.shape[0]}×{W.shape[1]})')
    axes[1, 0].set_xlabel('组件')
    axes[1, 0].set_ylabel('特征')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # H矩阵
    im4 = axes[1, 1].imshow(H, cmap='Greens', aspect='auto')
    axes[1, 1].set_title(f'系数矩阵 H ({H.shape[0]}×{H.shape[1]})')
    axes[1, 1].set_xlabel('样本')
    axes[1, 1].set_ylabel('组件')
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('lsa/nmf_basic_decomposition.png', dpi=150, bbox_inches='tight')
    print("\n图片已保存: lsa/nmf_basic_decomposition.png")
    plt.close()


def demo_2_convergence():
    """
    示例2: 收敛性分析
    """
    print("\n" + "=" * 70)
    print("示例2: NMF收敛性分析")
    print("=" * 70)
    
    # 生成测试数据
    np.random.seed(42)
    m, n, k = 50, 40, 5
    W_true = np.abs(np.random.randn(m, k))
    H_true = np.abs(np.random.randn(k, n))
    V = W_true @ H_true
    
    print(f"\n测试矩阵: V.shape = {V.shape}")
    
    # 测试不同的初始化方法
    init_methods = ['random', 'nndsvd', 'nndsvda']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for init_method in init_methods:
        nmf = NMF(n_components=k, init=init_method, max_iter=200, 
                 verbose=False, random_state=42)
        nmf.fit(V)
        
        # 绘制损失曲线
        axes[0].plot(nmf.loss_history_, label=f'{init_method}', linewidth=2)
        
        print(f"\n{init_method}初始化:")
        print(f"  迭代次数: {nmf.n_iter_}")
        print(f"  最终损失: {nmf.reconstruction_err_:.6f}")
    
    axes[0].set_xlabel('迭代次数')
    axes[0].set_ylabel('损失（Frobenius范数）')
    axes[0].set_title('不同初始化方法的收敛曲线')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # 测试不同的损失函数
    beta_losses = ['frobenius', 'kullback-leibler']
    
    for beta_loss in beta_losses:
        nmf = NMF(n_components=k, init='nndsvd', beta_loss=beta_loss,
                 max_iter=200, verbose=False, random_state=42)
        nmf.fit(V)
        
        axes[1].plot(nmf.loss_history_, label=beta_loss, linewidth=2)
        
        print(f"\n{beta_loss}损失:")
        print(f"  迭代次数: {nmf.n_iter_}")
        print(f"  最终损失: {nmf.reconstruction_err_:.6f}")
    
    axes[1].set_xlabel('迭代次数')
    axes[1].set_ylabel('损失值')
    axes[1].set_title('不同损失函数的收敛曲线')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('lsa/nmf_convergence.png', dpi=150, bbox_inches='tight')
    print("\n图片已保存: lsa/nmf_convergence.png")
    plt.close()


def demo_3_image_compression():
    """
    示例3: 图像压缩应用
    """
    print("\n" + "=" * 70)
    print("示例3: NMF用于图像压缩")
    print("=" * 70)
    
    # 创建一个简单的"图像"（棋盘图案 + 噪声）
    np.random.seed(42)
    size = 64
    image = np.zeros((size, size))
    
    # 创建棋盘图案
    square_size = 8
    for i in range(0, size, square_size * 2):
        for j in range(0, size, square_size * 2):
            image[i:i+square_size, j:j+square_size] = 1.0
            image[i+square_size:i+2*square_size, j+square_size:j+2*square_size] = 1.0
    
    # 添加一些噪声
    image = image + 0.1 * np.abs(np.random.randn(size, size))
    image = np.clip(image, 0, 1)
    
    print(f"\n原始图像大小: {image.shape}")
    print(f"原始数据量: {image.size} 个值")
    
    # 测试不同的组件数
    n_components_list = [2, 5, 10, 20]
    
    fig, axes = plt.subplots(2, len(n_components_list) + 1, figsize=(16, 7))
    
    # 显示原始图像
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('原始图像')
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    
    for idx, n_comp in enumerate(n_components_list, 1):
        # NMF分解
        nmf = NMF(n_components=n_comp, init='nndsvd', max_iter=200, 
                 verbose=False, random_state=42)
        H = nmf.fit_transform(image)
        W = nmf.W_
        
        # 重构图像
        image_reconstructed = W @ H
        
        # 计算压缩率和误差
        compressed_size = W.size + H.size
        compression_ratio = image.size / compressed_size
        mse = np.mean((image - image_reconstructed) ** 2)
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
        
        print(f"\nn_components = {n_comp}:")
        print(f"  压缩后数据量: {compressed_size}")
        print(f"  压缩率: {compression_ratio:.2f}x")
        print(f"  MSE: {mse:.6f}")
        print(f"  PSNR: {psnr:.2f} dB")
        
        # 显示重构图像
        axes[0, idx].imshow(image_reconstructed, cmap='gray')
        axes[0, idx].set_title(f'k={n_comp}\n压缩率: {compression_ratio:.1f}x')
        axes[0, idx].axis('off')
        
        # 显示误差
        error = np.abs(image - image_reconstructed)
        axes[1, idx].imshow(error, cmap='hot')
        axes[1, idx].set_title(f'误差\nPSNR: {psnr:.1f}dB')
        axes[1, idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('lsa/nmf_image_compression.png', dpi=150, bbox_inches='tight')
    print("\n图片已保存: lsa/nmf_image_compression.png")
    plt.close()


def demo_4_topic_modeling():
    """
    示例4: 主题模型（文本分析）
    """
    print("\n" + "=" * 70)
    print("示例4: NMF用于主题模型")
    print("=" * 70)
    
    # 简化的词-文档矩阵（行=词，列=文档）
    # 模拟3个主题：体育、科技、艺术
    np.random.seed(42)
    
    # 词汇表
    vocabulary = [
        # 体育相关 (0-9)
        'game', 'team', 'player', 'win', 'score', 'match', 'sport', 'coach', 'championship', 'competition',
        # 科技相关 (10-19)
        'computer', 'software', 'algorithm', 'data', 'program', 'code', 'technology', 'system', 'network', 'digital',
        # 艺术相关 (20-29)
        'art', 'music', 'paint', 'artist', 'creative', 'design', 'gallery', 'exhibition', 'performance', 'culture'
    ]
    
    n_words = len(vocabulary)
    n_docs = 20
    n_topics = 3
    
    # 创建词-文档矩阵
    # 每个文档主要属于一个主题
    V = np.zeros((n_words, n_docs))
    
    for doc_idx in range(n_docs):
        # 决定主题
        if doc_idx < 7:  # 体育文档
            topic_words = list(range(0, 10))
            V[topic_words, doc_idx] = np.abs(np.random.randn(10)) + 2
        elif doc_idx < 14:  # 科技文档
            topic_words = list(range(10, 20))
            V[topic_words, doc_idx] = np.abs(np.random.randn(10)) + 2
        else:  # 艺术文档
            topic_words = list(range(20, 30))
            V[topic_words, doc_idx] = np.abs(np.random.randn(10)) + 2
        
        # 添加一些其他主题的词（混合）
        other_words = np.random.choice(n_words, size=3, replace=False)
        V[other_words, doc_idx] += np.abs(np.random.randn(3)) * 0.5
    
    print(f"\n词-文档矩阵: {V.shape}")
    print(f"  词汇量: {n_words}")
    print(f"  文档数: {n_docs}")
    
    # NMF主题模型
    nmf = NMF(n_components=n_topics, init='nndsvd', max_iter=200, 
             verbose=True, random_state=42)
    H = nmf.fit_transform(V)
    W = nmf.W_
    
    # 显示每个主题的关键词
    print("\n" + "=" * 70)
    print("主题的关键词:")
    print("=" * 70)
    
    n_top_words = 10
    for topic_idx in range(n_topics):
        top_indices = nmf.get_feature_names(topic_idx, top_n=n_top_words)
        top_words = [vocabulary[i] for i in top_indices]
        top_weights = W[top_indices, topic_idx]
        
        print(f"\n主题 {topic_idx}:")
        for word, weight in zip(top_words, top_weights):
            print(f"  {word:15s} {weight:.4f}")
    
    # 可视化主题-词分布
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for topic_idx in range(n_topics):
        # 获取该主题的词权重
        weights = W[:, topic_idx]
        top_indices = np.argsort(weights)[::-1][:10]
        top_words = [vocabulary[i] for i in top_indices]
        top_weights = weights[top_indices]
        
        axes[topic_idx].barh(range(len(top_words)), top_weights)
        axes[topic_idx].set_yticks(range(len(top_words)))
        axes[topic_idx].set_yticklabels(top_words)
        axes[topic_idx].set_xlabel('权重')
        axes[topic_idx].set_title(f'主题 {topic_idx} 的关键词')
        axes[topic_idx].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('lsa/nmf_topic_modeling.png', dpi=150, bbox_inches='tight')
    print("\n图片已保存: lsa/nmf_topic_modeling.png")
    plt.close()


def demo_5_comparison_with_svd():
    """
    示例5: NMF与SVD的对比
    """
    print("\n" + "=" * 70)
    print("示例5: NMF与SVD(LSA)的对比")
    print("=" * 70)
    
    # 生成数据
    np.random.seed(42)
    m, n, k = 20, 15, 3
    W_true = np.abs(np.random.randn(m, k))
    H_true = np.abs(np.random.randn(k, n))
    V = W_true @ H_true
    
    print(f"\n数据矩阵: V.shape = {V.shape}")
    
    # NMF分解
    nmf = NMF(n_components=k, init='nndsvd', max_iter=200, verbose=False, random_state=42)
    H_nmf = nmf.fit_transform(V)
    W_nmf = nmf.W_
    V_nmf = W_nmf @ H_nmf
    
    # SVD分解（LSA）
    U, S, Vt = np.linalg.svd(V, full_matrices=False)
    W_svd = U[:, :k] @ np.diag(S[:k])
    H_svd = Vt[:k, :]
    V_svd = W_svd @ H_svd
    
    # 计算重构误差
    error_nmf = np.linalg.norm(V - V_nmf, 'fro')
    error_svd = np.linalg.norm(V - V_svd, 'fro')
    
    print(f"\n重构误差:")
    print(f"  NMF: {error_nmf:.6f}")
    print(f"  SVD: {error_svd:.6f}")
    
    # 检查非负性
    print(f"\n非负性:")
    print(f"  NMF W 最小值: {W_nmf.min():.6f}")
    print(f"  NMF H 最小值: {H_nmf.min():.6f}")
    print(f"  SVD W 最小值: {W_svd.min():.6f}")
    print(f"  SVD H 最小值: {H_svd.min():.6f}")
    
    # 稀疏性（零元素比例）
    threshold = 0.01
    sparsity_nmf_w = np.sum(np.abs(W_nmf) < threshold) / W_nmf.size
    sparsity_nmf_h = np.sum(np.abs(H_nmf) < threshold) / H_nmf.size
    sparsity_svd_w = np.sum(np.abs(W_svd) < threshold) / W_svd.size
    sparsity_svd_h = np.sum(np.abs(H_svd) < threshold) / H_svd.size
    
    print(f"\n稀疏性（值<{threshold}的比例）:")
    print(f"  NMF W: {sparsity_nmf_w:.2%}")
    print(f"  NMF H: {sparsity_nmf_h:.2%}")
    print(f"  SVD W: {sparsity_svd_w:.2%}")
    print(f"  SVD H: {sparsity_svd_h:.2%}")
    
    # 可视化对比
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # NMF
    im1 = axes[0, 0].imshow(W_nmf, cmap='Blues', aspect='auto')
    axes[0, 0].set_title('NMF: W矩阵')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(H_nmf, cmap='Greens', aspect='auto')
    axes[0, 1].set_title('NMF: H矩阵')
    plt.colorbar(im2, ax=axes[0, 1])
    
    im3 = axes[0, 2].imshow(V_nmf, cmap='YlOrRd', aspect='auto')
    axes[0, 2].set_title('NMF: 重构 W@H')
    plt.colorbar(im3, ax=axes[0, 2])
    
    error_map_nmf = np.abs(V - V_nmf)
    im4 = axes[0, 3].imshow(error_map_nmf, cmap='hot', aspect='auto')
    axes[0, 3].set_title(f'NMF: 误差\n(Frob={error_nmf:.3f})')
    plt.colorbar(im4, ax=axes[0, 3])
    
    # SVD
    im5 = axes[1, 0].imshow(W_svd, cmap='Blues', aspect='auto')
    axes[1, 0].set_title('SVD: U@S矩阵')
    plt.colorbar(im5, ax=axes[1, 0])
    
    im6 = axes[1, 1].imshow(H_svd, cmap='Greens', aspect='auto')
    axes[1, 1].set_title('SVD: Vt矩阵')
    plt.colorbar(im6, ax=axes[1, 1])
    
    im7 = axes[1, 2].imshow(V_svd, cmap='YlOrRd', aspect='auto')
    axes[1, 2].set_title('SVD: 重构 U@S@Vt')
    plt.colorbar(im7, ax=axes[1, 2])
    
    error_map_svd = np.abs(V - V_svd)
    im8 = axes[1, 3].imshow(error_map_svd, cmap='hot', aspect='auto')
    axes[1, 3].set_title(f'SVD: 误差\n(Frob={error_svd:.3f})')
    plt.colorbar(im8, ax=axes[1, 3])
    
    plt.tight_layout()
    plt.savefig('lsa/nmf_vs_svd.png', dpi=150, bbox_inches='tight')
    print("\n图片已保存: lsa/nmf_vs_svd.png")
    plt.close()
    
    print("\n" + "=" * 70)
    print("对比总结:")
    print("=" * 70)
    print("SVD:")
    print("  + 最优重构（最小Frobenius范数）")
    print("  + 唯一解（到符号和旋转）")
    print("  + 快速计算（直接分解）")
    print("  - 可能产生负值")
    print("  - 解释性较差")
    print("\nNMF:")
    print("  + 保证非负性（部分表示）")
    print("  + 通常更稀疏")
    print("  + 可解释性强（主题模型、组件分析）")
    print("  - 局部最优（多个解）")
    print("  - 迭代算法（可能较慢）")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("非负矩阵分解 (Non-negative Matrix Factorization, NMF)")
    print("=" * 70)
    
    # 运行所有示例
    demo_1_basic_nmf()
    demo_2_convergence()
    demo_3_image_compression()
    demo_4_topic_modeling()
    demo_5_comparison_with_svd()
    
    print("\n" + "=" * 70)
    print("所有示例运行完成!")
    print("=" * 70)
    print("\n总结:")
    print("1. 基本分解: 展示了NMF的基本原理和矩阵分解过程")
    print("2. 收敛性分析: 比较了不同初始化方法和损失函数")
    print("3. 图像压缩: 演示了NMF在图像压缩中的应用")
    print("4. 主题模型: 使用NMF进行文本主题提取")
    print("5. 与SVD对比: 分析了NMF和SVD的优缺点")
    print("\n核心特点:")
    print("- 乘法更新规则保证非负性和单调收敛")
    print("- 适合需要部分表示和可解释性的场景")
    print("- 在主题模型、推荐系统、图像处理等领域广泛应用")
