"""
奇异值分解 (Singular Value Decomposition, SVD)
==============================================

奇异值分解（SVD）是线性代数中一种重要的矩阵分解方法，它将任意矩阵分解为三个矩阵的乘积。
SVD在数据分析、降维、推荐系统、图像压缩等领域有广泛应用。

数学定义：
对于任意 m×n 矩阵 A，存在分解：
    A = U Σ V^T

其中：
- U: m×m 正交矩阵，列向量是 AA^T 的特征向量（左奇异向量）
- Σ: m×n 对角矩阵，对角元素是 A 的奇异值（按降序排列）
- V: n×n 正交矩阵，列向量是 A^T A 的特征向量（右奇异向量）

性质：
1. U^T U = I（单位矩阵）
2. V^T V = I（单位矩阵）
3. σ₁ ≥ σ₂ ≥ ... ≥ σᵣ > 0（奇异值降序排列）
4. r = rank(A)（矩阵的秩等于非零奇异值的个数）

截断SVD（低秩近似）：
    A ≈ A_k = U_k Σ_k V_k^T
保留前k个最大的奇异值，可以得到A的最佳k-秩近似（Frobenius范数意义下）

应用场景：
- 主成分分析（PCA）
- 图像压缩和去噪
- 推荐系统（协同过滤）
- 潜在语义分析（LSA）
- 数据降维和可视化

作者：Kshqsz
日期：2025-11-12
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import warnings

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class SVD:
    """
    奇异值分解实现类
    
    参数:
    -----------
    n_components : int, optional
        保留的奇异值数量（截断SVD），None表示保留全部
    """
    
    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components
        
        # 存储分解结果
        self.U_ = None  # 左奇异向量矩阵
        self.sigma_ = None  # 奇异值（一维数组）
        self.Vt_ = None  # 右奇异向量矩阵的转置
        self.original_shape_ = None
        self.explained_variance_ratio_ = None  # 每个成分的方差解释比例
    
    def fit(self, X: np.ndarray) -> 'SVD':
        """
        对矩阵进行奇异值分解
        
        参数:
        -----------
        X : array-like, shape (m, n)
            输入矩阵
            
        返回:
        -----------
        self : object
            返回自身实例
        """
        X = np.array(X, dtype=float)
        self.original_shape_ = X.shape
        m, n = X.shape
        
        print(f"\n{'='*60}")
        print(f"奇异值分解 (SVD)")
        print(f"{'='*60}")
        print(f"矩阵形状: {m} × {n}")
        print(f"矩阵秩: {np.linalg.matrix_rank(X)}")
        
        # 使用NumPy的SVD函数
        # full_matrices=False: 返回经济型SVD（更高效）
        U, sigma, Vt = np.linalg.svd(X, full_matrices=False)
        
        # 如果指定了n_components，进行截断
        if self.n_components is not None:
            k = min(self.n_components, len(sigma))
            print(f"截断SVD: 保留前 {k} 个成分")
            U = U[:, :k]
            sigma = sigma[:k]
            Vt = Vt[:k, :]
        else:
            k = len(sigma)
            print(f"完整SVD: {k} 个奇异值")
        
        self.U_ = U
        self.sigma_ = sigma
        self.Vt_ = Vt
        
        # 计算方差解释比例
        total_variance = np.sum(sigma ** 2)
        self.explained_variance_ratio_ = (sigma ** 2) / total_variance
        
        print(f"\n奇异值统计:")
        print(f"最大奇异值: {sigma[0]:.4f}")
        print(f"最小奇异值: {sigma[-1]:.4f}")
        print(f"条件数: {sigma[0]/sigma[-1]:.4f}")
        print(f"\n前5个奇异值:")
        for i, s in enumerate(sigma[:5]):
            print(f"  σ_{i+1} = {s:.4f} (解释方差: {self.explained_variance_ratio_[i]*100:.2f}%)")
        
        cumsum = np.cumsum(self.explained_variance_ratio_)
        print(f"\n累积方差解释比例:")
        for i in [0, min(4, k-1), min(9, k-1), min(19, k-1), k-1]:
            if i < k:
                print(f"  前 {i+1} 个成分: {cumsum[i]*100:.2f}%")
        
        print(f"{'='*60}\n")
        
        return self
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        分解并投影到低维空间
        
        参数:
        -----------
        X : array-like, shape (m, n)
            输入矩阵
            
        返回:
        -----------
        X_transformed : array, shape (m, k)
            降维后的数据（k = n_components）
        """
        self.fit(X)
        return self.transform(X)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        将数据投影到SVD空间
        
        参数:
        -----------
        X : array-like, shape (m, n)
            输入矩阵
            
        返回:
        -----------
        X_transformed : array, shape (m, k)
            投影后的数据
        """
        if self.U_ is None:
            raise ValueError("SVD not fitted. Call fit() first.")
        
        X = np.array(X, dtype=float)
        # X_transformed = X @ V @ Σ^(-1) = U @ Σ
        return self.U_ @ np.diag(self.sigma_)
    
    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        从降维空间重建原始数据
        
        参数:
        -----------
        X_transformed : array-like, shape (m, k)
            降维后的数据
            
        返回:
        -----------
        X_reconstructed : array, shape (m, n)
            重建的数据
        """
        if self.Vt_ is None:
            raise ValueError("SVD not fitted. Call fit() first.")
        
        # X ≈ U @ Σ @ V^T
        return X_transformed @ self.Vt_
    
    def reconstruct(self) -> np.ndarray:
        """
        重建原始矩阵（使用保留的奇异值）
        
        返回:
        -----------
        X_reconstructed : array
            重建的矩阵
        """
        if self.U_ is None:
            raise ValueError("SVD not fitted. Call fit() first.")
        
        # A_k = U_k @ Σ_k @ V_k^T
        return self.U_ @ np.diag(self.sigma_) @ self.Vt_
    
    def get_reconstruction_error(self, X: np.ndarray) -> float:
        """
        计算重建误差（Frobenius范数）
        
        参数:
        -----------
        X : array-like
            原始矩阵
            
        返回:
        -----------
        error : float
            重建误差
        """
        X = np.array(X, dtype=float)
        X_reconstructed = self.reconstruct()
        return np.linalg.norm(X - X_reconstructed, 'fro')
    
    def get_compression_ratio(self) -> float:
        """
        计算压缩比
        
        返回:
        -----------
        ratio : float
            压缩比（原始大小/压缩后大小）
        """
        if self.U_ is None:
            raise ValueError("SVD not fitted. Call fit() first.")
        
        m, n = self.original_shape_
        k = len(self.sigma_)
        
        original_size = m * n
        compressed_size = m * k + k + k * n  # U + sigma + Vt
        
        return original_size / compressed_size


def demo_basic_svd():
    """演示基础SVD分解"""
    print("\n" + "="*60)
    print("示例1: 基础奇异值分解")
    print("="*60)
    
    # 创建一个简单的矩阵
    np.random.seed(42)
    A = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12]
    ], dtype=float)
    
    print("\n原始矩阵 A (4×3):")
    print(A)
    print(f"\n矩阵秩: {np.linalg.matrix_rank(A)}")
    
    # 完整SVD
    svd = SVD()
    svd.fit(A)
    
    # 重建矩阵
    A_reconstructed = svd.reconstruct()
    
    print("\n重建矩阵 A' = U @ Σ @ V^T:")
    print(A_reconstructed)
    
    # 计算重建误差
    error = np.linalg.norm(A - A_reconstructed, 'fro')
    print(f"\n重建误差（Frobenius范数）: {error:.10f}")
    
    # 可视化
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    
    # 原始矩阵
    im1 = axes[0].imshow(A, cmap='viridis', aspect='auto')
    axes[0].set_title('原始矩阵 A', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('列')
    axes[0].set_ylabel('行')
    plt.colorbar(im1, ax=axes[0])
    
    # U矩阵
    im2 = axes[1].imshow(svd.U_, cmap='RdBu', aspect='auto')
    axes[1].set_title(f'左奇异向量 U\n({svd.U_.shape[0]}×{svd.U_.shape[1]})', 
                     fontsize=12, fontweight='bold')
    axes[1].set_xlabel('列')
    axes[1].set_ylabel('行')
    plt.colorbar(im2, ax=axes[1])
    
    # 奇异值
    axes[2].bar(range(1, len(svd.sigma_)+1), svd.sigma_, color='steelblue', alpha=0.7)
    axes[2].set_title('奇异值 Σ', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('索引')
    axes[2].set_ylabel('奇异值')
    axes[2].grid(True, alpha=0.3)
    
    # V^T矩阵
    im4 = axes[3].imshow(svd.Vt_, cmap='RdBu', aspect='auto')
    axes[3].set_title(f'右奇异向量转置 V^T\n({svd.Vt_.shape[0]}×{svd.Vt_.shape[1]})', 
                     fontsize=12, fontweight='bold')
    axes[3].set_xlabel('列')
    axes[3].set_ylabel('行')
    plt.colorbar(im4, ax=axes[3])
    
    plt.tight_layout()
    plt.savefig('svd/svd_basic_decomposition.png', dpi=150, bbox_inches='tight')
    print(f"\n图形已保存: svd/svd_basic_decomposition.png")
    plt.close()


def demo_truncated_svd():
    """演示截断SVD（低秩近似）"""
    print("\n" + "="*60)
    print("示例2: 截断SVD与低秩近似")
    print("="*60)
    
    # 创建一个更大的随机矩阵
    np.random.seed(42)
    m, n = 50, 30
    rank = 10
    
    # 构造一个秩为10的矩阵
    U_true = np.random.randn(m, rank)
    V_true = np.random.randn(rank, n)
    A = U_true @ V_true
    
    # 添加一些噪声
    A = A + np.random.randn(m, n) * 0.5
    
    print(f"\n原始矩阵: {m}×{n}")
    print(f"矩阵秩: {np.linalg.matrix_rank(A)}")
    
    # 测试不同的截断等级
    k_values = [1, 3, 5, 10, 15, 20, 30]
    errors = []
    compression_ratios = []
    
    print("\n不同截断等级的效果:")
    for k in k_values:
        svd = SVD(n_components=k)
        svd.fit(A)
        error = svd.get_reconstruction_error(A)
        compression_ratio = svd.get_compression_ratio()
        errors.append(error)
        compression_ratios.append(compression_ratio)
        
        cumsum_var = np.sum(svd.explained_variance_ratio_)
        print(f"k={k:2d}: 误差={error:8.4f}, 压缩比={compression_ratio:.2f}, "
              f"累积方差={cumsum_var*100:.2f}%")
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 原始矩阵
    im1 = axes[0, 0].imshow(A, cmap='viridis', aspect='auto')
    axes[0, 0].set_title(f'原始矩阵 ({m}×{n})', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 不同k值的重建结果
    k_show = [1, 5, 10, 20]
    for idx, k in enumerate(k_show[:3]):
        svd = SVD(n_components=k)
        svd.fit(A)
        A_k = svd.reconstruct()
        error = svd.get_reconstruction_error(A)
        
        row = (idx + 1) // 3
        col = (idx + 1) % 3
        im = axes[row, col].imshow(A_k, cmap='viridis', aspect='auto')
        axes[row, col].set_title(f'k={k} 重建\n误差={error:.2f}', 
                                fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=axes[row, col])
    
    # 重建误差曲线
    axes[1, 0].plot(k_values, errors, 'bo-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('保留的奇异值数量 k', fontsize=11)
    axes[1, 0].set_ylabel('重建误差（Frobenius范数）', fontsize=11)
    axes[1, 0].set_title('重建误差 vs k', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # 压缩比曲线
    axes[1, 1].plot(k_values, compression_ratios, 'go-', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('保留的奇异值数量 k', fontsize=11)
    axes[1, 1].set_ylabel('压缩比', fontsize=11)
    axes[1, 1].set_title('压缩比 vs k', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=1, color='r', linestyle='--', alpha=0.5, label='无压缩')
    axes[1, 1].legend()
    
    # 奇异值分布
    svd_full = SVD()
    svd_full.fit(A)
    axes[1, 2].semilogy(range(1, len(svd_full.sigma_)+1), svd_full.sigma_, 
                       'ro-', linewidth=2, markersize=6)
    axes[1, 2].set_xlabel('奇异值索引', fontsize=11)
    axes[1, 2].set_ylabel('奇异值（对数尺度）', fontsize=11)
    axes[1, 2].set_title('奇异值谱', fontsize=12, fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('svd/svd_truncated_approximation.png', dpi=150, bbox_inches='tight')
    print(f"\n图形已保存: svd/svd_truncated_approximation.png")
    plt.close()


def demo_image_compression():
    """演示图像压缩应用"""
    print("\n" + "="*60)
    print("示例3: SVD图像压缩")
    print("="*60)
    
    # 创建一个合成图像（或读取真实图像）
    np.random.seed(42)
    
    # 方法1: 创建一个简单的合成图像
    size = 100
    x = np.linspace(-5, 5, size)
    y = np.linspace(-5, 5, size)
    X, Y = np.meshgrid(x, y)
    
    # 创建一些有趣的图案
    image = np.sin(np.sqrt(X**2 + Y**2)) * np.exp(-0.1*(X**2 + Y**2))
    image = (image - image.min()) / (image.max() - image.min())  # 归一化到[0,1]
    
    print(f"\n图像大小: {image.shape[0]}×{image.shape[1]}")
    print(f"图像秩: {np.linalg.matrix_rank(image)}")
    
    # 测试不同的压缩等级
    k_values = [1, 5, 10, 20, 50]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    # 显示原始图像
    im0 = axes[0].imshow(image, cmap='gray')
    axes[0].set_title('原始图像', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0])
    
    print("\n不同压缩等级的效果:")
    for idx, k in enumerate(k_values):
        svd = SVD(n_components=k)
        svd.fit(image)
        image_compressed = svd.reconstruct()
        
        # 计算压缩质量指标
        error = svd.get_reconstruction_error(image)
        compression_ratio = svd.get_compression_ratio()
        
        # 计算PSNR（峰值信噪比）
        mse = np.mean((image - image_compressed)**2)
        if mse > 0:
            psnr = 10 * np.log10(1.0 / mse)
        else:
            psnr = float('inf')
        
        print(f"k={k:3d}: 误差={error:.4f}, 压缩比={compression_ratio:.2f}x, "
              f"PSNR={psnr:.2f}dB")
        
        # 显示压缩后的图像
        im = axes[idx+1].imshow(image_compressed, cmap='gray')
        axes[idx+1].set_title(f'k={k} (压缩比={compression_ratio:.1f}x)\n'
                             f'PSNR={psnr:.1f}dB', 
                             fontsize=11, fontweight='bold')
        axes[idx+1].axis('off')
        plt.colorbar(im, ax=axes[idx+1])
    
    plt.tight_layout()
    plt.savefig('svd/svd_image_compression.png', dpi=150, bbox_inches='tight')
    print(f"\n图形已保存: svd/svd_image_compression.png")
    plt.close()
    
    # 额外绘制：压缩质量分析
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    k_range = range(1, min(image.shape[0], image.shape[1])+1, 5)
    errors = []
    psnrs = []
    compression_ratios = []
    
    for k in k_range:
        svd = SVD(n_components=k)
        svd.fit(image)
        image_compressed = svd.reconstruct()
        
        error = svd.get_reconstruction_error(image)
        errors.append(error)
        
        mse = np.mean((image - image_compressed)**2)
        if mse > 0:
            psnr = 10 * np.log10(1.0 / mse)
        else:
            psnr = 100
        psnrs.append(psnr)
        
        compression_ratios.append(svd.get_compression_ratio())
    
    # 重建误差
    axes[0].plot(k_range, errors, 'b-', linewidth=2)
    axes[0].set_xlabel('保留的奇异值数量 k', fontsize=11)
    axes[0].set_ylabel('重建误差', fontsize=11)
    axes[0].set_title('重建误差 vs k', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # PSNR
    axes[1].plot(k_range, psnrs, 'g-', linewidth=2)
    axes[1].set_xlabel('保留的奇异值数量 k', fontsize=11)
    axes[1].set_ylabel('PSNR (dB)', fontsize=11)
    axes[1].set_title('峰值信噪比 vs k', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=30, color='r', linestyle='--', alpha=0.5, label='可接受质量')
    axes[1].legend()
    
    # PSNR vs 压缩比
    axes[2].plot(compression_ratios, psnrs, 'r-', linewidth=2)
    axes[2].set_xlabel('压缩比', fontsize=11)
    axes[2].set_ylabel('PSNR (dB)', fontsize=11)
    axes[2].set_title('质量-压缩率权衡', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('svd/svd_compression_analysis.png', dpi=150, bbox_inches='tight')
    print(f"图形已保存: svd/svd_compression_analysis.png")
    plt.close()


def demo_data_visualization():
    """演示SVD在数据降维和可视化中的应用"""
    print("\n" + "="*60)
    print("示例4: SVD降维与可视化")
    print("="*60)
    
    # 生成高维数据（模拟真实数据集）
    np.random.seed(42)
    n_samples = 200
    n_features = 50
    
    # 创建具有低维结构的高维数据
    # 数据实际上存在于3个主成分中
    latent_dim = 3
    Z = np.random.randn(n_samples, latent_dim)
    W = np.random.randn(latent_dim, n_features)
    X = Z @ W + np.random.randn(n_samples, n_features) * 0.5
    
    # 创建标签（3个类别）
    labels = np.zeros(n_samples)
    labels[n_samples//3:2*n_samples//3] = 1
    labels[2*n_samples//3:] = 2
    
    print(f"\n数据维度: {n_samples} 样本 × {n_features} 特征")
    print(f"真实类别数: 3")
    
    # 进行SVD降维
    svd = SVD(n_components=10)
    X_reduced = svd.fit_transform(X)
    
    print(f"\n降维后: {X_reduced.shape[0]} 样本 × {X_reduced.shape[1]} 特征")
    
    # 可视化
    fig = plt.figure(figsize=(18, 12))
    
    # 1. 奇异值谱
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(range(1, 11), svd.sigma_, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('成分索引', fontsize=11)
    ax1.set_ylabel('奇异值', fontsize=11)
    ax1.set_title('奇异值谱（前10个）', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. 方差解释比例
    ax2 = plt.subplot(2, 3, 2)
    ax2.bar(range(1, 11), svd.explained_variance_ratio_ * 100, 
           color='steelblue', alpha=0.7)
    ax2.set_xlabel('成分索引', fontsize=11)
    ax2.set_ylabel('方差解释比例 (%)', fontsize=11)
    ax2.set_title('各成分方差贡献', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. 累积方差解释比例
    ax3 = plt.subplot(2, 3, 3)
    cumsum = np.cumsum(svd.explained_variance_ratio_) * 100
    ax3.plot(range(1, 11), cumsum, 'go-', linewidth=2, markersize=8)
    ax3.axhline(y=90, color='r', linestyle='--', alpha=0.5, label='90%阈值')
    ax3.set_xlabel('成分数量', fontsize=11)
    ax3.set_ylabel('累积方差解释比例 (%)', fontsize=11)
    ax3.set_title('累积方差解释', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. 2D可视化（前两个主成分）
    ax4 = plt.subplot(2, 3, 4)
    for i, label in enumerate(['类别0', '类别1', '类别2']):
        mask = labels == i
        ax4.scatter(X_reduced[mask, 0], X_reduced[mask, 1], 
                   label=label, alpha=0.6, s=50)
    ax4.set_xlabel('第1主成分', fontsize=11)
    ax4.set_ylabel('第2主成分', fontsize=11)
    ax4.set_title('2D投影（PC1 vs PC2）', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 2D可视化（第1和第3主成分）
    ax5 = plt.subplot(2, 3, 5)
    for i, label in enumerate(['类别0', '类别1', '类别2']):
        mask = labels == i
        ax5.scatter(X_reduced[mask, 0], X_reduced[mask, 2], 
                   label=label, alpha=0.6, s=50)
    ax5.set_xlabel('第1主成分', fontsize=11)
    ax5.set_ylabel('第3主成分', fontsize=11)
    ax5.set_title('2D投影（PC1 vs PC3）', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 3D可视化
    ax6 = plt.subplot(2, 3, 6, projection='3d')
    for i, label in enumerate(['类别0', '类别1', '类别2']):
        mask = labels == i
        ax6.scatter(X_reduced[mask, 0], X_reduced[mask, 1], X_reduced[mask, 2],
                   label=label, alpha=0.6, s=30)
    ax6.set_xlabel('PC1', fontsize=10)
    ax6.set_ylabel('PC2', fontsize=10)
    ax6.set_zlabel('PC3', fontsize=10)
    ax6.set_title('3D投影', fontsize=12, fontweight='bold')
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig('svd/svd_dimensionality_reduction.png', dpi=150, bbox_inches='tight')
    print(f"\n图形已保存: svd/svd_dimensionality_reduction.png")
    plt.close()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("奇异值分解 (Singular Value Decomposition, SVD)")
    print("="*60)
    
    # 运行所有演示
    demo_basic_svd()
    demo_truncated_svd()
    demo_image_compression()
    demo_data_visualization()
    
    print("\n" + "="*60)
    print("所有演示完成！")
    print("="*60)
