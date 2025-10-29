"""
非线性支持向量机 - 核函数方法
Nonlinear Support Vector Machine - Kernel Method

算法原理:
1. 通过核函数将输入空间映射到高维特征空间
2. 在高维空间中寻找线性分离超平面
3. 核技巧: K(x_i, x_j) = φ(x_i)·φ(x_j)，无需显式计算映射

核函数类型:
1. 线性核: K(x, z) = x·z
2. 多项式核: K(x, z) = (γ·x·z + r)^d
3. 高斯RBF核: K(x, z) = exp(-γ||x-z||²)
4. Sigmoid核: K(x, z) = tanh(γ·x·z + r)

优势:
- 可处理非线性分类问题
- 核技巧避免显式高维映射
- 理论基础完善
- 泛化能力强

适用场景:
- 数据非线性可分
- XOR问题、圆形分布等
- 图像识别、文本分类
- 生物信息学
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS
plt.rcParams['axes.unicode_minus'] = False


class KernelSVM:
    """核函数支持向量机"""
    
    def __init__(self, C=1.0, kernel='rbf', gamma='auto', degree=3, coef0=0.0):
        """
        参数:
            C: 正则化参数（软间隔惩罚系数）
            kernel: 核函数类型 ('linear', 'poly', 'rbf', 'sigmoid')
            gamma: RBF、多项式、sigmoid核的系数，'auto'时为1/n_features
            degree: 多项式核的次数
            coef0: 多项式核和sigmoid核的独立项
        """
        self.C = C
        self.kernel_type = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        
        # 训练后的参数
        self.alpha = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.support_vector_alpha = None
        self.b = None
        
    def _compute_kernel_matrix(self, X1, X2=None):
        """
        计算核矩阵（Gram矩阵）
        
        参数:
            X1: 第一组样本 (n1, d)
            X2: 第二组样本 (n2, d)，如果为None则X2=X1
        
        返回:
            K: 核矩阵 (n1, n2)
        """
        if X2 is None:
            X2 = X1
        
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        K = np.zeros((n1, n2))
        
        # 自动设置gamma
        if self.gamma == 'auto':
            gamma_value = 1.0 / X1.shape[1]
        else:
            gamma_value = self.gamma
        
        if self.kernel_type == 'linear':
            # 线性核: K(x, z) = x·z
            K = np.dot(X1, X2.T)
            
        elif self.kernel_type == 'poly':
            # 多项式核: K(x, z) = (γ·x·z + r)^d
            K = (gamma_value * np.dot(X1, X2.T) + self.coef0) ** self.degree
            
        elif self.kernel_type == 'rbf':
            # 高斯RBF核: K(x, z) = exp(-γ||x-z||²)
            for i in range(n1):
                for j in range(n2):
                    diff = X1[i] - X2[j]
                    K[i, j] = np.exp(-gamma_value * np.dot(diff, diff))
                    
        elif self.kernel_type == 'sigmoid':
            # Sigmoid核: K(x, z) = tanh(γ·x·z + r)
            K = np.tanh(gamma_value * np.dot(X1, X2.T) + self.coef0)
            
        else:
            raise ValueError(f"未知的核函数类型: {self.kernel_type}")
        
        return K
    
    def _objective(self, alpha, K, y):
        """
        对偶问题的目标函数（最小化形式）
        
        min: (1/2)α^T Q α - 1^T α
        其中 Q_ij = y_i y_j K(x_i, x_j)
        """
        n = len(alpha)
        Q = np.outer(y, y) * K
        return 0.5 * np.dot(alpha, np.dot(Q, alpha)) - np.sum(alpha)
    
    def _objective_gradient(self, alpha, K, y):
        """目标函数的梯度"""
        Q = np.outer(y, y) * K
        return np.dot(Q, alpha) - np.ones(len(alpha))
    
    def fit(self, X, y, verbose=True):
        """
        训练核SVM
        
        参数:
            X: 训练样本 (n_samples, n_features)
            y: 标签 (n_samples,) 取值为+1或-1
        """
        n_samples, n_features = X.shape
        
        if verbose:
            print("="*70)
            print("非线性支持向量机 - 核函数方法")
            print("="*70)
            print(f"训练样本数: {n_samples}")
            print(f"特征维度: {n_features}")
            print(f"核函数类型: {self.kernel_type}")
            print(f"惩罚参数 C: {self.C}")
            
            if self.kernel_type == 'rbf' or self.kernel_type == 'poly' or self.kernel_type == 'sigmoid':
                gamma_value = 1.0 / n_features if self.gamma == 'auto' else self.gamma
                print(f"Gamma: {gamma_value:.4f}")
            
            if self.kernel_type == 'poly':
                print(f"多项式次数: {self.degree}")
                print(f"独立项: {self.coef0}")
            
            print()
        
        # 计算核矩阵（Gram矩阵）
        if verbose:
            print("计算核矩阵...")
        K = self._compute_kernel_matrix(X)
        
        # 对偶问题的约束条件
        # 约束: Σ α_i y_i = 0
        constraints = {'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y)}
        
        # 边界约束: 0 ≤ α_i ≤ C
        bounds = [(0, self.C) for _ in range(n_samples)]
        
        # 初始值
        alpha0 = np.zeros(n_samples)
        
        # 求解对偶问题
        if verbose:
            print("求解二次规划问题...")
        
        result = minimize(
            fun=self._objective,
            x0=alpha0,
            args=(K, y),
            method='SLSQP',
            jac=self._objective_gradient,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'disp': False}
        )
        
        if not result.success:
            print(f"⚠️  优化警告: {result.message}")
        
        self.alpha = result.x
        
        # 提取支持向量（α > 阈值）
        sv_threshold = 1e-5
        sv_indices = self.alpha > sv_threshold
        
        self.support_vectors = X[sv_indices]
        self.support_vector_labels = y[sv_indices]
        self.support_vector_alpha = self.alpha[sv_indices]
        
        n_support_vectors = len(self.support_vector_alpha)
        
        # 计算偏置 b
        # 使用边界支持向量 (0 < α < C)
        margin_threshold = 1e-4
        margin_sv_mask = (self.support_vector_alpha > margin_threshold) & \
                        (self.support_vector_alpha < self.C - margin_threshold)
        
        if np.sum(margin_sv_mask) > 0:
            # 使用边界支持向量计算b
            margin_sv = self.support_vectors[margin_sv_mask]
            margin_sv_labels = self.support_vector_labels[margin_sv_mask]
            
            # 计算核矩阵
            K_margin = self._compute_kernel_matrix(margin_sv, self.support_vectors)
            
            # b = y_s - Σ α_i y_i K(x_i, x_s)
            b_values = []
            for i in range(len(margin_sv)):
                b_val = margin_sv_labels[i] - np.sum(
                    self.support_vector_alpha * self.support_vector_labels * K_margin[i]
                )
                b_values.append(b_val)
            
            self.b = np.mean(b_values)
        else:
            # 如果没有边界支持向量，使用所有支持向量
            K_all = self._compute_kernel_matrix(self.support_vectors, self.support_vectors)
            b_values = []
            for i in range(n_support_vectors):
                b_val = self.support_vector_labels[i] - np.sum(
                    self.support_vector_alpha * self.support_vector_labels * K_all[i]
                )
                b_values.append(b_val)
            self.b = np.mean(b_values)
        
        if verbose:
            print("-"*70)
            print("训练完成！")
            print()
            
            print("支持向量信息:")
            print("-"*70)
            print(f"支持向量数量: {n_support_vectors}")
            print(f"支持向量比例: {n_support_vectors/n_samples*100:.2f}%")
            
            # 区分边界支持向量和内部支持向量
            n_margin_sv = np.sum(margin_sv_mask)
            n_inner_sv = n_support_vectors - n_margin_sv
            print(f"  - 边界支持向量 (0 < α < C): {n_margin_sv}")
            print(f"  - 内部支持向量 (α = C): {n_inner_sv}")
            
            print()
            print(f"偏置 b: {self.b:.6f}")
            print("="*70)
    
    def decision_function(self, X):
        """
        计算决策函数值
        
        f(x) = Σ α_i y_i K(x_i, x) + b
        """
        K = self._compute_kernel_matrix(X, self.support_vectors)
        return np.dot(K, self.support_vector_alpha * self.support_vector_labels) + self.b
    
    def predict(self, X):
        """预测样本类别"""
        return np.sign(self.decision_function(X))
    
    def plot_decision_boundary(self, X, y, title="", resolution=200):
        """可视化决策边界（仅支持2D数据）"""
        if X.shape[1] != 2:
            print("只支持2维特征的可视化")
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 创建网格
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution)
        )
        
        # 计算决策函数值
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self.decision_function(grid_points)
        Z = Z.reshape(xx.shape)
        
        # 绘制决策边界和间隔边界
        ax.contour(xx, yy, Z, colors='black', levels=[0], linewidths=2.5, linestyles='-')
        ax.contour(xx, yy, Z, colors='gray', levels=[-1, 1], linewidths=2, linestyles='--')
        
        # 绘制决策区域
        ax.contourf(xx, yy, Z, levels=[-np.inf, 0, np.inf], 
                   colors=['lightblue', 'lightcoral'], alpha=0.3)
        
        # 绘制训练样本
        for label, marker, color in [(1, 'o', 'red'), (-1, 's', 'blue')]:
            mask = y == label
            ax.scatter(X[mask, 0], X[mask, 1], c=color, marker=marker, s=120,
                      edgecolors='black', linewidths=1.5, label=f'类别 {label:+d}',
                      zorder=3)
        
        # 标记支持向量
        ax.scatter(self.support_vectors[:, 0], self.support_vectors[:, 1],
                  s=300, facecolors='none', edgecolors='green', linewidths=3,
                  label='支持向量', zorder=4)
        
        ax.set_xlabel('x₁', fontsize=13)
        ax.set_ylabel('x₂', fontsize=13)
        ax.set_title(f'非线性SVM - {self.kernel_type.upper()}核{title}', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 添加文本说明
        gamma_value = 1.0 / X.shape[1] if self.gamma == 'auto' else self.gamma
        text = f'核函数: {self.kernel_type}\n'
        text += f'C = {self.C}\n'
        
        if self.kernel_type == 'rbf':
            text += f'γ = {gamma_value:.4f}\n'
        elif self.kernel_type == 'poly':
            text += f'次数 d = {self.degree}\n'
            text += f'γ = {gamma_value:.4f}\n'
        
        text += f'支持向量: {len(self.support_vectors)}\n'
        text += f'b = {self.b:.4f}'
        
        ax.text(0.02, 0.98, text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # 保存图片
        kernel_name = self.kernel_type
        if self.kernel_type == 'poly':
            kernel_name = f'poly_d{self.degree}'
        elif self.kernel_type == 'rbf':
            kernel_name = f'rbf_g{gamma_value:.3f}'
        
        filename = f'svm_kernel_{kernel_name}_C{self.C}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n图像已保存至: {filename}")
        plt.show()


def demo_xor_problem():
    """演示1: XOR问题（经典非线性问题）"""
    print("\n" + "="*70)
    print("示例 1: XOR问题")
    print("="*70)
    print("XOR问题是经典的非线性分类问题，线性SVM无法解决")
    print()
    
    # XOR数据
    X_xor = np.array([
        [1, 1],
        [1, -1],
        [-1, 1],
        [-1, -1],
        [1.2, 1.2],
        [1.1, -1.1],
        [-1.1, 1.1],
        [-1.2, -1.2]
    ])
    
    y_xor = np.array([1, -1, -1, 1, 1, -1, -1, 1])
    
    print("训练数据 (XOR):")
    print("-"*70)
    for i, (x, label) in enumerate(zip(X_xor, y_xor)):
        category = "正例" if label == 1 else "负例"
        print(f"  样本 {i+1}: x = ({x[0]:6.1f}, {x[1]:6.1f})  →  y = {label:+d}  ({category})")
    print()
    
    # 使用RBF核
    svm = KernelSVM(C=1.0, kernel='rbf', gamma=1.0)
    svm.fit(X_xor, y_xor)
    
    # 评估
    y_pred = svm.predict(X_xor)
    accuracy = np.mean(y_pred == y_xor) * 100
    
    print("\n训练集预测结果:")
    print("-"*70)
    for i, (x, y_true, y_p) in enumerate(zip(X_xor, y_xor, y_pred)):
        score = svm.decision_function(x.reshape(1, -1))[0]
        result = "✓" if y_p == y_true else "✗"
        category = "正例" if y_true == 1 else "负例"
        print(f"  样本 {i+1} ({category}): f(x) = {score:+8.4f}  →  预测: {int(y_p):+d}  真实: {y_true:+d}  {result}")
    
    print(f"\n训练集准确率: {accuracy:.2f}%")
    print()
    
    # 可视化
    svm.plot_decision_boundary(X_xor, y_xor)


def demo_circles_problem():
    """演示2: 同心圆问题"""
    print("\n" + "="*70)
    print("示例 2: 同心圆分类问题")
    print("="*70)
    print("内圆为一类，外环为另一类，需要非线性决策边界")
    print()
    
    # 生成同心圆数据
    np.random.seed(42)
    n_samples_per_class = 20
    
    # 内圆（正类）
    r_inner = np.random.uniform(0, 1.5, n_samples_per_class)
    theta_inner = np.random.uniform(0, 2*np.pi, n_samples_per_class)
    X_inner = np.column_stack([
        r_inner * np.cos(theta_inner),
        r_inner * np.sin(theta_inner)
    ])
    y_inner = np.ones(n_samples_per_class)
    
    # 外环（负类）
    r_outer = np.random.uniform(3, 4.5, n_samples_per_class)
    theta_outer = np.random.uniform(0, 2*np.pi, n_samples_per_class)
    X_outer = np.column_stack([
        r_outer * np.cos(theta_outer),
        r_outer * np.sin(theta_outer)
    ])
    y_outer = -np.ones(n_samples_per_class)
    
    # 合并数据
    X_circles = np.vstack([X_inner, X_outer])
    y_circles = np.concatenate([y_inner, y_outer])
    
    print(f"训练数据: {len(X_circles)}个样本")
    print(f"  内圆（正类）: {n_samples_per_class}个")
    print(f"  外环（负类）: {n_samples_per_class}个")
    print()
    
    # 使用RBF核
    svm = KernelSVM(C=10.0, kernel='rbf', gamma=0.5)
    svm.fit(X_circles, y_circles)
    
    # 评估
    y_pred = svm.predict(X_circles)
    accuracy = np.mean(y_pred == y_circles) * 100
    print(f"\n训练集准确率: {accuracy:.2f}%")
    
    # 可视化
    svm.plot_decision_boundary(X_circles, y_circles)


def demo_polynomial_boundary():
    """演示3: 多项式决策边界"""
    print("\n" + "="*70)
    print("示例 3: 多项式核 - 抛物线分类问题")
    print("="*70)
    print("数据分布呈抛物线形状，适合使用多项式核")
    print()
    
    # 生成抛物线数据
    np.random.seed(42)
    n_samples = 30
    
    # 正类：抛物线上方
    X_pos = np.random.uniform(-3, 3, (n_samples, 2))
    X_pos[:, 1] = X_pos[:, 1] + 2  # 向上移动
    X_pos = X_pos[X_pos[:, 1] > X_pos[:, 0]**2 - 2][:n_samples//2]
    y_pos = np.ones(len(X_pos))
    
    # 负类：抛物线下方
    X_neg = np.random.uniform(-3, 3, (n_samples, 2))
    X_neg[:, 1] = X_neg[:, 1] - 2  # 向下移动
    X_neg = X_neg[X_neg[:, 1] < X_neg[:, 0]**2 - 4][:n_samples//2]
    y_neg = -np.ones(len(X_neg))
    
    # 合并数据
    X_poly = np.vstack([X_pos, X_neg])
    y_poly = np.concatenate([y_pos, y_neg])
    
    print(f"训练数据: {len(X_poly)}个样本")
    print(f"  正类: {len(X_pos)}个")
    print(f"  负类: {len(X_neg)}个")
    print()
    
    # 使用多项式核
    svm = KernelSVM(C=1.0, kernel='poly', degree=2, gamma=1.0, coef0=1.0)
    svm.fit(X_poly, y_poly)
    
    # 评估
    y_pred = svm.predict(X_poly)
    accuracy = np.mean(y_pred == y_poly) * 100
    print(f"\n训练集准确率: {accuracy:.2f}%")
    
    # 可视化
    svm.plot_decision_boundary(X_poly, y_poly)


def demo_kernel_comparison():
    """演示4: 不同核函数的对比"""
    print("\n" + "="*70)
    print("示例 4: 核函数对比")
    print("="*70)
    print("在同一数据集上比较不同核函数的效果")
    print()
    
    # 生成混合数据
    np.random.seed(42)
    
    # 类1: 两个簇
    X1_cluster1 = np.random.randn(15, 2) * 0.5 + np.array([2, 2])
    X1_cluster2 = np.random.randn(15, 2) * 0.5 + np.array([-2, -2])
    X_class1 = np.vstack([X1_cluster1, X1_cluster2])
    y_class1 = np.ones(30)
    
    # 类2: 中间区域
    X_class2 = np.random.randn(30, 2) * 0.8
    y_class2 = -np.ones(30)
    
    X_mix = np.vstack([X_class1, X_class2])
    y_mix = np.concatenate([y_class1, y_class2])
    
    print(f"训练数据: {len(X_mix)}个样本（正类30，负类30）")
    print()
    
    kernels = [
        ('linear', {}, '线性核'),
        ('poly', {'degree': 2}, '多项式核(d=2)'),
        ('rbf', {'gamma': 0.5}, 'RBF核(γ=0.5)'),
        ('rbf', {'gamma': 2.0}, 'RBF核(γ=2.0)')
    ]
    
    for kernel_type, params, name in kernels:
        print(f"\n{'='*70}")
        print(f"测试 {name}")
        print('='*70)
        
        svm = KernelSVM(C=1.0, kernel=kernel_type, **params)
        svm.fit(X_mix, y_mix, verbose=False)
        
        y_pred = svm.predict(X_mix)
        accuracy = np.mean(y_pred == y_mix) * 100
        
        print(f"准确率: {accuracy:.2f}%")
        print(f"支持向量数: {len(svm.support_vectors)}")
        
        svm.plot_decision_boundary(X_mix, y_mix, title=f" - {name}")


def main():
    """主函数"""
    print("\n" + "🎯 "*20)
    print("非线性支持向量机 - 核函数方法演示")
    print("🎯 "*20 + "\n")
    
    # 演示1: XOR问题
    demo_xor_problem()
    
    # 演示2: 同心圆问题
    demo_circles_problem()
    
    # 演示3: 多项式核
    demo_polynomial_boundary()
    
    # 演示4: 核函数对比
    demo_kernel_comparison()
    
    print("\n" + "="*70)
    print("所有演示完成！")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
