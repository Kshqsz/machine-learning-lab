"""
线性可分支持向量机 - 对偶形式
Linear Support Vector Machine - Dual Form

算法原理:
1. 对偶问题: max Σα_i - (1/2)ΣΣ α_i α_j y_i y_j (x_i·x_j)
   约束条件: Σ α_i y_i = 0, α_i ≥ 0

2. 求解得到最优α*后，计算参数:
   w* = Σ α_i* y_i x_i
   b* = y_j - Σ α_i* y_i (x_i·x_j)  (任选一个α_j* > 0的样本)

3. 分离超平面: w*·x + b* = 0
   决策函数: f(x) = sign(w*·x + b*)

数据集:
- 正例: x1=(3,3), x2=(4,3)  y=+1
- 负例: x3=(1,1)            y=-1
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS
plt.rcParams['axes.unicode_minus'] = False


class LinearSVMDual:
    """线性可分支持向量机 - 对偶形式"""
    
    def __init__(self):
        self.w = None          # 权重向量
        self.b = None          # 偏置
        self.alpha = None      # 拉格朗日乘子
        self.support_vectors = None      # 支持向量
        self.support_labels = None       # 支持向量的标签
        self.support_alpha = None        # 支持向量的α值
        
    def fit(self, X, y):
        """
        训练SVM模型
        
        参数:
            X: 训练样本特征 (n_samples, n_features)
            y: 训练样本标签 (n_samples,) 取值为+1或-1
        """
        n_samples, n_features = X.shape
        
        print("="*70)
        print("线性可分支持向量机 - 对偶形式")
        print("="*70)
        print(f"训练样本数: {n_samples}")
        print(f"特征维度: {n_features}")
        print()
        
        # 计算Gram矩阵 K[i,j] = x_i · x_j
        K = np.zeros((n_samples, n_samples))
        print("Gram矩阵 (内积矩阵):")
        print("-"*70)
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = np.dot(X[i], X[j])
        
        for i in range(n_samples):
            print(f"  K[{i}] = {K[i]}")
        print()
        
        # 定义对偶问题的目标函数（要最小化负的对偶目标）
        # 原始: max Σα_i - (1/2)ΣΣ α_i α_j y_i y_j K[i,j]
        # 转为最小化: min -Σα_i + (1/2)ΣΣ α_i α_j y_i y_j K[i,j]
        def objective(alpha):
            """对偶目标函数（最小化）"""
            return -np.sum(alpha) + 0.5 * np.sum(
                alpha[:, None] * alpha[None, :] * 
                y[:, None] * y[None, :] * K
            )
        
        def objective_grad(alpha):
            """目标函数的梯度"""
            return -np.ones(n_samples) + np.sum(
                alpha[None, :] * y[:, None] * y[None, :] * K,
                axis=1
            )
        
        # 约束条件: Σ α_i y_i = 0
        constraints = {
            'type': 'eq',
            'fun': lambda alpha: np.dot(alpha, y),
            'jac': lambda alpha: y
        }
        
        # 边界条件: α_i ≥ 0
        bounds = [(0, None) for _ in range(n_samples)]
        
        # 初始值
        alpha0 = np.zeros(n_samples)
        
        print("求解对偶问题...")
        print("-"*70)
        print("目标函数: min -Σα_i + (1/2)ΣΣ α_i α_j y_i y_j (x_i·x_j)")
        print("约束条件: Σ α_i y_i = 0, α_i ≥ 0")
        print()
        
        # 使用SLSQP求解二次规划问题
        result = minimize(
            objective,
            alpha0,
            method='SLSQP',
            jac=objective_grad,
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'disp': False}
        )
        
        if not result.success:
            print("警告: 优化未完全收敛")
        
        self.alpha = result.x
        
        print("优化结果:")
        print("-"*70)
        for i in range(n_samples):
            print(f"  α_{i+1} = {self.alpha[i]:.6f}")
        print()
        
        # 找出支持向量 (α > 0 的样本)
        sv_indices = self.alpha > 1e-5
        self.support_vectors = X[sv_indices]
        self.support_labels = y[sv_indices]
        self.support_alpha = self.alpha[sv_indices]
        
        print(f"支持向量个数: {len(self.support_vectors)}")
        print("-"*70)
        for i, (sv, label, alpha) in enumerate(zip(
            self.support_vectors, self.support_labels, self.support_alpha
        )):
            print(f"  支持向量 {i+1}: x={sv}, y={label:+d}, α={alpha:.6f}")
        print()
        
        # 计算权重向量 w = Σ α_i y_i x_i
        self.w = np.sum(
            self.alpha[:, None] * y[:, None] * X,
            axis=0
        )
        
        print("权重向量:")
        print("-"*70)
        print(f"  w = {self.w}")
        print(f"  ||w|| = {np.linalg.norm(self.w):.6f}")
        print()
        
        # 计算偏置 b，使用任意一个支持向量
        # b = y_j - w·x_j = y_j - Σ α_i y_i (x_i·x_j)
        sv_idx = np.where(sv_indices)[0][0]  # 选择第一个支持向量
        self.b = y[sv_idx] - np.sum(
            self.alpha * y * K[:, sv_idx]
        )
        
        print("偏置:")
        print("-"*70)
        print(f"  b = {self.b:.6f}")
        print()
        
        # 计算间隔
        margin = 2.0 / np.linalg.norm(self.w)
        print("分类间隔:")
        print("-"*70)
        print(f"  间隔 = 2/||w|| = {margin:.6f}")
        print()
        
        print("分离超平面:")
        print("-"*70)
        if n_features == 2:
            print(f"  {self.w[0]:.6f}x₁ + {self.w[1]:.6f}x₂ + {self.b:.6f} = 0")
        else:
            print(f"  w·x + b = 0")
        print()
        
        print("决策函数:")
        print("-"*70)
        print(f"  f(x) = sign(w·x + b)")
        print("="*70)
        
    def predict(self, X):
        """
        预测样本类别
        
        参数:
            X: 测试样本 (n_samples, n_features)
        
        返回:
            y_pred: 预测标签 (n_samples,)
        """
        return np.sign(np.dot(X, self.w) + self.b)
    
    def decision_function(self, X):
        """
        计算决策函数值
        
        参数:
            X: 测试样本 (n_samples, n_features)
        
        返回:
            scores: 决策函数值 (n_samples,)
        """
        return np.dot(X, self.w) + self.b
    
    def plot_decision_boundary(self, X, y):
        """绘制决策边界和支持向量"""
        if X.shape[1] != 2:
            print("只支持2维特征的可视化")
            return
        
        plt.figure(figsize=(10, 8))
        
        # 绘制数据点
        for label, marker, color in [(1, 'o', 'red'), (-1, 's', 'blue')]:
            mask = y == label
            plt.scatter(
                X[mask, 0], X[mask, 1],
                c=color, marker=marker, s=100,
                edgecolors='black', linewidths=1.5,
                label=f'类别 {label:+d}',
                zorder=3
            )
        
        # 绘制支持向量（用圆圈标记）
        plt.scatter(
            self.support_vectors[:, 0],
            self.support_vectors[:, 1],
            s=200, linewidths=2,
            facecolors='none', edgecolors='green',
            label='支持向量',
            zorder=4
        )
        
        # 绘制决策边界和间隔边界
        xlim = plt.xlim()
        ylim = plt.ylim()
        
        # 创建网格
        xx = np.linspace(xlim[0] - 1, xlim[1] + 1, 200)
        yy = np.linspace(ylim[0] - 1, ylim[1] + 1, 200)
        XX, YY = np.meshgrid(xx, yy)
        xy = np.c_[XX.ravel(), YY.ravel()]
        
        # 计算决策函数值
        Z = self.decision_function(xy).reshape(XX.shape)
        
        # 绘制决策边界 (f(x) = 0)
        plt.contour(
            XX, YY, Z,
            colors='black', levels=[0],
            linestyles='-', linewidths=2,
            label='分离超平面 (f(x)=0)'
        )
        
        # 绘制间隔边界 (f(x) = ±1)
        plt.contour(
            XX, YY, Z,
            colors='gray', levels=[-1, 1],
            linestyles='--', linewidths=1.5,
            label='间隔边界 (f(x)=±1)'
        )
        
        # 绘制决策区域
        plt.contourf(
            XX, YY, Z,
            levels=[-np.inf, 0, np.inf],
            colors=['lightblue', 'lightcoral'],
            alpha=0.3
        )
        
        plt.xlabel('x₁', fontsize=12)
        plt.ylabel('x₂', fontsize=12)
        plt.title('线性可分支持向量机 - 对偶形式', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim(xlim)
        plt.ylim(ylim)
        
        # 添加文本说明
        text = f'w = ({self.w[0]:.3f}, {self.w[1]:.3f})\n'
        text += f'b = {self.b:.3f}\n'
        text += f'间隔 = {2.0/np.linalg.norm(self.w):.3f}'
        plt.text(
            0.02, 0.98, text,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
        
        plt.tight_layout()
        plt.savefig('svm/svm_dual_linear.png', dpi=300, bbox_inches='tight')
        print("\n图像已保存至: svm/svm_dual_linear.png")
        plt.show()


def main():
    """主函数"""
    # 训练数据
    # 正例: x1=(3,3), x2=(4,3)  标签 y=+1
    # 负例: x3=(1,1)            标签 y=-1
    X_train = np.array([
        [3, 3],  # x1
        [4, 3],  # x2
        [1, 1]   # x3
    ])
    
    y_train = np.array([1, 1, -1])  # 对应的标签
    
    print("\n训练数据:")
    print("-"*70)
    for i, (x, label) in enumerate(zip(X_train, y_train)):
        category = "正例" if label == 1 else "负例"
        print(f"  x{i+1} = {x}  →  y = {label:+d}  ({category})")
    print()
    
    # 创建并训练SVM模型
    svm = LinearSVMDual()
    svm.fit(X_train, y_train)
    
    # 在训练集上测试
    print("\n训练集预测:")
    print("-"*70)
    for i, (x, y_true) in enumerate(zip(X_train, y_train)):
        y_pred = int(svm.predict(x.reshape(1, -1))[0])
        score = svm.decision_function(x.reshape(1, -1))[0]
        result = "✓" if y_pred == y_true else "✗"
        print(f"  x{i+1} = {x}  →  f(x) = {score:+.6f}  →  预测: {y_pred:+d}  真实: {y_true:+d}  {result}")
    
    # 计算准确率
    y_pred_train = svm.predict(X_train)
    accuracy = np.mean(y_pred_train == y_train) * 100
    print(f"\n训练集准确率: {accuracy:.2f}%")
    print()
    
    # 测试新样本
    print("新样本预测:")
    print("-"*70)
    X_test = np.array([
        [2, 2],
        [3, 2],
        [5, 4]
    ])
    
    for x in X_test:
        y_pred = int(svm.predict(x.reshape(1, -1))[0])
        score = svm.decision_function(x.reshape(1, -1))[0]
        category = "正例" if y_pred == 1 else "负例"
        print(f"  x = {x}  →  f(x) = {score:+.6f}  →  预测: {y_pred:+d}  ({category})")
    print()
    
    # 可视化
    svm.plot_decision_boundary(X_train, y_train)


if __name__ == "__main__":
    main()
