"""
线性支持向量机 - 软间隔 (对偶形式)
Linear Support Vector Machine - Soft Margin (Dual Form)

算法原理:
1. 引入松弛变量 ξ_i ≥ 0，允许部分样本不满足硬间隔约束
2. 对偶问题: max Σα_i - (1/2)ΣΣ α_i α_j y_i y_j (x_i·x_j)
   约束条件: Σ α_i y_i = 0, 0 ≤ α_i ≤ C

3. 参数C控制间隔最大化与误分类的权衡:
   - C越大，对误分类的惩罚越大，越接近硬间隔
   - C越小，允许更多误分类，间隔更大

4. 支持向量分为三类:
   - α_i = 0: 非支持向量（在间隔边界外）
   - 0 < α_i < C: 边界支持向量（在间隔边界上，ξ_i = 0）
   - α_i = C: 内部支持向量（在间隔内或误分类，ξ_i > 0）

适用场景:
- 线性不可分数据
- 存在噪声和异常点的数据
- 需要在间隔最大化和误分类之间权衡
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS
plt.rcParams['axes.unicode_minus'] = False


class LinearSVMSoftMargin:
    """线性支持向量机 - 软间隔"""
    
    def __init__(self, C=1.0):
        """
        参数:
            C: 惩罚参数，控制对误分类的容忍度
               C越大，对误分类惩罚越大，越接近硬间隔
               C越小，允许更多误分类，间隔更大
        """
        self.C = C
        self.w = None          # 权重向量
        self.b = None          # 偏置
        self.alpha = None      # 拉格朗日乘子
        self.support_vectors = None       # 支持向量
        self.support_labels = None        # 支持向量的标签
        self.support_alpha = None         # 支持向量的α值
        self.margin_sv_indices = []       # 边界支持向量索引
        self.inside_sv_indices = []       # 内部支持向量索引
        
    def fit(self, X, y, verbose=True):
        """
        训练SVM模型
        
        参数:
            X: 训练样本特征 (n_samples, n_features)
            y: 训练样本标签 (n_samples,) 取值为+1或-1
        """
        n_samples, n_features = X.shape
        
        if verbose:
            print("="*70)
            print("线性支持向量机 - 软间隔（对偶形式）")
            print("="*70)
            print(f"训练样本数: {n_samples}")
            print(f"特征维度: {n_features}")
            print(f"惩罚参数 C: {self.C}")
            print()
        
        # 计算Gram矩阵 K[i,j] = x_i · x_j
        K = np.zeros((n_samples, n_samples))
        if verbose:
            print("Gram矩阵 (内积矩阵):")
            print("-"*70)
        
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = np.dot(X[i], X[j])
        
        if verbose and n_samples <= 10:
            for i in range(n_samples):
                print(f"  K[{i}] = {K[i]}")
            print()
        
        # 定义对偶问题的目标函数（要最小化负的对偶目标）
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
        
        # 边界条件: 0 ≤ α_i ≤ C (软间隔的关键！)
        bounds = [(0, self.C) for _ in range(n_samples)]
        
        # 初始值
        alpha0 = np.zeros(n_samples)
        
        if verbose:
            print("求解对偶问题...")
            print("-"*70)
            print("目标函数: min -Σα_i + (1/2)ΣΣ α_i α_j y_i y_j (x_i·x_j)")
            print("约束条件: Σ α_i y_i = 0, 0 ≤ α_i ≤ C")
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
        
        if verbose:
            print("优化结果:")
            print("-"*70)
            for i in range(n_samples):
                alpha_val = self.alpha[i]
                if alpha_val < 1e-5:
                    status = "非支持向量"
                elif alpha_val < self.C - 1e-5:
                    status = "边界支持向量"
                else:
                    status = "内部支持向量"
                print(f"  α_{i+1} = {alpha_val:.6f}  ({status})")
            print()
        
        # 找出支持向量
        sv_indices = self.alpha > 1e-5
        self.support_vectors = X[sv_indices]
        self.support_labels = y[sv_indices]
        self.support_alpha = self.alpha[sv_indices]
        
        # 区分边界支持向量和内部支持向量
        self.margin_sv_indices = np.where((self.alpha > 1e-5) & (self.alpha < self.C - 1e-5))[0]
        self.inside_sv_indices = np.where(self.alpha >= self.C - 1e-5)[0]
        
        if verbose:
            print(f"支持向量总数: {len(self.support_vectors)}")
            print(f"  - 边界支持向量: {len(self.margin_sv_indices)} 个 (0 < α < C)")
            print(f"  - 内部支持向量: {len(self.inside_sv_indices)} 个 (α = C)")
            print("-"*70)
            
            if len(self.margin_sv_indices) > 0:
                print("边界支持向量 (在间隔边界上):")
                for idx in self.margin_sv_indices:
                    print(f"  x{idx+1} = {X[idx]}, y = {y[idx]:+d}, α = {self.alpha[idx]:.6f}")
            
            if len(self.inside_sv_indices) > 0:
                print("\n内部支持向量 (在间隔内或误分类):")
                for idx in self.inside_sv_indices:
                    print(f"  x{idx+1} = {X[idx]}, y = {y[idx]:+d}, α = {self.alpha[idx]:.6f}")
            print()
        
        # 计算权重向量 w = Σ α_i y_i x_i
        self.w = np.sum(
            self.alpha[:, None] * y[:, None] * X,
            axis=0
        )
        
        if verbose:
            print("权重向量:")
            print("-"*70)
            print(f"  w = {self.w}")
            print(f"  ||w|| = {np.linalg.norm(self.w):.6f}")
            print()
        
        # 计算偏置 b，使用边界支持向量（0 < α < C）
        # b = y_j - w·x_j
        if len(self.margin_sv_indices) > 0:
            # 使用所有边界支持向量的平均值来计算b，更稳定
            b_values = []
            for idx in self.margin_sv_indices:
                b_val = y[idx] - np.sum(self.alpha * y * K[:, idx])
                b_values.append(b_val)
            self.b = np.mean(b_values)
        else:
            # 如果没有边界支持向量，使用所有支持向量
            print("警告: 没有边界支持向量，使用所有支持向量计算偏置")
            sv_idx = np.where(sv_indices)[0][0]
            self.b = y[sv_idx] - np.sum(self.alpha * y * K[:, sv_idx])
        
        if verbose:
            print("偏置:")
            print("-"*70)
            print(f"  b = {self.b:.6f}")
            print()
        
        # 计算间隔
        margin = 2.0 / np.linalg.norm(self.w)
        
        if verbose:
            print("分类间隔:")
            print("-"*70)
            print(f"  间隔 = 2/||w|| = {margin:.6f}")
            print()
        
        if verbose:
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
    
    def plot_decision_boundary(self, X, y, title_suffix=""):
        """绘制决策边界和支持向量"""
        if X.shape[1] != 2:
            print("只支持2维特征的可视化")
            return
        
        plt.figure(figsize=(12, 8))
        
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
        
        # 标记边界支持向量（绿色圆圈）
        if len(self.margin_sv_indices) > 0:
            plt.scatter(
                X[self.margin_sv_indices, 0],
                X[self.margin_sv_indices, 1],
                s=250, linewidths=2.5,
                facecolors='none', edgecolors='green',
                label=f'边界支持向量 ({len(self.margin_sv_indices)}个)',
                zorder=4
            )
        
        # 标记内部支持向量（橙色叉号）
        if len(self.inside_sv_indices) > 0:
            plt.scatter(
                X[self.inside_sv_indices, 0],
                X[self.inside_sv_indices, 1],
                s=200, marker='x', linewidths=3,
                c='orange',
                label=f'内部支持向量 ({len(self.inside_sv_indices)}个)',
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
            linestyles='-', linewidths=2.5
        )
        
        # 绘制间隔边界 (f(x) = ±1)
        plt.contour(
            XX, YY, Z,
            colors='gray', levels=[-1, 1],
            linestyles='--', linewidths=2
        )
        
        # 绘制决策区域
        plt.contourf(
            XX, YY, Z,
            levels=[-np.inf, 0, np.inf],
            colors=['lightblue', 'lightcoral'],
            alpha=0.3
        )
        
        plt.xlabel('x₁', fontsize=13)
        plt.ylabel('x₂', fontsize=13)
        title = f'线性支持向量机 - 软间隔 (C={self.C})'
        if title_suffix:
            title += f' - {title_suffix}'
        plt.title(title, fontsize=15, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim(xlim)
        plt.ylim(ylim)
        
        # 添加文本说明
        text = f'C = {self.C}\n'
        text += f'w = ({self.w[0]:.3f}, {self.w[1]:.3f})\n'
        text += f'b = {self.b:.3f}\n'
        text += f'间隔 = {2.0/np.linalg.norm(self.w):.3f}\n'
        text += f'支持向量: {len(self.support_vectors)}个'
        plt.text(
            0.02, 0.98, text,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
        
        plt.tight_layout()
        filename = f'svm/svm_soft_margin_C{self.C}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n图像已保存至: {filename}")
        plt.show()


def example1_linearly_separable():
    """示例1: 线性可分数据（与硬间隔对比）"""
    print("\n" + "="*70)
    print("示例1: 线性可分数据")
    print("="*70 + "\n")
    
    # 与硬间隔SVM相同的数据
    X_train = np.array([
        [3, 3],
        [4, 3],
        [1, 1]
    ])
    y_train = np.array([1, 1, -1])
    
    print("训练数据:")
    print("-"*70)
    for i, (x, label) in enumerate(zip(X_train, y_train)):
        category = "正例" if label == 1 else "负例"
        print(f"  x{i+1} = {x}  →  y = {label:+d}  ({category})")
    print()
    
    # 使用较大的C值（接近硬间隔）
    svm = LinearSVMSoftMargin(C=100.0)
    svm.fit(X_train, y_train)
    
    # 预测
    print("\n训练集预测:")
    print("-"*70)
    for i, (x, y_true) in enumerate(zip(X_train, y_train)):
        y_pred = int(svm.predict(x.reshape(1, -1))[0])
        score = svm.decision_function(x.reshape(1, -1))[0]
        result = "✓" if y_pred == y_true else "✗"
        print(f"  x{i+1} = {x}  →  f(x) = {score:+.6f}  →  预测: {y_pred:+d}  真实: {y_true:+d}  {result}")
    
    accuracy = np.mean(svm.predict(X_train) == y_train) * 100
    print(f"\n训练集准确率: {accuracy:.2f}%\n")
    
    # 可视化
    svm.plot_decision_boundary(X_train, y_train, "线性可分")


def example2_noisy_data():
    """示例2: 含噪声数据（需要软间隔）"""
    print("\n" + "="*70)
    print("示例2: 含噪声数据")
    print("="*70 + "\n")
    
    # 基本线性可分的数据，但有一个噪声点
    X_train = np.array([
        [3, 3],
        [4, 3],
        [3.5, 3.5],
        [1, 1],
        [1.5, 1],
        [2.8, 2.5]  # 噪声点：正类区域中的负例
    ])
    y_train = np.array([1, 1, 1, -1, -1, -1])
    
    print("训练数据 (包含噪声点):")
    print("-"*70)
    for i, (x, label) in enumerate(zip(X_train, y_train)):
        category = "正例" if label == 1 else "负例"
        noise = " [噪声点]" if i == 5 else ""
        print(f"  x{i+1} = {x}  →  y = {label:+d}  ({category}){noise}")
    print()
    
    # 尝试不同的C值
    for C in [0.1, 1.0, 10.0]:
        print(f"\n{'='*70}")
        print(f"C = {C}")
        print('='*70 + "\n")
        
        svm = LinearSVMSoftMargin(C=C)
        svm.fit(X_train, y_train, verbose=(C == 1.0))  # 只在C=1时显示详细信息
        
        # 预测
        y_pred = svm.predict(X_train)
        accuracy = np.mean(y_pred == y_train) * 100
        
        print(f"\n训练集准确率 (C={C}): {accuracy:.2f}%")
        
        # 检查每个样本的分类情况
        print("样本分类情况:")
        for i, (x, y_true, y_p) in enumerate(zip(X_train, y_train, y_pred)):
            score = svm.decision_function(x.reshape(1, -1))[0]
            result = "✓" if y_p == y_true else "✗"
            print(f"  x{i+1}: f(x)={score:+.3f}, 预测={int(y_p):+d}, 真实={y_true:+d} {result}")
        
        # 可视化
        svm.plot_decision_boundary(X_train, y_train, f"含噪声")


def example3_overlapping_classes():
    """示例3: 类别重叠数据"""
    print("\n" + "="*70)
    print("示例3: 类别重叠数据")
    print("="*70 + "\n")
    
    np.random.seed(42)
    
    # 生成两类数据，有明显重叠
    n_samples = 20
    
    # 正类：中心在(3, 3)附近
    X_pos = np.random.randn(n_samples // 2, 2) * 0.8 + [3, 3]
    y_pos = np.ones(n_samples // 2)
    
    # 负类：中心在(1, 1)附近
    X_neg = np.random.randn(n_samples // 2, 2) * 0.8 + [1, 1]
    y_neg = -np.ones(n_samples // 2)
    
    X_train = np.vstack([X_pos, X_neg])
    y_train = np.hstack([y_pos, y_neg])
    
    print(f"训练数据: {n_samples}个样本，两类有重叠")
    print()
    
    # 比较不同C值的效果
    C_values = [0.1, 1.0, 100.0]
    
    for C in C_values:
        print(f"\n{'='*70}")
        print(f"惩罚参数 C = {C}")
        print('='*70)
        
        svm = LinearSVMSoftMargin(C=C)
        svm.fit(X_train, y_train, verbose=False)
        
        # 预测
        y_pred = svm.predict(X_train)
        accuracy = np.mean(y_pred == y_train) * 100
        
        print(f"\n训练集准确率: {accuracy:.2f}%")
        print(f"支持向量数: {len(svm.support_vectors)}")
        print(f"  - 边界支持向量: {len(svm.margin_sv_indices)}")
        print(f"  - 内部支持向量: {len(svm.inside_sv_indices)}")
        
        # 可视化
        svm.plot_decision_boundary(X_train, y_train, f"类别重叠")


def main():
    """主函数：运行所有示例"""
    print("\n" + "="*70)
    print("线性支持向量机 - 软间隔算法演示")
    print("="*70)
    
    # 示例1: 线性可分数据
    example1_linearly_separable()
    
    # 示例2: 含噪声数据
    example2_noisy_data()
    
    # 示例3: 类别重叠数据
    example3_overlapping_classes()
    
    print("\n" + "="*70)
    print("所有示例运行完成！")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
