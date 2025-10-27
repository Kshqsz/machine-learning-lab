"""
线性支持向量机 - 随机梯度下降 (SGD)
Linear Support Vector Machine - Stochastic Gradient Descent

算法原理:
1. 使用Hinge损失函数: L = max(0, 1 - y_i(w·x_i + b))
2. 完整目标函数: J = (1/2)||w||² + C·Σ max(0, 1 - y_i(w·x_i + b))
3. 随机梯度下降更新规则:
   - 如果 y_i(w·x_i + b) < 1: 
     w ← w - η(w - C·y_i·x_i)
     b ← b - η(-C·y_i)
   - 否则:
     w ← w - η·w
     b 不更新

特点:
- 在线学习算法，每次只使用一个样本更新参数
- 计算效率高，适合大规模数据
- 学习率需要仔细调整
- 支持软间隔（通过参数C控制）

适用场景:
- 大规模线性分类问题
- 需要在线学习或增量学习
- 内存受限的场景
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS
plt.rcParams['axes.unicode_minus'] = False


class LinearSVM_SGD:
    """线性支持向量机 - 随机梯度下降"""
    
    def __init__(self, C=1.0, learning_rate=0.01, n_epochs=10000, random_state=42):
        """
        参数:
            C: 正则化参数（惩罚参数）
            learning_rate: 学习率
            n_epochs: 训练轮数
            random_state: 随机种子
        """
        self.C = C
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.w = None
        self.b = None
        self.loss_history = []
        
    def _compute_loss(self, X, y):
        """
        计算总损失
        
        损失函数: J = (1/2)||w||² + C·Σ max(0, 1 - y_i(w·x_i + b))
        """
        n_samples = X.shape[0]
        
        # 正则化项
        reg_loss = 0.5 * np.sum(self.w ** 2)
        
        # Hinge损失
        margins = y * (np.dot(X, self.w) + self.b)
        hinge_loss = np.sum(np.maximum(0, 1 - margins))
        
        total_loss = reg_loss + self.C * hinge_loss / n_samples
        
        return total_loss
    
    def fit(self, X, y, verbose=True):
        """
        使用随机梯度下降训练SVM
        
        参数:
            X: 训练样本 (n_samples, n_features)
            y: 标签 (n_samples,) 取值为+1或-1
        """
        n_samples, n_features = X.shape
        
        if verbose:
            print("="*70)
            print("线性支持向量机 - 随机梯度下降 (SGD)")
            print("="*70)
            print(f"训练样本数: {n_samples}")
            print(f"特征维度: {n_features}")
            print(f"惩罚参数 C: {self.C}")
            print(f"学习率: {self.learning_rate}")
            print(f"训练轮数: {self.n_epochs}")
            print()
        
        # 初始化参数
        np.random.seed(self.random_state)
        self.w = np.zeros(n_features)
        self.b = 0.0
        
        self.loss_history = []
        
        if verbose:
            print("开始随机梯度下降训练...")
            print("-"*70)
        
        # SGD训练
        for epoch in range(self.n_epochs):
            # 随机打乱样本顺序
            indices = np.random.permutation(n_samples)
            
            # 对每个样本进行更新
            for idx in indices:
                xi = X[idx]
                yi = y[idx]
                
                # 计算间隔
                margin = yi * (np.dot(self.w, xi) + self.b)
                
                # 更新参数
                if margin < 1:
                    # 样本在间隔内或误分类，需要惩罚
                    # ∂L/∂w = w - C·y_i·x_i
                    # ∂L/∂b = -C·y_i
                    self.w = self.w - self.learning_rate * (self.w - self.C * yi * xi)
                    self.b = self.b - self.learning_rate * (-self.C * yi)
                else:
                    # 样本在间隔外，只更新正则化项
                    # ∂L/∂w = w
                    self.w = self.w - self.learning_rate * self.w
                    # b 不更新
            
            # 计算并记录损失
            loss = self._compute_loss(X, y)
            self.loss_history.append(loss)
            
            # 打印进度
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1:4d}/{self.n_epochs} | 损失: {loss:.6f}")
        
        if verbose:
            print("-"*70)
            print(f"训练完成！最终损失: {self.loss_history[-1]:.6f}")
            print()
            
            print("学习到的参数:")
            print("-"*70)
            print(f"权重向量 w: {self.w}")
            print(f"偏置 b: {self.b:.6f}")
            print(f"||w||: {np.linalg.norm(self.w):.6f}")
            print()
            
            # 计算间隔
            margin = 2.0 / np.linalg.norm(self.w)
            print(f"分类间隔: {margin:.6f}")
            print()
            
            print("决策边界:")
            print("-"*70)
            if n_features == 2:
                print(f"{self.w[0]:.6f}·x₁ + {self.w[1]:.6f}·x₂ + {self.b:.6f} = 0")
            print("="*70)
    
    def predict(self, X):
        """预测样本类别"""
        return np.sign(np.dot(X, self.w) + self.b)
    
    def decision_function(self, X):
        """计算决策函数值"""
        return np.dot(X, self.w) + self.b
    
    def plot_decision_boundary(self, X, y, title=""):
        """可视化决策边界"""
        if X.shape[1] != 2:
            print("只支持2维特征的可视化")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 左图：决策边界
        # 绘制数据点
        for label, marker, color in [(1, 'o', 'red'), (-1, 's', 'blue')]:
            mask = y == label
            ax1.scatter(
                X[mask, 0], X[mask, 1],
                c=color, marker=marker, s=120,
                edgecolors='black', linewidths=1.5,
                label=f'类别 {label:+d}',
                zorder=3
            )
        
        # 创建网格
        xlim = ax1.get_xlim()
        ylim = ax1.get_ylim()
        xx = np.linspace(xlim[0] - 1, xlim[1] + 1, 200)
        yy = np.linspace(ylim[0] - 1, ylim[1] + 1, 200)
        XX, YY = np.meshgrid(xx, yy)
        xy = np.c_[XX.ravel(), YY.ravel()]
        
        # 计算决策函数值
        Z = self.decision_function(xy).reshape(XX.shape)
        
        # 绘制决策边界
        ax1.contour(
            XX, YY, Z,
            colors='black', levels=[0],
            linestyles='-', linewidths=2.5
        )
        
        # 绘制间隔边界
        ax1.contour(
            XX, YY, Z,
            colors='gray', levels=[-1, 1],
            linestyles='--', linewidths=2
        )
        
        # 绘制决策区域
        ax1.contourf(
            XX, YY, Z,
            levels=[-np.inf, 0, np.inf],
            colors=['lightblue', 'lightcoral'],
            alpha=0.3
        )
        
        ax1.set_xlabel('x₁', fontsize=13)
        ax1.set_ylabel('x₂', fontsize=13)
        ax1.set_title(f'决策边界 - SGD线性SVM{title}', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # 添加文本说明
        text = f'C = {self.C}\n'
        text += f'学习率 = {self.learning_rate}\n'
        text += f'训练轮数 = {self.n_epochs}\n'
        text += f'w = ({self.w[0]:.3f}, {self.w[1]:.3f})\n'
        text += f'b = {self.b:.3f}\n'
        text += f'间隔 = {2.0/np.linalg.norm(self.w):.3f}'
        ax1.text(
            0.02, 0.98, text,
            transform=ax1.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
        
        # 右图：损失曲线
        ax2.plot(self.loss_history, linewidth=2, color='darkblue')
        ax2.set_xlabel('迭代轮数 (Epoch)', fontsize=13)
        ax2.set_ylabel('损失值', fontsize=13)
        ax2.set_title('训练损失曲线', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, len(self.loss_history))
        
        plt.tight_layout()
        filename = f'svm_sgd_linear_C{self.C}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n图像已保存至: {filename}")
        plt.show()


def main():
    """主函数"""
    print("\n线性支持向量机 - 随机梯度下降算法演示\n")
    
    # 训练数据
    X_train = np.array([
        [5, 9],      # 正例
        [3, 12],     # 正例
        [-1, 12],    # 正例
        [2, 10],     # 正例
        [1, 12],     # 正例
        [2, -3],     # 正例
        [4, 4.5],    # 正例
        [1, 1],      # 负例
        [1.5, -3],   # 负例
        [3, -2],     # 负例
        [4, -5],     # 负例
        [3.5, 8],    # 负例
        [-1, 4],     # 负例
        [-1, -1]     # 负例
    ])
    
    y_train = np.array([1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1])
    
    print("训练数据:")
    print("-"*70)
    for i, (x, label) in enumerate(zip(X_train, y_train)):
        category = "正例" if label == 1 else "负例"
        print(f"  样本 {i+1:2d}: x = ({x[0]:5.1f}, {x[1]:5.1f})  →  y = {label:+d}  ({category})")
    print()
    
    # 创建并训练SVM
    svm = LinearSVM_SGD(C=0.5, learning_rate=0.01, n_epochs=1000, random_state=42)
    svm.fit(X_train, y_train)
    
    # 在训练集上评估
    print("\n训练集预测结果:")
    print("-"*70)
    y_pred = svm.predict(X_train)
    
    for i, (x, y_true, y_p) in enumerate(zip(X_train, y_train, y_pred)):
        score = svm.decision_function(x.reshape(1, -1))[0]
        result = "✓" if y_p == y_true else "✗"
        category = "正例" if y_true == 1 else "负例"
        print(f"  样本 {i+1:2d} ({category}): f(x) = {score:+8.4f}  →  预测: {int(y_p):+d}  真实: {y_true:+d}  {result}")
    
    # 计算准确率
    accuracy = np.mean(y_pred == y_train) * 100
    print(f"\n训练集准确率: {accuracy:.2f}%")
    
    # 统计支持向量
    margins = y_train * svm.decision_function(X_train)
    support_vectors_mask = margins <= 1.0
    n_support_vectors = np.sum(support_vectors_mask)
    
    print(f"\n支持向量信息:")
    print("-"*70)
    print(f"支持向量数量: {n_support_vectors}")
    if n_support_vectors > 0:
        print("支持向量索引:", np.where(support_vectors_mask)[0] + 1)
    
    # 测试新样本
    print("\n新样本预测:")
    print("-"*70)
    X_test = np.array([
        [0, 10],
        [3, 5],
        [2, -4]
    ])
    
    for x in X_test:
        y_pred = int(svm.predict(x.reshape(1, -1))[0])
        score = svm.decision_function(x.reshape(1, -1))[0]
        category = "正例" if y_pred == 1 else "负例"
        print(f"  x = ({x[0]:5.1f}, {x[1]:5.1f})  →  f(x) = {score:+8.4f}  →  预测: {y_pred:+d}  ({category})")
    print()
    
    # 可视化
    svm.plot_decision_boundary(X_train, y_train)
    
    # 比较不同参数的效果
    print("\n" + "="*70)
    print("比较不同C值的效果")
    print("="*70)
    
    for C in [0.1, 1.0, 10.0]:
        print(f"\nC = {C}:")
        svm_temp = LinearSVM_SGD(C=C, learning_rate=0.01, n_epochs=1000, random_state=42)
        svm_temp.fit(X_train, y_train, verbose=False)
        
        y_pred_temp = svm_temp.predict(X_train)
        accuracy_temp = np.mean(y_pred_temp == y_train) * 100
        
        margins_temp = y_train * svm_temp.decision_function(X_train)
        n_sv_temp = np.sum(margins_temp <= 1.0)
        
        print(f"  准确率: {accuracy_temp:.2f}%")
        print(f"  支持向量数: {n_sv_temp}")
        print(f"  最终损失: {svm_temp.loss_history[-1]:.6f}")
        print(f"  ||w||: {np.linalg.norm(svm_temp.w):.6f}")
        
        # 可视化
        svm_temp.plot_decision_boundary(X_train, y_train, f" (C={C})")
    
    print("\n" + "="*70)
    print("所有示例运行完成！")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
