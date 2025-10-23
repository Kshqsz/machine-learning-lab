"""
多项逻辑斯谛回归 - BFGS优化算法
参考：《机器学习方法（第2版）》李航

使用BFGS拟牛顿法求解多项逻辑斯谛回归模型参数
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs
from collections import Counter

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS系统中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class MultinomialLogisticRegression:
    """
    多项逻辑斯谛回归分类器 - 使用BFGS算法
    
    模型: P(Y=k|x) = exp(w_k·x) / Σ_j exp(w_j·x)
    损失函数: 负对数似然
    优化方法: BFGS拟牛顿法
    """
    
    def __init__(self, max_iter=100):
        """
        初始化多项逻辑回归模型
        
        参数:
            max_iter: BFGS最大迭代次数
        """
        self.max_iter = max_iter
        self.W = None  # 权重矩阵 (n_features, n_classes)
        self.classes_ = None
        self.n_classes = None
        self.n_features = None
        self.loss_history = []
        
    def softmax(self, z):
        """
        Softmax函数
        softmax(z_k) = exp(z_k) / Σ_j exp(z_j)
        
        参数:
            z: (n_samples, n_classes)
        返回:
            概率矩阵 (n_samples, n_classes)
        """
        # 数值稳定性：减去最大值
        z_max = np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z - z_max)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def negative_log_likelihood(self, W_flat, X, y_onehot):
        """
        计算负对数似然（损失函数）
        
        L(W) = -Σ_i Σ_k y_ik * log(p_ik)
        
        参数:
            W_flat: 展平的权重向量
            X: 特征矩阵 (n_samples, n_features)
            y_onehot: one-hot编码的标签 (n_samples, n_classes)
        """
        # 重塑权重矩阵
        W = W_flat.reshape(self.n_features, self.n_classes)
        
        # 计算预测概率
        z = np.dot(X, W)  # (n_samples, n_classes)
        p = self.softmax(z)
        
        # 避免log(0)
        epsilon = 1e-10
        p = np.clip(p, epsilon, 1 - epsilon)
        
        # 负对数似然
        loss = -np.sum(y_onehot * np.log(p)) / X.shape[0]
        
        # 记录损失
        self.loss_history.append(loss)
        
        return loss
    
    def gradient(self, W_flat, X, y_onehot):
        """
        计算梯度
        
        ∂L/∂W_k = (1/n) * Σ_i (p_ik - y_ik) * x_i
        
        参数:
            W_flat: 展平的权重向量
            X: 特征矩阵 (n_samples, n_features)
            y_onehot: one-hot编码的标签 (n_samples, n_classes)
        """
        # 重塑权重矩阵
        W = W_flat.reshape(self.n_features, self.n_classes)
        
        # 计算预测概率
        z = np.dot(X, W)
        p = self.softmax(z)
        
        # 计算梯度
        grad = np.dot(X.T, (p - y_onehot)) / X.shape[0]
        
        return grad.flatten()
    
    def fit(self, X, y, verbose=True):
        """
        使用BFGS算法训练模型
        
        参数:
            X: 训练特征 (n_samples, n_features)
            y: 训练标签 (n_samples,)
            verbose: 是否打印训练过程
        """
        n_samples, n_features = X.shape
        self.n_features = n_features
        
        # 获取类别信息
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        
        # 将标签转换为one-hot编码
        y_onehot = np.zeros((n_samples, self.n_classes))
        for i, c in enumerate(self.classes_):
            y_onehot[y == c, i] = 1
        
        if verbose:
            print("="*70)
            print("多项逻辑斯谛回归 - BFGS优化算法")
            print("="*70)
            print(f"训练样本数: {n_samples}")
            print(f"特征维度: {n_features}")
            print(f"类别数: {self.n_classes}")
            print(f"类别: {self.classes_}")
            print(f"最大迭代次数: {self.max_iter}")
            print("-"*70)
        
        # 初始化权重
        W_init = np.zeros((n_features, self.n_classes))
        W_flat_init = W_init.flatten()
        
        # 清空损失历史
        self.loss_history = []
        
        # 使用BFGS优化
        if verbose:
            print("开始BFGS优化...")
            print()
        
        W_flat_opt = fmin_bfgs(
            f=self.negative_log_likelihood,
            x0=W_flat_init,
            fprime=self.gradient,
            args=(X, y_onehot),
            maxiter=self.max_iter,
            disp=verbose
        )
        
        # 保存优化后的权重
        self.W = W_flat_opt.reshape(n_features, self.n_classes)
        
        if verbose:
            print("-"*70)
            print(f"训练完成！")
            print(f"最终损失: {self.loss_history[-1]:.6f}")
            print(f"权重矩阵形状: {self.W.shape}")
            print("="*70)
    
    def predict_proba(self, X):
        """
        预测概率分布
        """
        z = np.dot(X, self.W)
        return self.softmax(z)
    
    def predict(self, X):
        """
        预测类别
        """
        proba = self.predict_proba(X)
        y_pred_idx = np.argmax(proba, axis=1)
        return self.classes_[y_pred_idx]
    
    def score(self, X, y):
        """
        计算准确率
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


def plot_results(X, y, model, feature_names=None):
    """
    可视化多分类结果（2D特征）
    """
    if X.shape[1] != 2:
        print("跳过可视化：仅支持2维特征")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 子图1: 决策边界
    ax1 = axes[0]
    
    # 创建网格
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # 预测网格点
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界
    from matplotlib.colors import ListedColormap
    colors = ['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FFFFAA', '#FFAAFF', '#AAFFFF']
    cmap = ListedColormap(colors[:model.n_classes])
    ax1.contourf(xx, yy, Z, alpha=0.3, cmap=cmap, levels=model.n_classes-1)
    
    # 绘制数据点
    scatter_colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan']
    markers = ['o', 's', '^', 'D', 'v', '<']
    
    for i, c in enumerate(model.classes_):
        mask = y == c
        ax1.scatter(X[mask, 0], X[mask, 1], 
                   c=scatter_colors[i], marker=markers[i], 
                   s=100, label=f'类别 {c}', 
                   edgecolors='black', linewidths=1.5)
    
    if feature_names:
        ax1.set_xlabel(feature_names[0], fontsize=12, fontfamily='sans-serif')
        ax1.set_ylabel(feature_names[1], fontsize=12, fontfamily='sans-serif')
    else:
        ax1.set_xlabel('特征 1', fontsize=12, fontfamily='sans-serif')
        ax1.set_ylabel('特征 2', fontsize=12, fontfamily='sans-serif')
    
    ax1.set_title('多项逻辑斯谛回归 - 决策边界', fontsize=14, 
                 fontweight='bold', fontfamily='sans-serif')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 损失函数曲线
    ax2 = axes[1]
    if len(model.loss_history) > 0:
        ax2.plot(model.loss_history, 'b-', linewidth=2)
        ax2.set_xlabel('迭代次数', fontsize=12, fontfamily='sans-serif')
        ax2.set_ylabel('负对数似然损失', fontsize=12, fontfamily='sans-serif')
        ax2.set_title('BFGS优化过程', fontsize=14, fontweight='bold', fontfamily='sans-serif')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    import os
    save_path = 'multinomial_logistic_regression_result.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存至: {os.path.abspath(save_path)}")
    plt.show()


def main():
    """
    主函数 - 鸢尾花数据集分类
    """
    print("\n训练数据集 - 鸢尾花分类 (简化版):")
    print("-"*70)
    
    # 简化的鸢尾花数据集（3类，2个特征）
    # 特征: 花瓣长度(cm), 花瓣宽度(cm)
    # 类别: 0=Setosa, 1=Versicolor, 2=Virginica
    data = np.array([
        # Setosa (类别 0)
        [1.4, 0.2, 0], [1.4, 0.2, 0], [1.3, 0.2, 0], [1.5, 0.2, 0],
        [1.4, 0.2, 0], [1.7, 0.4, 0], [1.4, 0.3, 0], [1.5, 0.2, 0],
        [1.4, 0.2, 0], [1.5, 0.1, 0], [1.5, 0.2, 0], [1.6, 0.2, 0],
        [1.4, 0.1, 0], [1.1, 0.1, 0], [1.2, 0.2, 0], [1.5, 0.4, 0],
        
        # Versicolor (类别 1)
        [4.7, 1.4, 1], [4.5, 1.5, 1], [4.9, 1.5, 1], [4.0, 1.3, 1],
        [4.6, 1.5, 1], [4.5, 1.3, 1], [4.7, 1.6, 1], [3.3, 1.0, 1],
        [4.6, 1.3, 1], [3.9, 1.4, 1], [3.5, 1.0, 1], [4.2, 1.5, 1],
        [4.0, 1.0, 1], [4.7, 1.4, 1], [3.6, 1.3, 1], [4.4, 1.4, 1],
        
        # Virginica (类别 2)
        [6.0, 2.5, 2], [5.1, 1.9, 2], [5.9, 2.1, 2], [5.6, 1.8, 2],
        [5.8, 2.2, 2], [6.6, 2.1, 2], [4.5, 1.7, 2], [6.3, 1.8, 2],
        [5.8, 1.8, 2], [6.1, 2.5, 2], [5.1, 2.0, 2], [5.3, 1.9, 2],
        [5.5, 2.1, 2], [5.0, 2.0, 2], [5.1, 1.8, 2], [5.3, 2.3, 2],
    ])
    
    X = data[:, :2]  # 花瓣长度和宽度
    y = data[:, 2].astype(int)  # 类别
    
    class_names = ['山鸢尾(Setosa)', '变色鸢尾(Versicolor)', '维吉尼亚鸢尾(Virginica)']
    
    print("样本数:", len(X))
    print("特征: 花瓣长度(cm), 花瓣宽度(cm)")
    print("\n各类别样本数:")
    for i, name in enumerate(class_names):
        count = np.sum(y == i)
        print(f"  {name}: {count} 个样本")
    print("-"*70)
    print()
    
    # 创建并训练模型
    model = MultinomialLogisticRegression(max_iter=100)
    model.fit(X, y, verbose=True)
    
    # 模型评估
    print("\n模型评估:")
    print("="*70)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    
    accuracy = model.score(X, y)
    print(f"训练集准确率: {accuracy * 100:.2f}%")
    
    # 每个类别的准确率
    print("\n各类别准确率:")
    for i, name in enumerate(class_names):
        mask = y == i
        if np.sum(mask) > 0:
            acc = np.mean(y_pred[mask] == y[mask])
            print(f"  {name}: {acc * 100:.2f}%")
    
    # 混淆矩阵
    print("\n混淆矩阵:")
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, y_pred)
    print("         预测->")
    print("实际  ", end="")
    for i in range(model.n_classes):
        print(f"  类{i}  ", end="")
    print()
    for i in range(model.n_classes):
        print(f"类{i}  ", end="")
        for j in range(model.n_classes):
            print(f"  {cm[i,j]:3d}  ", end="")
        print()
    
    # 显示部分预测结果
    print("\n部分预测结果示例 (前10个样本):")
    print("-"*70)
    print("样本 | 花瓣长 | 花瓣宽 | 真实类别 | 预测类别 | Setosa | Versic | Virgin")
    print("-"*70)
    for i in range(min(10, len(X))):
        print(f"{i+1:4d} | {X[i,0]:6.1f} | {X[i,1]:6.1f} | "
              f"{y[i]:^8d} | {y_pred[i]:^8d} | "
              f"{y_proba[i,0]:6.3f} | {y_proba[i,1]:6.3f} | {y_proba[i,2]:6.3f}")
    print("-"*70)
    
    # 测试新样本
    print("\n测试新样本预测:")
    print("="*70)
    test_samples = np.array([
        [1.5, 0.2],  # 应该是 Setosa
        [4.5, 1.4],  # 应该是 Versicolor
        [5.8, 2.0],  # 应该是 Virginica
    ])
    
    test_proba = model.predict_proba(test_samples)
    test_pred = model.predict(test_samples)
    
    for i, sample in enumerate(test_samples):
        pred_class = test_pred[i]
        print(f"\n样本 {i+1}: 花瓣长度={sample[0]:.1f}cm, 花瓣宽度={sample[1]:.1f}cm")
        print(f"  预测类别: {pred_class} ({class_names[pred_class]})")
        print(f"  概率分布: Setosa={test_proba[i,0]:.3f}, "
              f"Versicolor={test_proba[i,1]:.3f}, "
              f"Virginica={test_proba[i,2]:.3f}")
    
    print("\n模型权重矩阵:")
    print("-"*70)
    print("      特征1(长度)  特征2(宽度)")
    for i in range(model.n_classes):
        print(f"类{i}:  {model.W[0,i]:12.4f}  {model.W[1,i]:12.4f}")
    print("="*70)
    
    # 可视化结果
    plot_results(X, y, model, feature_names=['花瓣长度(cm)', '花瓣宽度(cm)'])


if __name__ == "__main__":
    main()
