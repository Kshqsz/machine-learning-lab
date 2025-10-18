"""
感知机学习算法 - 原始形式
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS系统中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class PerceptronPrimal:
    """
    感知机原始形式
    模型: f(x) = sign(w·x + b)
    策略: 最小化误分类点到超平面的距离
    算法: 随机梯度下降
    """
    
    def __init__(self, learning_rate=1):
        """
        初始化感知机
        
        参数:
            learning_rate: 学习率 η
        """
        self.learning_rate = learning_rate
        self.w = None
        self.b = None
        self.history = []  # 记录训练过程
        
    def fit(self, X, y, max_iter=1000):
        """
        训练感知机模型
        
        参数:
            X: 训练数据，shape (n_samples, n_features)
            y: 标签，取值 +1 或 -1
            max_iter: 最大迭代次数
        """
        n_samples, n_features = X.shape
        
        # 初始化参数
        self.w = np.zeros(n_features)
        self.b = 0
        
        print("=" * 60)
        print("感知机学习算法 - 原始形式")
        print("=" * 60)
        print(f"学习率 η = {self.learning_rate}")
        print(f"初始参数: w = {self.w}, b = {self.b}")
        print()
        
        iteration = 0
        while iteration < max_iter:
            # 记录是否有误分类点
            has_misclassified = False
            
            # 遍历所有样本点
            for i in range(n_samples):
                # 计算 y_i(w·x_i + b)
                if y[i] * (np.dot(self.w, X[i]) + self.b) <= 0:
                    # 误分类点，更新参数
                    has_misclassified = True
                    
                    # 记录更新前的参数
                    w_old = self.w.copy()
                    b_old = self.b
                    
                    # 梯度下降更新
                    self.w = self.w + self.learning_rate * y[i] * X[i]
                    self.b = self.b + self.learning_rate * y[i]
                    
                    iteration += 1
                    
                    # 记录历史
                    self.history.append({
                        'iteration': iteration,
                        'misclassified_point': i,
                        'x': X[i],
                        'y': y[i],
                        'w_old': w_old,
                        'b_old': b_old,
                        'w_new': self.w.copy(),
                        'b_new': self.b
                    })
                    
                    print(f"第 {iteration} 次迭代:")
                    print(f"  误分类点: x{i+1} = {X[i]}, y{i+1} = {y[i]:+d}")
                    print(f"  更新前: w = {w_old}, b = {b_old}")
                    print(f"  更新后: w = {self.w}, b = {self.b}")
                    print()
                    
                    break  # 找到一个误分类点就更新，然后重新开始
            
            # 如果没有误分类点，说明已经收敛
            if not has_misclassified:
                print("=" * 60)
                print("训练完成！所有点都被正确分类")
                print("=" * 60)
                break
        
        return self
    
    def predict(self, X):
        """
        预测
        
        参数:
            X: 待预测数据
            
        返回:
            预测标签 (+1 或 -1)
        """
        return np.sign(np.dot(X, self.w) + self.b)
    
    def score(self, X, y):
        """
        计算准确率
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


def plot_perceptron(X, y, model, title="感知机原始形式分类结果"):
    """
    可视化感知机分类结果
    """
    plt.figure(figsize=(10, 8))
    
    # 绘制数据点
    for i in range(len(X)):
        if y[i] == 1:
            plt.scatter(X[i, 0], X[i, 1], c='red', s=200, marker='o', 
                       edgecolors='black', linewidths=2, label='正样本' if i == 0 else '')
            plt.text(X[i, 0] + 0.1, X[i, 1] + 0.1, f'x{i+1}{tuple(X[i])}', fontsize=12)
        else:
            plt.scatter(X[i, 0], X[i, 1], c='blue', s=200, marker='s', 
                       edgecolors='black', linewidths=2, label='负样本' if i == 2 else '')
            plt.text(X[i, 0] + 0.1, X[i, 1] + 0.1, f'x{i+1}{tuple(X[i])}', fontsize=12)
    
    # 绘制分类超平面 w·x + b = 0
    # 即 w1*x1 + w2*x2 + b = 0
    # 解出 x2 = -(w1*x1 + b) / w2
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x1_line = np.linspace(x1_min, x1_max, 100)
    
    if model.w[1] != 0:
        x2_line = -(model.w[0] * x1_line + model.b) / model.w[1]
        plt.plot(x1_line, x2_line, 'g-', linewidth=2, 
                label=f'分离超平面: {model.w[0]:.0f}x₁ + {model.w[1]:.0f}x₂ + {model.b:.0f} = 0')
    
    plt.xlabel('x₁', fontsize=14)
    plt.ylabel('x₂', fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # 设置坐标轴范围
    plt.xlim(0, 5)
    plt.ylim(0, 4)
    
    plt.tight_layout()
    plt.savefig('perceptron_primal_result.png', dpi=300, bbox_inches='tight')
    print("\n图表已保存至: perceptron_primal_result.png")
    plt.show()


def main():
    # 训练数据
    # 正样本: x1=(3,3), x2=(4,3)
    # 负样本: x3=(1,1)
    X = np.array([
        [3, 3],  # x1, 正样本
        [4, 3],  # x2, 正样本
        [1, 1]   # x3, 负样本
    ])
    
    y = np.array([1, 1, -1])  # 标签: +1 表示正样本, -1 表示负样本
    
    print("训练数据:")
    print("正样本点: x1=(3,3)ᵀ, x2=(4,3)ᵀ")
    print("负样本点: x3=(1,1)ᵀ")
    print()
    
    # 创建并训练感知机
    perceptron = PerceptronPrimal(learning_rate=1)
    perceptron.fit(X, y)
    
    # 输出最终结果
    print(f"\n最终模型参数:")
    print(f"w = {perceptron.w}")
    print(f"b = {perceptron.b}")
    print(f"\n分离超平面方程: {perceptron.w[0]:.0f}x₁ + {perceptron.w[1]:.0f}x₂ + {perceptron.b:.0f} = 0")
    
    # 验证
    print(f"\n训练集准确率: {perceptron.score(X, y) * 100:.1f}%")
    
    # 可视化
    plot_perceptron(X, y, perceptron)


if __name__ == "__main__":
    main()
