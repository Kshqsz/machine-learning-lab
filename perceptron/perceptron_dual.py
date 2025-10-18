"""
感知机学习算法 - 对偶形式
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS系统中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class PerceptronDual:
    """
    感知机对偶形式
    模型: f(x) = sign(Σ(αᵢyᵢxᵢ)·x + b)
    对偶形式通过存储每个样本的更新次数 α 来表示模型
    """
    
    def __init__(self, learning_rate=1):
        """
        初始化感知机对偶形式
        
        参数:
            learning_rate: 学习率 η
        """
        self.learning_rate = learning_rate
        self.alpha = None  # 每个样本的更新次数
        self.b = None
        self.X_train = None  # 保存训练数据用于预测
        self.y_train = None
        self.gram_matrix = None  # Gram矩阵
        self.history = []
        
    def compute_gram_matrix(self, X):
        """
        计算 Gram 矩阵 G = [xᵢ·xⱼ]
        """
        n_samples = X.shape[0]
        G = np.zeros((n_samples, n_samples))
        
        print("计算 Gram 矩阵:")
        for i in range(n_samples):
            for j in range(n_samples):
                G[i, j] = np.dot(X[i], X[j])
                print(f"  x{i+1}·x{j+1} = {X[i]} · {X[j]} = {G[i, j]:.0f}")
        
        print(f"\nGram 矩阵 G:")
        print(G)
        print()
        
        return G
        
    def fit(self, X, y, max_iter=1000):
        """
        训练感知机模型 - 对偶形式
        
        参数:
            X: 训练数据，shape (n_samples, n_features)
            y: 标签，取值 +1 或 -1
            max_iter: 最大迭代次数
        """
        n_samples = X.shape[0]
        
        # 保存训练数据
        self.X_train = X
        self.y_train = y
        
        # 初始化参数
        self.alpha = np.zeros(n_samples)
        self.b = 0
        
        # 预计算 Gram 矩阵
        self.gram_matrix = self.compute_gram_matrix(X)
        
        print("=" * 60)
        print("感知机学习算法 - 对偶形式")
        print("=" * 60)
        print(f"学习率 η = {self.learning_rate}")
        print(f"初始参数: α = {self.alpha}, b = {self.b}")
        print()
        
        iteration = 0
        while iteration < max_iter:
            has_misclassified = False
            
            # 遍历所有样本点
            for i in range(n_samples):
                # 计算 Σ(αⱼyⱼxⱼ·xᵢ) + b
                # 利用 Gram 矩阵: Σ(αⱼyⱼ(xⱼ·xᵢ))
                sum_term = np.sum(self.alpha * y * self.gram_matrix[:, i])
                
                # 判断是否误分类: yᵢ(Σ(αⱼyⱼxⱼ·xᵢ) + b) <= 0
                if y[i] * (sum_term + self.b) <= 0:
                    has_misclassified = True
                    
                    # 记录更新前的参数
                    alpha_old = self.alpha.copy()
                    b_old = self.b
                    
                    # 更新参数
                    self.alpha[i] += self.learning_rate
                    self.b = self.b + self.learning_rate * y[i]
                    
                    iteration += 1
                    
                    # 记录历史
                    self.history.append({
                        'iteration': iteration,
                        'misclassified_point': i,
                        'alpha_old': alpha_old,
                        'b_old': b_old,
                        'alpha_new': self.alpha.copy(),
                        'b_new': self.b
                    })
                    
                    print(f"第 {iteration} 次迭代:")
                    print(f"  误分类点: x{i+1} = {X[i]}, y{i+1} = {y[i]:+d}")
                    print(f"  更新前: α = {alpha_old}, b = {b_old}")
                    print(f"  更新后: α = {self.alpha}, b = {self.b}")
                    print()
                    
                    break
            
            if not has_misclassified:
                print("=" * 60)
                print("训练完成！所有点都被正确分类")
                print("=" * 60)
                break
        
        return self
    
    def get_w(self):
        """
        从对偶形式恢复原始形式的参数 w
        w = Σ(αᵢyᵢxᵢ)
        """
        w = np.sum(self.alpha[:, np.newaxis] * self.y_train[:, np.newaxis] * self.X_train, axis=0)
        return w
    
    def predict(self, X):
        """
        预测
        
        参数:
            X: 待预测数据
            
        返回:
            预测标签 (+1 或 -1)
        """
        # 计算 Σ(αᵢyᵢxᵢ·x) + b
        result = np.zeros(X.shape[0])
        for i in range(len(self.alpha)):
            result += self.alpha[i] * self.y_train[i] * np.dot(X, self.X_train[i])
        result += self.b
        
        return np.sign(result)
    
    def score(self, X, y):
        """
        计算准确率
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


def plot_perceptron(X, y, model, title="感知机对偶形式分类结果"):
    """
    可视化感知机分类结果
    """
    plt.figure(figsize=(10, 8))
    
    # 绘制数据点
    for i in range(len(X)):
        if y[i] == 1:
            plt.scatter(X[i, 0], X[i, 1], c='red', s=200, marker='o', 
                       edgecolors='black', linewidths=2, label='正样本' if i == 0 else '')
            plt.text(X[i, 0] + 0.1, X[i, 1] + 0.1, 
                    f'x{i+1}{tuple(X[i])}\nα={model.alpha[i]:.0f}', fontsize=11)
        else:
            plt.scatter(X[i, 0], X[i, 1], c='blue', s=200, marker='s', 
                       edgecolors='black', linewidths=2, label='负样本' if i == 2 else '')
            plt.text(X[i, 0] + 0.1, X[i, 1] + 0.1, 
                    f'x{i+1}{tuple(X[i])}\nα={model.alpha[i]:.0f}', fontsize=11)
    
    # 从对偶形式恢复 w
    w = model.get_w()
    
    # 绘制分类超平面
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x1_line = np.linspace(x1_min, x1_max, 100)
    
    if w[1] != 0:
        x2_line = -(w[0] * x1_line + model.b) / w[1]
        plt.plot(x1_line, x2_line, 'g-', linewidth=2, 
                label=f'分离超平面: {w[0]:.0f}x₁ + {w[1]:.0f}x₂ + {model.b:.0f} = 0')
    
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
    plt.savefig('perceptron_dual_result.png', dpi=300, bbox_inches='tight')
    print("\n图表已保存至: perceptron_dual_result.png")
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
    
    y = np.array([1, 1, -1])  # 标签
    
    print("训练数据:")
    print("正样本点: x1=(3,3)ᵀ, x2=(4,3)ᵀ")
    print("负样本点: x3=(1,1)ᵀ")
    print()
    
    # 创建并训练感知机
    perceptron = PerceptronDual(learning_rate=1)
    perceptron.fit(X, y)
    
    # 输出最终结果
    w = perceptron.get_w()
    print(f"\n最终模型参数:")
    print(f"α = {perceptron.alpha}")
    print(f"b = {perceptron.b}")
    print(f"\n从对偶形式恢复的原始参数:")
    print(f"w = Σ(αᵢyᵢxᵢ) = {w}")
    print(f"b = {perceptron.b}")
    print(f"\n分离超平面方程: {w[0]:.0f}x₁ + {w[1]:.0f}x₂ + {perceptron.b:.0f} = 0")
    
    # 验证
    print(f"\n训练集准确率: {perceptron.score(X, y) * 100:.1f}%")
    
    # 可视化
    plot_perceptron(X, y, perceptron)


if __name__ == "__main__":
    main()
