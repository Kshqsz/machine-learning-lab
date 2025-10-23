"""
二项逻辑斯谛回归 - 梯度下降法
参考：《机器学习方法（第2版）》李航

使用梯度下降法求解二项逻辑斯谛回归模型参数
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS系统中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class BinomialLogisticRegression:
    """
    二项逻辑斯谛回归分类器 - 使用梯度下降法
    
    模型: P(Y=1|x) = 1 / (1 + exp(-(w·x + b)))
    损失函数: 负对数似然 L(w,b) = -Σ[y*log(p) + (1-y)*log(1-p)]
    """
    
    def __init__(self, learning_rate=0.1, n_iterations=1000, tol=1e-6):
        """
        初始化逻辑回归模型
        
        参数:
            learning_rate: 学习率
            n_iterations: 最大迭代次数
            tol: 收敛阈值
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tol = tol
        self.w = None
        self.b = None
        self.loss_history = []
        
    def sigmoid(self, z):
        """
        Sigmoid函数（逻辑斯谛函数）
        σ(z) = 1 / (1 + exp(-z))
        """
        # 防止数值溢出
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
    
    def compute_loss(self, X, y):
        """
        计算负对数似然损失函数
        L(w,b) = -Σ[y_i*log(p_i) + (1-y_i)*log(1-p_i)]
        """
        n = len(y)
        z = np.dot(X, self.w) + self.b
        p = self.sigmoid(z)
        
        # 防止log(0)
        epsilon = 1e-10
        p = np.clip(p, epsilon, 1 - epsilon)
        
        loss = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)) / n
        return loss
    
    def fit(self, X, y, verbose=True):
        """
        使用梯度下降法训练模型
        
        梯度公式:
            ∂L/∂w = (1/n) * Σ(p_i - y_i) * x_i
            ∂L/∂b = (1/n) * Σ(p_i - y_i)
        
        参数:
            X: 训练特征 (n_samples, n_features)
            y: 训练标签 (n_samples,)
            verbose: 是否打印训练过程
        """
        n_samples, n_features = X.shape
        
        # 初始化参数
        self.w = np.zeros(n_features)
        self.b = 0.0
        
        if verbose:
            print("="*70)
            print("二项逻辑斯谛回归 - 梯度下降法")
            print("="*70)
            print(f"训练样本数: {n_samples}")
            print(f"特征维度: {n_features}")
            print(f"学习率: {self.learning_rate}")
            print(f"最大迭代次数: {self.n_iterations}")
            print(f"收敛阈值: {self.tol}")
            print("-"*70)
        
        # 梯度下降迭代
        for iteration in range(self.n_iterations):
            # 前向传播：计算预测概率
            z = np.dot(X, self.w) + self.b
            p = self.sigmoid(z)
            
            # 计算损失
            loss = self.compute_loss(X, y)
            self.loss_history.append(loss)
            
            # 计算梯度
            dw = np.dot(X.T, (p - y)) / n_samples
            db = np.sum(p - y) / n_samples
            
            # 更新参数
            w_old = self.w.copy()
            b_old = self.b
            
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            
            # 打印训练过程（每100次迭代）
            if verbose and (iteration + 1) % 100 == 0:
                print(f"迭代 {iteration + 1:4d} | 损失: {loss:.6f} | "
                      f"w: {self.w[0]:.4f} | b: {self.b:.4f}")
            
            # 检查收敛
            if iteration > 0:
                w_diff = np.abs(self.w - w_old).max()
                b_diff = np.abs(self.b - b_old)
                if w_diff < self.tol and b_diff < self.tol:
                    if verbose:
                        print(f"\n在第 {iteration + 1} 次迭代时收敛！")
                    break
        
        if verbose:
            print("-"*70)
            print(f"训练完成！")
            print(f"最终参数: w = {self.w[0]:.6f}, b = {self.b:.6f}")
            print(f"最终损失: {loss:.6f}")
            print("="*70)
    
    def predict_proba(self, X):
        """
        预测概率 P(Y=1|x)
        """
        z = np.dot(X, self.w) + self.b
        return self.sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """
        预测类别（0或1）
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def score(self, X, y):
        """
        计算准确率
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


def plot_results(X, y, model):
    """
    可视化逻辑回归结果
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 子图1: 数据点和决策边界
    ax1 = axes[0]
    
    # 绘制训练数据
    for label in [0, 1]:
        mask = y == label
        label_name = '未通过' if label == 0 else '通过'
        color = 'red' if label == 0 else 'green'
        marker = 'x' if label == 0 else 'o'
        if label == 0:
            # 'x' 标记不支持边缘颜色，只设置颜色
            ax1.scatter(X[mask, 0], y[mask], c=color, marker=marker, 
                       s=100, label=label_name, linewidths=1.5)
        else:
            # 'o' 标记支持边缘颜色
            ax1.scatter(X[mask, 0], y[mask], c=color, marker=marker, 
                       s=100, label=label_name, edgecolors='black', linewidths=1.5)
    
    # 绘制逻辑斯谛曲线
    x_plot = np.linspace(X.min() - 0.5, X.max() + 0.5, 300).reshape(-1, 1)
    y_proba = model.predict_proba(x_plot)
    ax1.plot(x_plot, y_proba, 'b-', linewidth=2, label='P(Y=1|x)')
    
    # 绘制决策边界 (P=0.5)
    ax1.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, 
               label='决策边界 (p=0.5)')
    
    ax1.set_xlabel('学习时长（小时）', fontsize=12, fontfamily='sans-serif')
    ax1.set_ylabel('通过概率 / 实际标签', fontsize=12, fontfamily='sans-serif')
    ax1.set_title('逻辑斯谛回归 - 学生考试通过预测', fontsize=14, 
                 fontweight='bold', fontfamily='sans-serif')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.1)
    
    # 子图2: 损失函数曲线
    ax2 = axes[1]
    ax2.plot(model.loss_history, 'b-', linewidth=2)
    ax2.set_xlabel('迭代次数', fontsize=12, fontfamily='sans-serif')
    ax2.set_ylabel('负对数似然损失', fontsize=12, fontfamily='sans-serif')
    ax2.set_title('训练损失曲线', fontsize=14, fontweight='bold', fontfamily='sans-serif')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片到当前目录
    import os
    save_path = 'binomial_logistic_regression_result.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存至: {os.path.abspath(save_path)}")
    plt.show()


def main():
    """
    主函数 - 学生考试通过预测
    """
    # 训练数据：20位学生的学习时长（小时）与考试通过与否
    data = np.array([
        [0.50, 0], [0.75, 0], [1.00, 0], [1.25, 0], [1.50, 0],
        [1.75, 0], [1.75, 1], [2.00, 0], [2.25, 1], [2.50, 0],
        [2.75, 1], [3.00, 0], [3.25, 1], [3.50, 0], [4.00, 1],
        [4.25, 1], [4.50, 1], [4.75, 1], [5.00, 1], [5.50, 1]
    ])
    
    X = data[:, 0].reshape(-1, 1)  # 学习时长
    y = data[:, 1].astype(int)      # 是否通过 (0/1)
    
    print("\n训练数据集 - 学生考试通过情况:")
    print("-"*70)
    print("序号 | 学习时长(小时) | 考试结果")
    print("-"*70)
    for i, (hours, result) in enumerate(data, 1):
        result_str = "通过" if result == 1 else "未通过"
        print(f"{i:4d} | {hours:14.2f} | {result_str}")
    print("-"*70)
    print(f"通过人数: {np.sum(y == 1)}/{len(y)}")
    print(f"未通过人数: {np.sum(y == 0)}/{len(y)}")
    print()
    
    # 创建并训练模型
    model = BinomialLogisticRegression(
        learning_rate=0.25,
        n_iterations=2000,
        tol=1e-6
    )
    
    model.fit(X, y, verbose=True)
    
    # 模型评估
    print("\n模型评估:")
    print("="*70)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    
    accuracy = model.score(X, y)
    print(f"训练集准确率: {accuracy * 100:.2f}%")
    
    # 详细预测结果
    print("\n详细预测结果:")
    print("-"*70)
    print("序号 | 学习时长 | 真实标签 | 预测概率 | 预测标签 | 是否正确")
    print("-"*70)
    for i in range(len(X)):
        correct = "✓" if y_pred[i] == y[i] else "✗"
        print(f"{i+1:4d} | {X[i,0]:8.2f} | {y[i]:8d} | {y_proba[i]:8.4f} | "
              f"{y_pred[i]:8d} | {correct:^8s}")
    print("-"*70)
    
    # 测试新样本
    print("\n测试新样本预测:")
    print("="*70)
    test_hours = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    test_proba = model.predict_proba(test_hours)
    test_pred = model.predict(test_hours)
    
    for hours, proba, pred in zip(test_hours.flatten(), test_proba, test_pred):
        pred_str = "通过" if pred == 1 else "未通过"
        print(f"学习 {hours:.1f} 小时 -> 通过概率: {proba:.4f} -> 预测: {pred_str}")
    
    # 找到决策边界（P=0.5时的学习时长）
    # 求解: sigmoid(w*x + b) = 0.5 => w*x + b = 0 => x = -b/w
    decision_boundary = -model.b / model.w[0]
    print(f"\n决策边界: 学习时长 = {decision_boundary:.2f} 小时")
    print(f"(学习时长 >= {decision_boundary:.2f} 小时时，预测通过)")
    
    print("\n模型参数解释:")
    print("-"*70)
    print(f"权重 w = {model.w[0]:.6f}  (学习时长的系数)")
    print(f"偏置 b = {model.b:.6f}  (截距项)")
    print(f"\n逻辑斯谛函数: P(Y=1|x) = 1 / (1 + exp(-({model.w[0]:.4f}*x + {model.b:.4f})))")
    print("="*70)
    
    # 可视化结果
    plot_results(X, y, model)


if __name__ == "__main__":
    main()
