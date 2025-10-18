"""
梯度下降法求解一次线性方程
目标函数: y = 0.5x + 1.5 + 高斯噪声
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS系统中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置随机种子，保证结果可复现
np.random.seed(42)

# 生成训练数据
def generate_data(n_samples=10):
    """
    生成训练数据
    真实模型: y = 0.5x + 1.5 + 噪声
    """
    X = np.random.uniform(0, 10, n_samples)  # 在0到10之间均匀采样
    noise = np.random.normal(0, 0.1, n_samples)  # 高斯噪声，均值0，标准差0.5
    y = 0.5 * X + 1.5 + noise
    return X, y

# 梯度下降算法
def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    """
    使用梯度下降法求解线性回归参数
    模型: y = w * x + b
    
    参数:
        X: 输入特征
        y: 目标值
        learning_rate: 学习率
        n_iterations: 迭代次数
    
    返回:
        w: 斜率
        b: 截距
        losses: 每次迭代的损失值
    """
    # 初始化参数
    w = 0.0
    b = 0.0
    n = len(X)
    losses = []
    
    # 梯度下降迭代
    for i in range(n_iterations):
        # 预测值
        y_pred = w * X + b
        
        # 计算损失 (均方误差)
        loss = np.mean((y_pred - y) ** 2)
        losses.append(loss)
        
        # 计算梯度
        dw = (2 / n) * np.sum((y_pred - y) * X)
        db = (2 / n) * np.sum(y_pred - y)
        
        # 更新参数
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        # 每100次迭代打印一次信息
        if (i + 1) % 100 == 0:
            print(f"迭代 {i+1}/{n_iterations}, 损失: {loss:.4f}, w: {w:.4f}, b: {b:.4f}")
    
    return w, b, losses

# 可视化结果
def plot_results(X, y, w, b, losses):
    """
    可视化训练数据、拟合直线和损失曲线
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 子图1: 数据点和拟合直线
    ax1.scatter(X, y, color='blue', label='训练数据', alpha=0.6, s=100)
    
    # 拟合直线
    X_line = np.linspace(X.min(), X.max(), 100)
    y_line = w * X_line + b
    ax1.plot(X_line, y_line, color='red', linewidth=2, label=f'拟合直线: y = {w:.3f}x + {b:.3f}')
    
    # 真实直线
    y_true = 0.5 * X_line + 1.5
    ax1.plot(X_line, y_true, color='green', linewidth=2, linestyle='--', label='真实直线: y = 0.5x + 1.5')
    
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('线性回归拟合结果', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 损失曲线
    ax2.plot(losses, color='purple', linewidth=2)
    ax2.set_xlabel('迭代次数', fontsize=12)
    ax2.set_ylabel('损失 (MSE)', fontsize=12)
    ax2.set_title('损失函数变化曲线', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gradient_descent_result.png', dpi=300, bbox_inches='tight')
    print("\n图表已保存至: gradient_descent_result.png")
    plt.show()

# 主函数
def main():
    print("=" * 60)
    print("梯度下降法求解线性回归")
    print("=" * 60)
    
    # 生成数据
    print("\n1. 生成训练数据...")
    X, y = generate_data(n_samples=10)
    print(f"   生成了 {len(X)} 个样本")
    print(f"   真实模型: y = 0.5x + 1.5 + 高斯噪声")
    
    # 训练模型
    print("\n2. 开始训练...")
    learning_rate = 0.01
    n_iterations = 1000
    print(f"   学习率: {learning_rate}")
    print(f"   迭代次数: {n_iterations}")
    print()
    
    w, b, losses = gradient_descent(X, y, learning_rate, n_iterations)
    
    # 输出结果
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"学到的参数:")
    print(f"  斜率 w = {w:.4f} (真实值: 0.5)")
    print(f"  截距 b = {b:.4f} (真实值: 1.5)")
    print(f"最终损失: {losses[-1]:.4f}")
    
    # 可视化
    print("\n3. 生成可视化图表...")
    plot_results(X, y, w, b, losses)

if __name__ == "__main__":
    main()
