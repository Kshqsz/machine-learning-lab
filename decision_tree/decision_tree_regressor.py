"""
决策树 - 回归树 (Regression Tree)
参考：《机器学习方法（第2版）》李航 - 第5章

使用均方误差 (MSE) 作为分裂标准
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS系统中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class DecisionTreeRegressor:
    """
    决策树回归器 - 使用均方误差 (MSE)
    """
    
    def __init__(self, max_depth=None, min_samples_split=2, min_mse_decrease=0.0):
        """
        参数:
            max_depth: 树的最大深度
            min_samples_split: 分裂所需的最小样本数
            min_mse_decrease: MSE减少的最小值
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_mse_decrease = min_mse_decrease
        self.tree = None
        
    def calculate_mse(self, y):
        """
        计算均方误差 (MSE)
        
        MSE = (1/n) * Σ(y_i - ȳ)^2
        """
        if len(y) == 0:
            return 0
        
        mean = np.mean(y)
        mse = np.mean((y - mean) ** 2)
        
        return mse
    
    def calculate_mse_split(self, y_left, y_right):
        """
        计算分裂后的加权均方误差
        
        MSE = (n_left/n) * MSE_left + (n_right/n) * MSE_right
        """
        n = len(y_left) + len(y_right)
        n_left = len(y_left)
        n_right = len(y_right)
        
        if n_left == 0 or n_right == 0:
            return float('inf')
        
        mse_left = self.calculate_mse(y_left)
        mse_right = self.calculate_mse(y_right)
        
        weighted_mse = (n_left / n) * mse_left + (n_right / n) * mse_right
        
        return weighted_mse
    
    def find_best_split(self, X, y):
        """
        找到最优的分裂点
        
        对于一维特征，尝试所有相邻点的中点作为分裂点
        """
        n_samples = len(X)
        
        if n_samples <= self.min_samples_split:
            return None
        
        # 当前MSE
        current_mse = self.calculate_mse(y)
        
        best_mse = float('inf')
        best_split = None
        
        # 对X排序
        sorted_indices = np.argsort(X.flatten())
        X_sorted = X[sorted_indices]
        y_sorted = y[sorted_indices]
        
        # 尝试所有可能的分裂点
        for i in range(1, n_samples):
            # 使用相邻点的中点作为分裂点
            split_point = (X_sorted[i-1] + X_sorted[i]) / 2
            
            y_left = y_sorted[:i]
            y_right = y_sorted[i:]
            
            mse = self.calculate_mse_split(y_left, y_right)
            
            if mse < best_mse:
                best_mse = mse
                best_split = split_point
        
        # 检查MSE是否有足够的减少
        if current_mse - best_mse < self.min_mse_decrease:
            return None
        
        return best_split
    
    def build_tree(self, X, y, depth=0):
        """
        递归构建回归树
        """
        n_samples = len(X)
        
        # 叶节点的值是该区域所有样本的平均值
        leaf_value = np.mean(y)
        
        # 停止条件
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           self.calculate_mse(y) < 1e-7:  # MSE接近0
            return {'leaf': True, 'value': leaf_value, 'n_samples': n_samples}
        
        # 找到最优分裂
        split_point = self.find_best_split(X, y)
        
        if split_point is None:
            return {'leaf': True, 'value': leaf_value, 'n_samples': n_samples}
        
        # 分裂数据
        left_mask = X.flatten() <= split_point
        right_mask = ~left_mask
        
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]
        
        # 计算MSE减少量
        mse_before = self.calculate_mse(y)
        mse_after = self.calculate_mse_split(y_left, y_right)
        mse_decrease = mse_before - mse_after
        
        # 递归构建左右子树
        left_tree = self.build_tree(X_left, y_left, depth + 1)
        right_tree = self.build_tree(X_right, y_right, depth + 1)
        
        return {
            'leaf': False,
            'split_point': split_point,
            'mse_decrease': mse_decrease,
            'left': left_tree,
            'right': right_tree
        }
    
    def fit(self, X, y):
        """
        训练回归树
        """
        print("=" * 70)
        print("决策树回归器 - 均方误差 (MSE)")
        print("=" * 70)
        print(f"\n训练样本数: {len(y)}")
        print(f"特征维度: {X.shape[1] if len(X.shape) > 1 else 1}")
        print(f"目标值范围: [{np.min(y):.2f}, {np.max(y):.2f}]")
        print()
        
        # 确保X是2D数组
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        self.tree = self.build_tree(X, y)
        
        print("\n回归树结构:")
        print("-" * 70)
        self.print_tree(self.tree)
        
        return self
    
    def print_tree(self, node=None, depth=0, prefix="Root"):
        """
        打印回归树结构
        """
        if node is None:
            node = self.tree
        
        indent = "  " * depth
        
        if node['leaf']:
            print(f"{indent}{prefix}: 叶节点 -> 预测值 = {node['value']:.2f} (样本数: {node['n_samples']})")
        else:
            split_val = float(node['split_point'])
            mse_dec = float(node['mse_decrease'])
            print(f"{indent}{prefix}: [X <= {split_val:.2f}] (MSE减少: {mse_dec:.4f})")
            self.print_tree(node['left'], depth + 1, "├─ 左")
            self.print_tree(node['right'], depth + 1, "└─ 右")
    
    def predict_single(self, x, node=None):
        """
        预测单个样本
        """
        if node is None:
            node = self.tree
        
        if node['leaf']:
            return node['value']
        
        if x <= node['split_point']:
            return self.predict_single(x, node['left'])
        else:
            return self.predict_single(x, node['right'])
    
    def predict(self, X):
        """
        预测
        """
        if len(X.shape) == 1:
            return np.array([self.predict_single(x) for x in X])
        else:
            return np.array([self.predict_single(x[0]) for x in X])
    
    def score(self, X, y):
        """
        计算R²分数
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2


def plot_regression_tree(X, y, model, title="回归树拟合结果"):
    """
    可视化回归树的拟合结果
    """
    plt.figure(figsize=(12, 8))
    
    # 绘制原始数据点
    plt.scatter(X, y, color='blue', s=100, alpha=0.6, edgecolors='black', 
                linewidths=1.5, label='训练数据', zorder=3)
    
    # 生成预测曲线
    X_test = np.linspace(X.min() - 0.5, X.max() + 0.5, 300)
    y_pred = model.predict(X_test)
    
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='回归树预测', zorder=2)
    
    # 标注数据点
    for i, (x, y_val) in enumerate(zip(X, y), 1):
        plt.annotate(f'({x:.0f}, {y_val:.2f})', 
                    xy=(x, y_val), 
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=9,
                    alpha=0.7)
    
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('decision_tree_regressor_result.png', dpi=300, bbox_inches='tight')
    print("\n图表已保存至: decision_tree_regressor_result.png")
    plt.show()


def main():
    """
    主函数
    """
    # 训练数据
    X_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y_train = np.array([4.50, 4.75, 4.91, 5.34, 5.80, 7.05, 7.90, 8.23, 8.70, 9.00])
    
    print("训练数据集:")
    print("-" * 70)
    print(" X  |   Y   ")
    print("-" * 70)
    for x, y in zip(X_train, y_train):
        print(f"{x:3.0f} | {y:5.2f}")
    print("-" * 70)
    print()
    
    # 创建并训练模型
    model = DecisionTreeRegressor(max_depth=3, min_samples_split=2, min_mse_decrease=0.01)
    model.fit(X_train, y_train)
    
    # 计算训练误差
    y_pred_train = model.predict(X_train)
    mse_train = np.mean((y_train - y_pred_train) ** 2)
    r2_train = model.score(X_train, y_train)
    
    print("\n" + "=" * 70)
    print("模型评估")
    print("=" * 70)
    print(f"训练集 MSE: {mse_train:.4f}")
    print(f"训练集 R²: {r2_train:.4f}")
    
    # 预测结果对比
    print("\n预测结果对比:")
    print("-" * 70)
    print(" X  | 真实值 | 预测值 | 误差  ")
    print("-" * 70)
    for x, y_true, y_pred in zip(X_train, y_train, y_pred_train):
        error = y_true - y_pred
        print(f"{x:3.0f} | {y_true:6.2f} | {y_pred:6.2f} | {error:6.2f}")
    print("-" * 70)
    
    # 测试新数据
    X_test = np.array([2.5, 5.5, 8.5])
    y_pred_test = model.predict(X_test)
    
    print("\n新数据预测:")
    print("-" * 70)
    for x, y_pred in zip(X_test, y_pred_test):
        print(f"X = {x:.1f} -> 预测 Y = {y_pred:.2f}")
    
    # 可视化
    plot_regression_tree(X_train, y_train, model)
    
    print("\n" + "=" * 70)
    print("回归树构建完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
