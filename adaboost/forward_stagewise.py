"""
前向分步算法 (Forward Stagewise Algorithm)
============================================

前向分步算法是一种通用的加法模型学习算法，AdaBoost是其特例。

算法思想：
1. 加法模型：f(x) = Σ β_m * b(x; γ_m)
   - b(x; γ_m) 是基函数，γ_m 是基函数参数
   - β_m 是基函数系数

2. 前向分步策略：
   - 从前向后，每一步只学习一个基函数及其系数
   - 逐步逼近优化目标函数
   - 简化了优化的复杂度

3. 损失函数：
   - 可以使用不同的损失函数（平方损失、指数损失等）
   - AdaBoost使用指数损失：L(y, f(x)) = exp(-yf(x))

参考：《统计学习方法》第8章 提升方法
作者：李航
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.family'] = 'Arial Unicode MS'
rcParams['axes.unicode_minus'] = False


class DecisionStump:
    """决策树桩 - 简单的弱分类器"""
    
    def __init__(self):
        self.threshold = None
        self.direction = None  # +1 或 -1
        
    def fit(self, X, y, weights):
        """
        训练决策树桩
        
        参数:
            X: 训练数据 (N,)
            y: 标签 {-1, +1} (N,)
            weights: 样本权重 (N,)
        """
        n_samples = len(X)
        min_error = float('inf')
        
        # 尝试所有可能的阈值
        thresholds = np.unique(X)
        # 添加边界阈值
        if len(thresholds) > 0:
            step = (thresholds[-1] - thresholds[0]) / (len(thresholds) + 1)
            thresholds = np.concatenate([
                [thresholds[0] - step],
                thresholds,
                [thresholds[-1] + step]
            ])
        
        for threshold in thresholds:
            for direction in [1, -1]:
                # 预测
                predictions = np.ones(n_samples)
                predictions[direction * X < direction * threshold] = -1
                
                # 计算加权错误率
                error = np.sum(weights[predictions != y])
                
                if error < min_error:
                    min_error = error
                    self.threshold = threshold
                    self.direction = direction
        
        return min_error
    
    def predict(self, X):
        """预测"""
        predictions = np.ones(len(X))
        predictions[self.direction * X < self.direction * self.threshold] = -1
        return predictions


class ForwardStagewiseClassifier:
    """
    前向分步算法分类器
    
    通用的加法模型学习算法，支持不同的损失函数。
    """
    
    def __init__(self, n_estimators=10, loss='exponential', learning_rate=1.0):
        """
        参数:
            n_estimators: 基学习器数量（迭代次数）
            loss: 损失函数类型
                  'exponential' - 指数损失（等价于AdaBoost）
                  'squared' - 平方损失
            learning_rate: 学习率（步长）
        """
        self.n_estimators = n_estimators
        self.loss = loss
        self.learning_rate = learning_rate
        self.estimators = []  # 基学习器列表
        self.estimator_weights = []  # 基学习器权重
        self.estimator_errors = []  # 每轮错误率
        self.training_errors = []  # 训练误差历史
        
    def _exponential_loss(self, y, f):
        """指数损失函数：L(y, f) = exp(-yf)"""
        return np.exp(-y * f)
    
    def _squared_loss(self, y, f):
        """平方损失函数：L(y, f) = (y - f)^2"""
        return (y - f) ** 2
    
    def _compute_loss(self, y, f):
        """计算损失"""
        if self.loss == 'exponential':
            return self._exponential_loss(y, f)
        elif self.loss == 'squared':
            return self._squared_loss(y, f)
        else:
            raise ValueError(f"Unknown loss function: {self.loss}")
    
    def fit(self, X, y):
        """
        训练前向分步算法
        
        算法流程：
        1. 初始化 f_0(x) = 0
        2. 对 m = 1, 2, ..., M:
           a) 极小化损失函数得到参数：
              (β_m, γ_m) = argmin Σ L(y_i, f_{m-1}(x_i) + β*b(x_i; γ))
           b) 更新模型：
              f_m(x) = f_{m-1}(x) + β_m * b(x; γ_m)
        
        参数:
            X: 训练数据 (N,)
            y: 标签 {-1, +1} (N,)
        """
        X = np.array(X)
        y = np.array(y)
        n_samples = len(X)
        
        # 初始化 f(x) = 0
        f = np.zeros(n_samples)
        
        print("=" * 70)
        print(f"前向分步算法训练 - 损失函数: {self.loss}")
        print("=" * 70)
        print(f"训练样本数: {n_samples}")
        print(f"基学习器数量: {self.n_estimators}")
        print(f"学习率: {self.learning_rate}")
        print("=" * 70)
        
        for m in range(self.n_estimators):
            print(f"\n{'='*70}")
            print(f"第 {m+1} 轮迭代")
            print(f"{'='*70}")
            
            # 计算当前损失
            loss_values = self._compute_loss(y, f)
            
            if self.loss == 'exponential':
                # 指数损失：权重为 w_i = exp(-y_i * f(x_i))
                weights = loss_values
                weights = weights / np.sum(weights)  # 归一化
                
                print(f"样本权重分布 D_{m+1}:")
                print(f"  {weights}")
                
                # 训练弱分类器
                estimator = DecisionStump()
                error = estimator.fit(X, y, weights)
                
                print(f"\n弱分类器 G_{m+1}(x):")
                print(f"  阈值 threshold = {estimator.threshold:.4f}")
                print(f"  方向 direction = {'+1' if estimator.direction > 0 else '-1'}")
                
                # 计算分类器权重 α
                if error > 0 and error < 1:
                    alpha = 0.5 * np.log((1 - error) / error)
                elif error == 0:
                    alpha = 10.0  # 完美分类器
                else:
                    alpha = 0.0  # 随机猜测
                
                alpha *= self.learning_rate
                
                print(f"  加权错误率 e_{m+1} = {error:.4f}")
                print(f"  分类器权重 α_{m+1} = {alpha:.4f}")
                
                # 更新 f(x)
                predictions = estimator.predict(X)
                f = f + alpha * predictions
                
                self.estimators.append(estimator)
                self.estimator_weights.append(alpha)
                self.estimator_errors.append(error)
                
            elif self.loss == 'squared':
                # 平方损失：残差拟合
                residuals = y - f
                
                print(f"当前残差:")
                print(f"  {residuals}")
                
                # 用残差训练弱学习器
                weights = np.ones(n_samples) / n_samples
                estimator = DecisionStump()
                estimator.fit(X, np.sign(residuals), weights)
                
                print(f"\n弱分类器 b_{m+1}(x):")
                print(f"  阈值 threshold = {estimator.threshold:.4f}")
                print(f"  方向 direction = {'+1' if estimator.direction > 0 else '-1'}")
                
                # 计算系数 β（这里简化为固定学习率）
                beta = self.learning_rate
                
                # 更新 f(x)
                predictions = estimator.predict(X)
                f = f + beta * predictions
                
                print(f"  系数 β_{m+1} = {beta:.4f}")
                
                self.estimators.append(estimator)
                self.estimator_weights.append(beta)
                
                # 计算MSE
                mse = np.mean((y - f) ** 2)
                self.estimator_errors.append(mse)
            
            # 计算训练误差（分类错误率）
            final_predictions = np.sign(f)
            final_predictions[final_predictions == 0] = 1  # 处理边界情况
            train_error = np.mean(final_predictions != y)
            self.training_errors.append(train_error)
            
            print(f"\n当前训练误差: {train_error:.4f} ({int(train_error*n_samples)}/{n_samples})")
        
        # 最终结果
        print("\n" + "=" * 70)
        print("训练完成")
        print("=" * 70)
        
        final_predictions = self.predict(X)
        accuracy = np.mean(final_predictions == y)
        
        print(f"\n最终强分类器:")
        if self.loss == 'exponential':
            formula = "f(x) = sign("
            terms = []
            for i, (alpha, estimator) in enumerate(zip(self.estimator_weights, self.estimators)):
                terms.append(f"{alpha:.4f}·G_{i+1}(x)")
            formula += " + ".join(terms) + ")"
            print(f"  {formula}")
        
        print(f"\n训练集准确率: {accuracy*100:.2f}% ({int(accuracy*n_samples)}/{n_samples})")
        
        # 显示预测详情
        print("\n预测详情:")
        print(f"{'样本':<6} {'真实标签':<10} {'预测标签':<10} {'结果':<6}")
        print("-" * 40)
        for i in range(n_samples):
            pred = final_predictions[i]
            true = y[i]
            result = "✓" if pred == true else "✗"
            print(f"{i:<6} {int(true):<10} {int(pred):<10} {result:<6}")
        
        return self
    
    def predict(self, X):
        """
        预测
        
        f(x) = sign(Σ β_m * b(x; γ_m))
        """
        X = np.array(X)
        n_samples = len(X) if hasattr(X, '__len__') else 1
        
        # 初始化预测值
        if n_samples == 1:
            f = 0.0
        else:
            f = np.zeros(n_samples)
        
        # 累加所有基学习器的预测
        for weight, estimator in zip(self.estimator_weights, self.estimators):
            f += weight * estimator.predict(X if hasattr(X, '__len__') else np.array([X]))
        
        # 返回符号函数
        predictions = np.sign(f)
        if n_samples == 1:
            return predictions[0] if predictions[0] != 0 else 1
        else:
            predictions[predictions == 0] = 1  # 处理边界情况
            return predictions
    
    def plot_results(self, X, y, save_path=None):
        """可视化结果"""
        X = np.array(X)
        y = np.array(y)
        
        # 创建图形
        n_estimators = len(self.estimators)
        # 需要的子图：每个弱分类器 + 强分类器 + 训练误差曲线
        total_plots = n_estimators + 2  # +1强分类器 +1误差曲线
        n_cols = min(3, total_plots)
        n_rows = (total_plots + n_cols - 1) // n_cols  # 向上取整
        
        fig = plt.figure(figsize=(5*n_cols, 4*n_rows))
        
        # 绘制每个弱分类器
        for i, (estimator, weight, error) in enumerate(zip(
            self.estimators, self.estimator_weights, self.estimator_errors
        )):
            ax = plt.subplot(n_rows, n_cols, i + 1)
            
            # 绘制数据点
            ax.scatter(X[y == 1], y[y == 1], c='blue', marker='o', 
                      s=100, label='正类 (+1)', edgecolors='k', alpha=0.7)
            ax.scatter(X[y == -1], y[y == -1], c='red', marker='s', 
                      s=100, label='负类 (-1)', edgecolors='k', alpha=0.7)
            
            # 绘制决策边界
            x_min, x_max = X.min() - 1, X.max() + 1
            xx = np.linspace(x_min, x_max, 200)
            yy = estimator.predict(xx)
            
            ax.plot(xx, yy, 'g--', linewidth=2, label='决策边界')
            ax.axvline(x=estimator.threshold, color='purple', 
                      linestyle=':', linewidth=2, label=f'阈值={estimator.threshold:.2f}')
            
            # 填充分类区域
            ax.fill_between(xx, -1.5, 1.5, where=(yy > 0), alpha=0.2, color='blue')
            ax.fill_between(xx, -1.5, 1.5, where=(yy < 0), alpha=0.2, color='red')
            
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(-1.5, 1.5)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            
            if self.loss == 'exponential':
                ax.set_title(f'弱分类器 G_{i+1}(x)\nα={weight:.4f}, e={error:.4f}')
            else:
                ax.set_title(f'弱分类器 b_{i+1}(x)\nβ={weight:.4f}, MSE={error:.4f}')
            
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # 绘制最终强分类器
        ax = plt.subplot(n_rows, n_cols, n_estimators + 1)
        
        ax.scatter(X[y == 1], y[y == 1], c='blue', marker='o', 
                  s=100, label='正类 (+1)', edgecolors='k', alpha=0.7)
        ax.scatter(X[y == -1], y[y == -1], c='red', marker='s', 
                  s=100, label='负类 (-1)', edgecolors='k', alpha=0.7)
        
        # 最终预测
        xx = np.linspace(x_min, x_max, 200)
        yy = self.predict(xx)
        
        ax.plot(xx, yy, 'k-', linewidth=3, label='强分类器 f(x)')
        ax.fill_between(xx, -1.5, 1.5, where=(yy > 0), alpha=0.3, color='blue')
        ax.fill_between(xx, -1.5, 1.5, where=(yy < 0), alpha=0.3, color='red')
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'最终强分类器 f(x)\nM={n_estimators}')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 绘制训练误差曲线
        if len(self.training_errors) > 1:
            ax = plt.subplot(n_rows, n_cols, n_estimators + 2)
            iterations = range(1, len(self.training_errors) + 1)
            ax.plot(iterations, self.training_errors, 'bo-', linewidth=2, markersize=8)
            ax.set_xlabel('迭代次数 m')
            ax.set_ylabel('训练误差')
            ax.set_title('训练误差曲线')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([-0.05, max(self.training_errors) + 0.1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n图像已保存到: {save_path}")
        
        plt.show()


def demo_exponential_loss():
    """演示：指数损失（等价于AdaBoost）"""
    print("\n" + "="*70)
    print("演示1: 前向分步算法 - 指数损失（等价于AdaBoost）")
    print("="*70)
    
    # 训练数据
    X = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])
    
    print(f"\n训练数据:")
    print(f"X = {X}")
    print(f"y = {y}")
    
    # 训练模型
    clf = ForwardStagewiseClassifier(n_estimators=3, loss='exponential')
    clf.fit(X, y)
    
    # 可视化
    clf.plot_results(X, y, save_path='forward_stagewise_exponential.png')


def demo_squared_loss():
    """演示：平方损失"""
    print("\n" + "="*70)
    print("演示2: 前向分步算法 - 平方损失")
    print("="*70)
    
    # 训练数据
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = np.array([1, 1, 1, -1, -1, -1, -1, 1, 1, 1])
    
    print(f"\n训练数据:")
    print(f"X = {X}")
    print(f"y = {y}")
    
    # 训练模型
    clf = ForwardStagewiseClassifier(
        n_estimators=5, 
        loss='squared', 
        learning_rate=0.5
    )
    clf.fit(X, y)
    
    # 可视化
    clf.plot_results(X, y, save_path='forward_stagewise_squared.png')


def demo_comparison():
    """演示：对比不同损失函数"""
    print("\n" + "="*70)
    print("演示3: 对比指数损失 vs 平方损失")
    print("="*70)
    
    # 训练数据
    X = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])
    
    print(f"\n训练数据:")
    print(f"X = {X}")
    print(f"y = {y}")
    
    # 指数损失
    print("\n--- 指数损失 ---")
    clf_exp = ForwardStagewiseClassifier(n_estimators=5, loss='exponential')
    clf_exp.fit(X, y)
    
    # 平方损失
    print("\n--- 平方损失 ---")
    clf_sq = ForwardStagewiseClassifier(n_estimators=5, loss='squared', learning_rate=0.5)
    clf_sq.fit(X, y)
    
    # 对比可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    for ax, clf, title in zip(axes, [clf_exp, clf_sq], 
                               ['指数损失 (Exponential)', '平方损失 (Squared)']):
        # 绘制数据点
        ax.scatter(X[y == 1], y[y == 1], c='blue', marker='o', 
                  s=100, label='正类 (+1)', edgecolors='k', alpha=0.7)
        ax.scatter(X[y == -1], y[y == -1], c='red', marker='s', 
                  s=100, label='负类 (-1)', edgecolors='k', alpha=0.7)
        
        # 最终预测
        x_min, x_max = X.min() - 1, X.max() + 1
        xx = np.linspace(x_min, x_max, 200)
        yy = clf.predict(xx)
        
        ax.plot(xx, yy, 'k-', linewidth=3, label='强分类器 f(x)')
        ax.fill_between(xx, -1.5, 1.5, where=(yy > 0), alpha=0.3, color='blue')
        ax.fill_between(xx, -1.5, 1.5, where=(yy < 0), alpha=0.3, color='red')
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('forward_stagewise_comparison.png', dpi=300, bbox_inches='tight')
    print("\n对比图已保存到: forward_stagewise_comparison.png")
    plt.show()


if __name__ == '__main__':
    # 演示1: 指数损失（等价于AdaBoost）
    demo_exponential_loss()
    
    # 演示2: 平方损失
    demo_squared_loss()
    
    # 演示3: 对比不同损失函数
    demo_comparison()
