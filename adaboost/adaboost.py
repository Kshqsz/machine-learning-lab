"""
AdaBoost算法 (Adaptive Boosting)
提升方法 - 自适应提升算法

算法原理:
1. 通过改变训练样本的权重，学习多个弱分类器
2. 将这些弱分类器进行线性组合，构成强分类器
3. 加大分类误差率小的弱分类器的权重
4. 减小分类误差率大的弱分类器的权重

基本弱分类器:
- 使用决策树桩（decision stump）
- 单层决策树，只使用一个特征进行分类
- 阈值分类器：v=1 if x>threshold else v=-1

算法流程:
1. 初始化样本权重分布
2. 对每轮迭代：
   - 使用当前权重分布训练弱分类器
   - 计算弱分类器的误差率
   - 计算弱分类器的权重
   - 更新样本权重分布
3. 组合所有弱分类器

特点:
- 不改变训练数据，改变样本权重
- 提升准确率，降低偏差
- 对噪声敏感，容易过拟合
- 训练误差以指数速率下降

适用场景:
- 二分类问题
- 提高弱分类器性能
- 特征选择
- 集成学习
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS
plt.rcParams['axes.unicode_minus'] = False


class DecisionStump:
    """决策树桩 - 单层决策树"""
    
    def __init__(self):
        self.threshold = None  # 分类阈值
        self.direction = None  # 分类方向：1表示x>threshold时预测为1，-1相反
        self.feature_index = 0  # 特征索引（本例只有一个特征）
        
    def predict(self, X):
        """预测"""
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)
        
        if self.direction == 1:
            # x <= threshold 预测为-1
            predictions[X[:, self.feature_index] <= self.threshold] = -1
        else:
            # x > threshold 预测为-1
            predictions[X[:, self.feature_index] > self.threshold] = 1
        
        return predictions


class AdaBoost:
    """AdaBoost算法"""
    
    def __init__(self, n_estimators=3):
        """
        参数:
            n_estimators: 弱分类器数量
        """
        self.n_estimators = n_estimators
        self.estimators = []  # 存储弱分类器
        self.estimator_weights = []  # 存储弱分类器权重
        self.estimator_errors = []  # 存储弱分类器误差
        
    def fit(self, X, y, verbose=True):
        """
        训练AdaBoost分类器
        
        参数:
            X: 训练样本 (n_samples, n_features)
            y: 标签 (n_samples,) 取值为+1或-1
        """
        n_samples, n_features = X.shape
        
        if verbose:
            print("="*70)
            print("AdaBoost算法训练")
            print("="*70)
            print(f"训练样本数: {n_samples}")
            print(f"特征维度: {n_features}")
            print(f"弱分类器数量: {self.n_estimators}")
            print()
            
            print("训练数据:")
            print("-"*70)
            for i in range(n_samples):
                print(f"  样本 {i}: x = {X[i, 0]:.1f}  →  y = {y[i]:+d}")
            print()
        
        # 初始化样本权重 D_1(i) = 1/N
        weights = np.ones(n_samples) / n_samples
        
        if verbose:
            print("初始权重分布:")
            print(f"  D_1 = {weights}")
            print()
        
        # 训练M个弱分类器
        for m in range(self.n_estimators):
            if verbose:
                print(f"{'='*70}")
                print(f"第 {m+1} 轮迭代")
                print(f"{'='*70}")
                print(f"当前权重分布 D_{m+1}:")
                for i in range(n_samples):
                    print(f"  w_{i} = {weights[i]:.4f}")
                print()
            
            # 训练弱分类器 - 找到最佳阈值
            best_stump = None
            min_error = float('inf')
            best_predictions = None
            
            # 尝试所有可能的阈值
            feature_values = X[:, 0]
            # 候选阈值：样本值之间的中点
            thresholds = []
            sorted_values = np.sort(np.unique(feature_values))
            for i in range(len(sorted_values) - 1):
                thresholds.append((sorted_values[i] + sorted_values[i+1]) / 2)
            
            # 也尝试边界值
            thresholds = [-0.5] + thresholds + [sorted_values[-1] + 0.5]
            
            if verbose:
                print(f"尝试的阈值: {thresholds}")
                print()
            
            # 对每个阈值和方向进行尝试
            for threshold in thresholds:
                for direction in [1, -1]:
                    stump = DecisionStump()
                    stump.threshold = threshold
                    stump.direction = direction
                    
                    predictions = stump.predict(X)
                    
                    # 计算加权误差
                    misclassified = (predictions != y)
                    error = np.sum(weights[misclassified])
                    
                    if error < min_error:
                        min_error = error
                        best_stump = stump
                        best_predictions = predictions
            
            if verbose:
                print(f"选择的弱分类器 G_{m+1}(x):")
                print(f"  阈值: {best_stump.threshold:.1f}")
                print(f"  方向: {'x > threshold → +1' if best_stump.direction == 1 else 'x <= threshold → +1'}")
                print()
                
                print(f"分类结果:")
                for i in range(n_samples):
                    pred = int(best_predictions[i])
                    true = int(y[i])
                    status = "✓" if pred == true else "✗"
                    print(f"  样本 {i}: 预测 {pred:+d}, 真实 {true:+d}  {status}")
                print()
            
            # 计算误差率 e_m
            error_rate = min_error
            self.estimator_errors.append(error_rate)
            
            if verbose:
                print(f"误差率 e_{m+1} = {error_rate:.4f}")
            
            # 计算弱分类器权重 α_m
            # α_m = 0.5 * ln((1 - e_m) / e_m)
            if error_rate == 0:
                alpha = 10  # 避免除以0，给一个大权重
            elif error_rate >= 0.5:
                alpha = 0  # 误差率过大，权重为0
            else:
                alpha = 0.5 * np.log((1 - error_rate) / error_rate)
            
            self.estimator_weights.append(alpha)
            
            if verbose:
                print(f"分类器权重 α_{m+1} = 0.5 * ln((1 - {error_rate:.4f}) / {error_rate:.4f}) = {alpha:.4f}")
                print()
            
            # 保存弱分类器
            self.estimators.append(best_stump)
            
            # 更新样本权重
            # w_{m+1,i} = w_{m,i} * exp(-α_m * y_i * G_m(x_i)) / Z_m
            weights = weights * np.exp(-alpha * y * best_predictions)
            
            # 归一化
            Z_m = np.sum(weights)
            weights = weights / Z_m
            
            if verbose:
                print(f"更新权重:")
                print(f"  归一化因子 Z_{m+1} = {Z_m:.4f}")
                print(f"  新权重分布 D_{m+2}:")
                for i in range(n_samples):
                    print(f"    w_{i} = {weights[i]:.4f}")
                print()
        
        if verbose:
            print("="*70)
            print("训练完成！")
            print("="*70)
            print()
            
            print("最终的强分类器:")
            print("-"*70)
            print("f(x) = sign(", end="")
            for m in range(self.n_estimators):
                stump = self.estimators[m]
                alpha = self.estimator_weights[m]
                if m > 0:
                    print(" + ", end="")
                print(f"{alpha:.4f}·G_{m+1}(x)", end="")
            print(")")
            print()
            
            print("各弱分类器:")
            for m in range(self.n_estimators):
                stump = self.estimators[m]
                alpha = self.estimator_weights[m]
                error = self.estimator_errors[m]
                direction_str = "x > " if stump.direction == 1 else "x <= "
                print(f"  G_{m+1}(x): {direction_str}{stump.threshold:.1f} → +1, 否则 -1")
                print(f"    α_{m+1} = {alpha:.4f}, e_{m+1} = {error:.4f}")
            print("="*70)
    
    def predict(self, X):
        """预测"""
        # 计算加权投票
        predictions = np.zeros(X.shape[0])
        
        for alpha, estimator in zip(self.estimator_weights, self.estimators):
            predictions += alpha * estimator.predict(X)
        
        return np.sign(predictions)
    
    def predict_scores(self, X):
        """返回决策函数的值（用于可视化）"""
        scores = np.zeros(X.shape[0])
        
        for alpha, estimator in zip(self.estimator_weights, self.estimators):
            scores += alpha * estimator.predict(X)
        
        return scores
    
    def plot_results(self, X, y):
        """可视化AdaBoost结果"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 图1: 弱分类器的决策边界
        ax1 = axes[0, 0]
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        xx = np.linspace(x_min, x_max, 1000).reshape(-1, 1)
        
        for i, (stump, alpha, error) in enumerate(zip(self.estimators, 
                                                       self.estimator_weights, 
                                                       self.estimator_errors)):
            yy = stump.predict(xx)
            label = f'G_{i+1}(x): 阈值={stump.threshold:.1f}, α={alpha:.3f}, e={error:.3f}'
            ax1.plot(xx, yy + i*0.1, label=label, linewidth=2, alpha=0.7)
        
        # 绘制训练样本
        for label_val in [1, -1]:
            mask = y == label_val
            ax1.scatter(X[mask, 0], np.zeros(np.sum(mask)), 
                       c='red' if label_val == 1 else 'blue',
                       marker='o' if label_val == 1 else 's',
                       s=150, edgecolors='black', linewidths=2,
                       label=f'真实类别 {label_val:+d}', zorder=5)
        
        ax1.set_xlabel('x', fontsize=12)
        ax1.set_ylabel('预测值', fontsize=12)
        ax1.set_title('各弱分类器的决策函数', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
        
        # 图2: 强分类器的决策函数
        ax2 = axes[0, 1]
        scores = self.predict_scores(xx)
        predictions = np.sign(scores)
        
        ax2.plot(xx, scores, 'g-', linewidth=3, label='f(x) = Σ α_m·G_m(x)')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=2, label='决策边界')
        ax2.fill_between(xx.ravel(), -10, scores.ravel(), 
                         where=(scores.ravel() > 0), alpha=0.3, color='red', label='预测为+1区域')
        ax2.fill_between(xx.ravel(), -10, scores.ravel(), 
                         where=(scores.ravel() < 0), alpha=0.3, color='blue', label='预测为-1区域')
        
        # 绘制训练样本
        for label_val in [1, -1]:
            mask = y == label_val
            ax2.scatter(X[mask, 0], np.zeros(np.sum(mask)), 
                       c='red' if label_val == 1 else 'blue',
                       marker='o' if label_val == 1 else 's',
                       s=150, edgecolors='black', linewidths=2,
                       label=f'真实类别 {label_val:+d}', zorder=5)
        
        ax2.set_xlabel('x', fontsize=12)
        ax2.set_ylabel('f(x)', fontsize=12)
        ax2.set_title('强分类器的决策函数', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-3, 3)
        
        # 图3: 分类器权重
        ax3 = axes[1, 0]
        x_pos = np.arange(len(self.estimator_weights))
        bars = ax3.bar(x_pos, self.estimator_weights, color='steelblue', 
                      edgecolor='black', linewidth=1.5, alpha=0.8)
        
        # 在柱子上标注数值
        for i, (bar, weight) in enumerate(zip(bars, self.estimator_weights)):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{weight:.4f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax3.set_xlabel('弱分类器', fontsize=12)
        ax3.set_ylabel('权重 α', fontsize=12)
        ax3.set_title('各弱分类器的权重', fontsize=14, fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([f'G_{i+1}' for i in range(len(self.estimator_weights))])
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 图4: 误差率
        ax4 = axes[1, 1]
        x_pos = np.arange(len(self.estimator_errors))
        bars = ax4.bar(x_pos, self.estimator_errors, color='coral', 
                      edgecolor='black', linewidth=1.5, alpha=0.8)
        
        # 在柱子上标注数值
        for i, (bar, error) in enumerate(zip(bars, self.estimator_errors)):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{error:.4f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax4.set_xlabel('弱分类器', fontsize=12)
        ax4.set_ylabel('误差率 e', fontsize=12)
        ax4.set_title('各弱分类器的误差率', fontsize=14, fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([f'G_{i+1}' for i in range(len(self.estimator_errors))])
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_ylim(0, max(self.estimator_errors) * 1.2)
        
        plt.tight_layout()
        
        # 保存图片
        filename = f'adaboost_M{self.n_estimators}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n图像已保存至: {filename}")
        plt.show()


def main():
    """主函数"""
    print("\n" + "🎯 "*20)
    print("AdaBoost算法演示")
    print("🎯 "*20 + "\n")
    
    # 训练数据（李航《统计学习方法》例8.1）
    X_train = np.array([
        [0],
        [1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [7],
        [8],
        [9]
    ])
    
    y_train = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])
    
    # 创建并训练AdaBoost
    adaboost = AdaBoost(n_estimators=3)
    adaboost.fit(X_train, y_train)
    
    # 在训练集上评估
    print("\n训练集预测结果:")
    print("-"*70)
    y_pred = adaboost.predict(X_train)
    scores = adaboost.predict_scores(X_train)
    
    for i, (x, y_true, y_p, score) in enumerate(zip(X_train, y_train, y_pred, scores)):
        result = "✓" if y_p == y_true else "✗"
        print(f"  样本 {i}: x = {x[0]:.1f}  →  f(x) = {score:+7.4f}  →  预测: {int(y_p):+d}  真实: {y_true:+d}  {result}")
    
    # 计算准确率
    accuracy = np.mean(y_pred == y_train) * 100
    print(f"\n训练集准确率: {accuracy:.2f}%")
    print()
    
    # 测试新样本
    print("新样本预测:")
    print("-"*70)
    X_test = np.array([[0.5], [2.5], [5.5], [7.5]])
    
    for x in X_test:
        y_pred = int(adaboost.predict(x.reshape(1, -1))[0])
        score = adaboost.predict_scores(x.reshape(1, -1))[0]
        print(f"  x = {x[0]:.1f}  →  f(x) = {score:+7.4f}  →  预测: {y_pred:+d}")
    print()
    
    # 可视化
    adaboost.plot_results(X_train, y_train)
    
    print("\n" + "="*70)
    print("演示完成！")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
