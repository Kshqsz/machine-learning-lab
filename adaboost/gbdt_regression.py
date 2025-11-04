"""
GBDT 梯度提升决策树 - 回归 (Gradient Boosting Decision Tree - Regression)
===========================================================================

GBDT是一种集成学习算法，通过梯度提升框架组合多个决策树。
与AdaBoost不同，GBDT使用梯度下降的思想，每次拟合损失函数的负梯度（残差）。

算法原理：
1. 初始化模型：f_0(x) = argmin Σ L(y_i, c)
2. 对m=1,2,...,M:
   - 计算负梯度（残差）：r_{mi} = -∂L(y_i, f(x_i))/∂f(x_i)
   - 拟合残差学习弱学习器 h_m(x)
   - 更新模型：f_m(x) = f_{m-1}(x) + ν·h_m(x)
3. 输出最终模型：f_M(x)

对于平方损失函数：L(y, f(x)) = (y - f(x))^2 / 2
负梯度为：r = y - f(x)（即残差）

参考：《统计学习方法》第8章 提升方法
作者：李航
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.family'] = 'Arial Unicode MS'
rcParams['axes.unicode_minus'] = False


class RegressionTree:
    """回归树 - 用于GBDT的弱学习器"""
    
    def __init__(self, max_depth=1, min_samples_split=2):
        """
        参数:
            max_depth: 树的最大深度（GBDT通常使用浅树，如深度1-3）
            min_samples_split: 分裂所需的最小样本数
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        
    def _mse(self, y):
        """计算均方误差"""
        if len(y) == 0:
            return 0
        return np.var(y) * len(y)
    
    def _find_best_split(self, X, y):
        """寻找最佳分裂点"""
        best_mse = float('inf')
        best_split = None
        
        n_samples = len(X)
        if n_samples < self.min_samples_split:
            return None
        
        # 尝试所有可能的分裂点
        for i in range(n_samples - 1):
            threshold = (X[i] + X[i + 1]) / 2
            
            # 分裂
            left_mask = X <= threshold
            right_mask = X > threshold
            
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue
            
            # 计算分裂后的MSE
            left_mse = self._mse(y[left_mask])
            right_mse = self._mse(y[right_mask])
            total_mse = left_mse + right_mse
            
            if total_mse < best_mse:
                best_mse = total_mse
                best_split = {
                    'threshold': threshold,
                    'left_mask': left_mask,
                    'right_mask': right_mask,
                    'mse': total_mse
                }
        
        return best_split
    
    def _build_tree(self, X, y, depth=0):
        """递归构建回归树"""
        n_samples = len(y)
        
        # 终止条件
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return {'type': 'leaf', 'value': np.mean(y)}
        
        # 寻找最佳分裂
        split = self._find_best_split(X, y)
        
        if split is None:
            return {'type': 'leaf', 'value': np.mean(y)}
        
        # 递归构建左右子树
        left_tree = self._build_tree(X[split['left_mask']], y[split['left_mask']], depth + 1)
        right_tree = self._build_tree(X[split['right_mask']], y[split['right_mask']], depth + 1)
        
        return {
            'type': 'split',
            'threshold': split['threshold'],
            'left': left_tree,
            'right': right_tree
        }
    
    def fit(self, X, y):
        """训练回归树"""
        X = np.array(X).flatten()
        y = np.array(y).flatten()
        
        # 排序
        sorted_indices = np.argsort(X)
        X = X[sorted_indices]
        y = y[sorted_indices]
        
        self.tree = self._build_tree(X, y)
        return self
    
    def _predict_single(self, x, tree):
        """预测单个样本"""
        if tree['type'] == 'leaf':
            return tree['value']
        
        if x <= tree['threshold']:
            return self._predict_single(x, tree['left'])
        else:
            return self._predict_single(x, tree['right'])
    
    def predict(self, X):
        """预测"""
        X = np.array(X).flatten()
        return np.array([self._predict_single(x, self.tree) for x in X])
    
    def get_tree_info(self, tree=None, depth=0):
        """获取树的信息（用于打印）"""
        if tree is None:
            tree = self.tree
        
        indent = "  " * depth
        if tree['type'] == 'leaf':
            return f"{indent}叶节点: 预测值 = {tree['value']:.4f}\n"
        else:
            info = f"{indent}分裂节点: x <= {tree['threshold']:.4f}\n"
            info += f"{indent}├─ 左子树:\n"
            info += self.get_tree_info(tree['left'], depth + 1)
            info += f"{indent}└─ 右子树:\n"
            info += self.get_tree_info(tree['right'], depth + 1)
            return info


class GBDTRegressor:
    """
    GBDT回归器
    
    使用梯度提升框架训练多个回归树
    """
    
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=1, 
                 min_samples_split=2, loss='squared'):
        """
        参数:
            n_estimators: 树的数量（迭代次数）
            learning_rate: 学习率（收缩参数）ν ∈ (0, 1]
            max_depth: 每棵树的最大深度
            min_samples_split: 分裂所需的最小样本数
            loss: 损失函数类型 ('squared' - 平方损失)
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.loss = loss
        
        self.trees = []  # 存储所有树
        self.init_value = None  # 初始值
        self.train_history = []  # 训练历史
        
    def _compute_residuals(self, y, y_pred):
        """计算残差（负梯度）"""
        if self.loss == 'squared':
            # 平方损失的负梯度就是残差
            return y - y_pred
        else:
            raise ValueError(f"Unknown loss function: {self.loss}")
    
    def fit(self, X, y, verbose=True):
        """
        训练GBDT模型
        
        参数:
            X: 训练数据特征 (N,)
            y: 训练数据标签 (N,)
            verbose: 是否打印详细信息
        """
        X = np.array(X).flatten()
        y = np.array(y).flatten()
        n_samples = len(X)
        
        if verbose:
            print("=" * 70)
            print("GBDT 梯度提升决策树 - 回归训练")
            print("=" * 70)
            print(f"训练样本数: {n_samples}")
            print(f"树的数量: {self.n_estimators}")
            print(f"学习率: {self.learning_rate}")
            print(f"树的最大深度: {self.max_depth}")
            print(f"损失函数: {self.loss}")
            print("=" * 70)
        
        # 1. 初始化 f_0(x) = argmin Σ L(y_i, c)
        # 对于平方损失，最优常数是均值
        self.init_value = np.mean(y)
        f = np.full(n_samples, self.init_value)
        
        if verbose:
            print(f"\n初始化模型:")
            print(f"  f_0(x) = {self.init_value:.4f} (所有样本的均值)")
            initial_mse = np.mean((y - f) ** 2)
            print(f"  初始MSE: {initial_mse:.4f}")
        
        # 2. 迭代训练
        for m in range(self.n_estimators):
            if verbose:
                print(f"\n{'='*70}")
                print(f"第 {m+1} 轮迭代")
                print(f"{'='*70}")
            
            # 计算残差（负梯度）
            residuals = self._compute_residuals(y, f)
            
            if verbose:
                print(f"\n残差统计:")
                print(f"  残差范围: [{np.min(residuals):.4f}, {np.max(residuals):.4f}]")
                print(f"  残差均值: {np.mean(residuals):.4f}")
                print(f"  残差标准差: {np.std(residuals):.4f}")
            
            # 拟合回归树到残差
            tree = RegressionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X, residuals)
            
            if verbose:
                print(f"\n回归树 T_{m+1}(x) 结构:")
                print(tree.get_tree_info())
            
            # 预测
            tree_predictions = tree.predict(X)
            
            # 更新模型 f_m(x) = f_{m-1}(x) + ν * h_m(x)
            f = f + self.learning_rate * tree_predictions
            
            # 保存树
            self.trees.append(tree)
            
            # 计算当前MSE
            mse = np.mean((y - f) ** 2)
            self.train_history.append(mse)
            
            if verbose:
                print(f"\n更新后的预测:")
                for i in range(n_samples):
                    print(f"  样本 {i+1}: x={X[i]:.0f}, "
                          f"真实值={y[i]:.2f}, "
                          f"预测值={f[i]:.4f}, "
                          f"残差={residuals[i]:.4f}, "
                          f"树预测={tree_predictions[i]:.4f}")
                
                print(f"\n当前模型性能:")
                print(f"  MSE: {mse:.4f}")
                print(f"  RMSE: {np.sqrt(mse):.4f}")
        
        # 最终结果
        if verbose:
            print("\n" + "=" * 70)
            print("训练完成")
            print("=" * 70)
            
            final_predictions = self.predict(X)
            final_mse = np.mean((y - final_predictions) ** 2)
            final_rmse = np.sqrt(final_mse)
            
            print(f"\n最终模型:")
            print(f"  f(x) = {self.init_value:.4f}", end="")
            for m in range(len(self.trees)):
                print(f" + {self.learning_rate}·T_{m+1}(x)", end="")
            print()
            
            print(f"\n最终性能:")
            print(f"  训练集 MSE: {final_mse:.4f}")
            print(f"  训练集 RMSE: {final_rmse:.4f}")
            
            print(f"\n最终预测结果:")
            print(f"{'样本':<6} {'X':<8} {'真实Y':<10} {'预测Y':<10} {'误差':<10}")
            print("-" * 50)
            for i in range(n_samples):
                error = y[i] - final_predictions[i]
                print(f"{i+1:<6} {X[i]:<8.0f} {y[i]:<10.2f} {final_predictions[i]:<10.4f} {error:<10.4f}")
        
        return self
    
    def predict(self, X):
        """
        预测
        
        f(x) = f_0 + ν·Σ T_m(x)
        """
        X = np.array(X).flatten()
        
        # 初始预测
        predictions = np.full(len(X), self.init_value)
        
        # 累加所有树的预测
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        
        return predictions
    
    def plot_results(self, X_train, y_train, save_path=None):
        """可视化结果"""
        X_train = np.array(X_train).flatten()
        y_train = np.array(y_train).flatten()
        
        # 创建图形
        n_trees = len(self.trees)
        n_cols = min(3, n_trees + 2)
        n_rows = (n_trees + 3) // n_cols
        
        fig = plt.figure(figsize=(5*n_cols, 4*n_rows))
        
        # 生成平滑的预测曲线
        x_min, x_max = X_train.min() - 0.5, X_train.max() + 0.5
        X_plot = np.linspace(x_min, x_max, 300)
        
        # 初始预测
        f = np.full(len(X_train), self.init_value)
        f_plot = np.full(len(X_plot), self.init_value)
        
        # 绘制每棵树的贡献
        for i, tree in enumerate(self.trees):
            ax = plt.subplot(n_rows, n_cols, i + 1)
            
            # 绘制训练数据
            ax.scatter(X_train, y_train, c='black', s=100, 
                      label='训练数据', zorder=3, edgecolors='white', linewidths=1.5)
            
            # 当前累积预测
            tree_pred_train = tree.predict(X_train)
            tree_pred_plot = tree.predict(X_plot)
            
            f = f + self.learning_rate * tree_pred_train
            f_plot = f_plot + self.learning_rate * tree_pred_plot
            
            # 绘制当前累积预测曲线
            ax.plot(X_plot, f_plot, 'b-', linewidth=2, 
                   label=f'累积预测 f_{i+1}(x)', zorder=2)
            
            # 绘制当前树的预测（缩放后）
            ax.plot(X_plot, self.init_value + self.learning_rate * tree_pred_plot, 
                   'g--', linewidth=1.5, alpha=0.7, 
                   label=f'当前树 ν·T_{i+1}(x)', zorder=1)
            
            # 计算MSE
            mse = self.train_history[i]
            
            ax.set_xlabel('x', fontsize=10)
            ax.set_ylabel('y', fontsize=10)
            ax.set_title(f'第{i+1}轮: MSE={mse:.4f}', fontsize=11, fontweight='bold')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(x_min, x_max)
        
        # 绘制最终模型
        ax = plt.subplot(n_rows, n_cols, n_trees + 1)
        
        final_pred_train = self.predict(X_train)
        final_pred_plot = self.predict(X_plot)
        
        ax.scatter(X_train, y_train, c='black', s=100, 
                  label='训练数据', zorder=3, edgecolors='white', linewidths=1.5)
        ax.plot(X_plot, final_pred_plot, 'r-', linewidth=3, 
               label='最终模型 f(x)', zorder=2)
        
        final_mse = np.mean((y_train - final_pred_train) ** 2)
        final_rmse = np.sqrt(final_mse)
        
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('y', fontsize=10)
        ax.set_title(f'最终模型\nMSE={final_mse:.4f}, RMSE={final_rmse:.4f}', 
                    fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_min, x_max)
        
        # 绘制MSE变化曲线
        ax = plt.subplot(n_rows, n_cols, n_trees + 2)
        
        iterations = range(1, len(self.train_history) + 1)
        ax.plot(iterations, self.train_history, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('迭代次数 m', fontsize=10)
        ax.set_ylabel('MSE', fontsize=10)
        ax.set_title('训练误差曲线', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, max(self.train_history) * 1.1])
        
        # 添加最终MSE标注
        ax.axhline(y=final_mse, color='r', linestyle='--', alpha=0.5, 
                  label=f'最终MSE={final_mse:.4f}')
        ax.legend(loc='best', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n图像已保存到: {save_path}")
        
        plt.show()


def main():
    """主函数 - 演示GBDT回归"""
    
    print("\n" + "="*70)
    print("GBDT 梯度提升决策树 - 回归示例")
    print("="*70)
    
    # 训练数据
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = np.array([5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05])
    
    print(f"\n训练数据:")
    print(f"X = {X}")
    print(f"y = {y}")
    print(f"\n数据特点:")
    print(f"  样本数: {len(X)}")
    print(f"  X范围: [{X.min()}, {X.max()}]")
    print(f"  y范围: [{y.min():.2f}, {y.max():.2f}]")
    print(f"  y均值: {y.mean():.4f}")
    print(f"  y标准差: {y.std():.4f}")
    
    # 训练GBDT模型
    print("\n" + "="*70)
    print("开始训练GBDT模型")
    print("="*70)
    
    gbdt = GBDTRegressor(
        n_estimators=6,      # 使用6棵树
        learning_rate=0.1,   # 学习率0.1
        max_depth=2,         # 树深度2
        min_samples_split=2
    )
    
    gbdt.fit(X, y, verbose=True)
    
    # 可视化结果
    gbdt.plot_results(X, y, save_path='gbdt_regression.png')
    
    # 测试预测
    print("\n" + "="*70)
    print("预测新样本")
    print("="*70)
    
    X_test = np.array([1.5, 5.5, 10.5])
    y_pred = gbdt.predict(X_test)
    
    print(f"\n测试样本预测:")
    for x, y in zip(X_test, y_pred):
        print(f"  x = {x:.1f} → 预测 y = {y:.4f}")


if __name__ == '__main__':
    main()
