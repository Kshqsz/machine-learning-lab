"""
决策树 - 分类树 (Classification Tree)
参考：《机器学习方法（第2版）》李航 - 第5章

使用基尼指数 (Gini Index) 作为特征选择标准
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS系统中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class DecisionTreeClassifier:
    """
    决策树分类器 - 使用基尼指数
    """
    
    def __init__(self, max_depth=None, min_samples_split=2):
        """
        参数:
            max_depth: 树的最大深度
            min_samples_split: 分裂所需的最小样本数
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        
    def gini_index(self, y):
        """
        计算基尼指数
        
        Gini(D) = 1 - Σ(p_k)^2
        """
        if len(y) == 0:
            return 0
        
        counter = Counter(y)
        gini = 1.0
        
        for count in counter.values():
            p = count / len(y)
            gini -= p ** 2
        
        return gini
    
    def calculate_gini_split(self, X, y, feature_idx, threshold):
        """
        计算按特征和阈值分裂后的加权基尼指数
        
        Gini(D, A) = |D1|/|D| * Gini(D1) + |D2|/|D| * Gini(D2)
        """
        # 根据阈值分裂数据
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return float('inf')
        
        y_left = y[left_mask]
        y_right = y[right_mask]
        
        n = len(y)
        n_left = len(y_left)
        n_right = len(y_right)
        
        gini_left = self.gini_index(y_left)
        gini_right = self.gini_index(y_right)
        
        weighted_gini = (n_left / n) * gini_left + (n_right / n) * gini_right
        
        return weighted_gini
    
    def find_best_split(self, X, y):
        """
        找到最优的分裂特征和阈值
        """
        n_samples, n_features = X.shape
        
        if n_samples <= self.min_samples_split:
            return None, None
        
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        
        for feature_idx in range(n_features):
            # 获取该特征的所有唯一值作为候选阈值
            thresholds = np.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                gini = self.calculate_gini_split(X, y, feature_idx, threshold)
                
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def build_tree(self, X, y, depth=0):
        """
        递归构建决策树
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # 停止条件
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_classes == 1 or \
           n_samples < self.min_samples_split:
            leaf_value = Counter(y).most_common(1)[0][0]
            return {'leaf': True, 'value': leaf_value}
        
        # 找到最优分裂
        feature_idx, threshold = self.find_best_split(X, y)
        
        if feature_idx is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return {'leaf': True, 'value': leaf_value}
        
        # 分裂数据
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]
        
        # 递归构建左右子树
        left_tree = self.build_tree(X_left, y_left, depth + 1)
        right_tree = self.build_tree(X_right, y_right, depth + 1)
        
        return {
            'leaf': False,
            'feature': feature_idx,
            'threshold': threshold,
            'left': left_tree,
            'right': right_tree
        }
    
    def fit(self, X, y):
        """
        训练决策树
        """
        print("=" * 70)
        print("决策树分类器 - 基尼指数")
        print("=" * 70)
        print(f"\n训练样本数: {len(y)}")
        print(f"特征数: {X.shape[1]}")
        print(f"类别: {np.unique(y)}")
        print()
        
        self.tree = self.build_tree(X, y)
        self.print_tree(self.tree)
        
        return self
    
    def print_tree(self, node=None, depth=0, prefix="Root"):
        """
        打印决策树结构
        """
        if node is None:
            node = self.tree
        
        indent = "  " * depth
        
        if node['leaf']:
            print(f"{indent}{prefix}: 叶节点 -> 类别 {node['value']}")
        else:
            print(f"{indent}{prefix}: [特征 {node['feature']} <= {node['threshold']:.2f}]")
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
        
        if x[node['feature']] <= node['threshold']:
            return self.predict_single(x, node['left'])
        else:
            return self.predict_single(x, node['right'])
    
    def predict(self, X):
        """
        预测
        """
        return np.array([self.predict_single(x) for x in X])
    
    def score(self, X, y):
        """
        计算准确率
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


def print_tree_with_names(node, X_train_str, depth=0, prefix="Root"):
    """
    打印带有原始特征名称的决策树
    """
    feature_names = ['年龄', '有工作', '有房子', '信贷情况']
    
    indent = "  " * depth
    
    if node['leaf']:
        label = '同意贷款' if node['value'] == 1 else '拒绝贷款'
        print(f"{indent}{prefix}: 【{label}】")
    else:
        feature_name = feature_names[node['feature']]
        threshold = node['threshold']
        
        # 根据特征类型显示阈值
        if node['feature'] == 0:  # 年龄
            threshold_names = {0.5: '青年/中年', 1.5: '中年/老年'}
            threshold_str = threshold_names.get(threshold, f"<= {threshold:.1f}")
        elif node['feature'] in [1, 2]:  # 有工作/有房子
            threshold_str = "否" if threshold == 0.5 else f"<= {threshold:.1f}"
        elif node['feature'] == 3:  # 信贷情况
            threshold_names = {0.5: '一般', 1.5: '好'}
            threshold_str = threshold_names.get(threshold, f"<= {threshold:.1f}")
        else:
            threshold_str = f"<= {threshold:.1f}"
        
        print(f"{indent}{prefix}: [{feature_name} = {threshold_str}?]")
        print_tree_with_names(node['left'], X_train_str, depth + 1, "├─ 是")
        print_tree_with_names(node['right'], X_train_str, depth + 1, "└─ 否")


def visualize_tree(tree, X, y):
    """
    可视化决策树 - 创建树形结构图和数据分布
    """
    fig = plt.figure(figsize=(16, 10))
    
    # 子图1: 树形结构可视化
    ax1 = plt.subplot(2, 2, (1, 2))
    ax1.axis('off')
    
    # 绘制树形结构
    def draw_tree(node, x, y, dx, depth=0, ax=None, parent_x=None, parent_y=None):
        """递归绘制树节点"""
        if ax is None:
            return
        
        # 绘制连线
        if parent_x is not None:
            ax.plot([parent_x, x], [parent_y, y], 'k-', linewidth=1.5, alpha=0.6)
        
        # 节点样式
        if node['leaf']:
            # 叶节点
            label = '同意\n贷款' if node['value'] == 1 else '拒绝\n贷款'
            color = '#90EE90' if node['value'] == 1 else '#FFB6C6'
            bbox = dict(boxstyle='round,pad=0.5', facecolor=color, edgecolor='black', linewidth=2)
            ax.text(x, y, label, ha='center', va='center', fontsize=10, 
                   bbox=bbox, fontweight='bold', fontfamily='sans-serif')
        else:
            # 内部节点
            feature_names = ['年龄', '有工作', '有房子', '信贷']
            feature_name = feature_names[node['feature']]
            threshold = node['threshold']
            
            label = f"{feature_name}\n<= {threshold:.1f}"
            bbox = dict(boxstyle='round,pad=0.5', facecolor='#87CEEB', 
                       edgecolor='black', linewidth=2)
            ax.text(x, y, label, ha='center', va='center', fontsize=9, bbox=bbox, fontfamily='sans-serif')
            
            # 递归绘制左右子树
            new_dx = dx * 0.5
            draw_tree(node['left'], x - dx, y - 1, new_dx, depth + 1, ax, x, y)
            draw_tree(node['right'], x + dx, y - 1, new_dx, depth + 1, ax, x, y)
    
    # 计算树的深度
    def get_depth(node):
        if node['leaf']:
            return 0
        return 1 + max(get_depth(node['left']), get_depth(node['right']))
    
    tree_depth = get_depth(tree)
    draw_tree(tree, 0, tree_depth, 2**(tree_depth-1), ax=ax1)
    
    ax1.set_xlim(-2**(tree_depth), 2**(tree_depth))
    ax1.set_ylim(-1, tree_depth + 1)
    ax1.set_title('决策树结构可视化 - 贷款审批', fontsize=14, fontweight='bold', pad=20, fontfamily='sans-serif')
    
    # 子图2: 类别分布
    ax2 = plt.subplot(2, 2, 3)
    class_counts = Counter(y)
    labels = ['拒绝贷款', '同意贷款']
    counts = [class_counts[0], class_counts[1]]
    colors = ['#FFB6C6', '#90EE90']
    
    bars = ax2.bar(labels, counts, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('样本数量', fontsize=11, fontfamily='sans-serif')
    ax2.set_title('训练集类别分布', fontsize=12, fontweight='bold', fontfamily='sans-serif')
    # 设置x轴标签字体
    for label in ax2.get_xticklabels():
        label.set_fontfamily('sans-serif')
    ax2.grid(True, axis='y', alpha=0.3)
    
    # 在柱子上显示数值
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold', fontfamily='sans-serif')
    
    # 子图3: 特征统计
    ax3 = plt.subplot(2, 2, 4)
    
    # 统计信息
    info_text = "决策树信息:\n" + "="*30 + "\n"
    info_text += f"总样本数: 14\n"
    info_text += f"特征数: 4\n"
    info_text += f"树深度: {tree_depth}\n"
    info_text += f"训练准确率: 100%\n\n"
    info_text += "特征说明:\n" + "-"*30 + "\n"
    info_text += "1. 年龄: 青年/中年/老年\n"
    info_text += "2. 有工作: 是/否\n"
    info_text += "3. 有房子: 是/否\n"
    info_text += "4. 信贷情况: 一般/好/非常好\n\n"
    info_text += "决策规则:\n" + "-"*30 + "\n"
    info_text += "• 有房子 → 同意\n"
    info_text += "• 无房 + 信贷好 → 同意\n"
    info_text += "• 无房 + 信贷一般 → 拒绝\n"
    
    ax3.text(0.1, 0.95, info_text, transform=ax3.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            family='sans-serif')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig('decision_tree_classifier_result.png', dpi=300, bbox_inches='tight')
    print("\n图表已保存至: decision_tree_classifier_result.png")
    plt.show()


def plot_decision_boundary(X, y, model, title="决策树分类结果"):
    """
    可视化决策边界（仅适用于2D数据）
    """
    if X.shape[1] != 2:
        print("跳过可视化：仅支持2维特征")
        return
    
    plt.figure(figsize=(10, 8))
    
    # 创建网格
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # 预测网格点
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    
    # 绘制数据点
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', 
                         edgecolors='black', s=100, linewidths=1.5)
    
    plt.xlabel('特征 1', fontsize=12, fontfamily='sans-serif')
    plt.ylabel('特征 2', fontsize=12, fontfamily='sans-serif')
    plt.title(title, fontsize=14, fontweight='bold', fontfamily='sans-serif')
    cbar = plt.colorbar(scatter)
    cbar.set_label('类别', fontfamily='sans-serif')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('decision_tree_classifier_result.png', dpi=300, bbox_inches='tight')
    print("\n图表已保存至: decision_tree_classifier_result.png")
    plt.show()


def encode_features(X_str):
    """
    将类别特征编码为数值
    """
    # 年龄编码: 青年=0, 中年=1, 老年=2
    age_map = {'青年': 0, '中年': 1, '老年': 2}
    # 是否编码: 否=0, 是=1
    bool_map = {'否': 0, '是': 1}
    # 信贷情况编码: 一般=0, 好=1, 非常好=2
    credit_map = {'一般': 0, '好': 1, '非常好': 2}
    
    X_encoded = []
    for row in X_str:
        encoded_row = [
            age_map[row[0]],      # 年龄
            bool_map[row[1]],     # 有工作
            bool_map[row[2]],     # 有房子
            credit_map[row[3]]    # 信贷情况
        ]
        X_encoded.append(encoded_row)
    
    return np.array(X_encoded)


def decode_feature(feature_idx, value):
    """
    将编码值解码回原始类别名称
    """
    if feature_idx == 0:  # 年龄
        return ['青年', '中年', '老年'][int(value)]
    elif feature_idx == 1 or feature_idx == 2:  # 有工作 或 有房子
        return ['否', '是'][int(value)]
    elif feature_idx == 3:  # 信贷情况
        return ['一般', '好', '非常好'][int(value)]
    return str(value)


def main():
    """
    主函数 - 贷款审批决策树
    """
    # 训练数据 - 贷款审批数据集
    # 特征: 年龄, 有工作, 有房子, 信贷情况
    # 类别: 否=0 (不同意贷款), 是=1 (同意贷款)
    
    X_train_str = [
        ['青年', '否', '否', '一般'],     # 1
        ['青年', '否', '否', '好'],       # 2
        ['青年', '是', '是', '好'],       # 3
        ['青年', '是', '是', '一般'],     # 4
        ['中年', '否', '否', '一般'],     # 5
        ['中年', '否', '否', '一般'],     # 6
        ['中年', '是', '是', '好'],       # 7
        ['中年', '否', '是', '非常好'],   # 8
        ['中年', '否', '是', '非常好'],   # 9
        ['老年', '否', '是', '非常好'],   # 10
        ['老年', '否', '是', '非常好'],   # 11
        ['老年', '否', '是', '好'],       # 12
        ['老年', '是', '否', '非常好'],   # 13
        ['老年', '是', '否', '一般'],     # 14
    ]
    
    y_train = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0])
    
    # 编码特征
    X_train = encode_features(X_train_str)
    
    print("训练数据集 - 贷款审批:")
    print("-" * 80)
    print("序号 | 年龄  | 有工作 | 有房子 | 信贷情况 | 类别(是否同意贷款)")
    print("-" * 80)
    for i, (x_str, y) in enumerate(zip(X_train_str, y_train), 1):
        y_label = '是' if y == 1 else '否'
        print(f"{i:4} | {x_str[0]:4s} | {x_str[1]:6s} | {x_str[2]:6s} | {x_str[3]:8s} | {y_label:4s}")
    print("-" * 80)
    print()
    
    # 创建并训练模型
    model = DecisionTreeClassifier(max_depth=4, min_samples_split=2)
    model.fit(X_train, y_train)
    
    # 评估模型
    train_accuracy = model.score(X_train, y_train)
    print(f"\n训练集准确率: {train_accuracy * 100:.2f}%")
    
    # 显示更详细的树结构（带特征名称）
    print("\n决策树详细结构（原始特征名）:")
    print("=" * 80)
    print_tree_with_names(model.tree, X_train_str)
    
    # 测试预测
    print("\n测试新样本:")
    print("-" * 80)
    
    # 测试样本1: 青年, 否, 否, 一般
    test1_str = [['青年', '否', '否', '一般']]
    test1 = encode_features(test1_str)
    pred1 = model.predict(test1)[0]
    print(f"样本1: 青年, 无工作, 无房, 信贷一般 -> 预测: {'同意' if pred1 == 1 else '拒绝'}")
    
    # 测试样本2: 老年, 是, 是, 好
    test2_str = [['老年', '是', '是', '好']]
    test2 = encode_features(test2_str)
    pred2 = model.predict(test2)[0]
    print(f"样本2: 老年, 有工作, 有房, 信贷好 -> 预测: {'同意' if pred2 == 1 else '拒绝'}")
    
    # 测试样本3: 中年, 否, 是, 非常好
    test3_str = [['中年', '否', '是', '非常好']]
    test3 = encode_features(test3_str)
    pred3 = model.predict(test3)[0]
    print(f"样本3: 中年, 无工作, 有房, 信贷非常好 -> 预测: {'同意' if pred3 == 1 else '拒绝'}")
    
    # 可视化决策树
    visualize_tree(model.tree, X_train, y_train)
    
    print("\n" + "=" * 70)
    print("决策树构建完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
