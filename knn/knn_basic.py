"""
k近邻法 (k-Nearest Neighbors) - k-d树实现
参考：《机器学习方法（第2版）》李航 - 第3章
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS系统中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义k-d树节点
KDNode = namedtuple('KDNode', ['point', 'label', 'left', 'right', 'axis'])


class KDTree:
    """
    k-d树 (k-dimensional tree)
    用于高效的最近邻搜索
    """
    
    def __init__(self):
        self.root = None
        self.k = None  # 数据维度
        
    def build(self, points, depth=0):
        """
        构造平衡k-d树
        
        参数:
            points: 数据点列表，每个点是 (坐标, 标签) 的元组
            depth: 当前深度
            
        返回:
            k-d树的根节点
        """
        if not points:
            return None
        
        # 确定数据维度
        if self.k is None:
            self.k = len(points[0][0])
        
        # 选择切分轴：按深度循环选择坐标轴
        axis = depth % self.k
        
        # 按选定的轴对点进行排序
        points.sort(key=lambda x: x[0][axis])
        
        # 选择中位数点作为切分点
        median_idx = len(points) // 2
        median_point, median_label = points[median_idx]
        
        print(f"深度 {depth}, 切分轴: x{axis+1}, 切分点: {tuple(median_point)}")
        
        # 递归构造左右子树
        left_points = points[:median_idx]
        right_points = points[median_idx + 1:]
        
        return KDNode(
            point=median_point,
            label=median_label,
            left=self.build(left_points, depth + 1),
            right=self.build(right_points, depth + 1),
            axis=axis
        )
    
    def fit(self, X, y=None):
        """
        构建k-d树
        
        参数:
            X: 训练数据，shape (n_samples, n_features)
            y: 标签（可选）
        """
        if y is None:
            y = list(range(len(X)))
        
        points = [(X[i], y[i]) for i in range(len(X))]
        
        print("=" * 60)
        print("构造平衡k-d树")
        print("=" * 60)
        print(f"训练数据: {len(X)} 个样本，维度: {X.shape[1]}")
        print()
        
        self.root = self.build(points)
        
        print("\n" + "=" * 60)
        print("k-d树构造完成")
        print("=" * 60)
        
        return self
    
    def search_nearest(self, query_point, k=1):
        """
        搜索最近邻点
        
        参数:
            query_point: 查询点
            k: 返回最近的k个点
            
        返回:
            最近邻点及其距离
        """
        if self.root is None:
            return None
        
        print(f"\n搜索点 {tuple(query_point)} 的最近邻:")
        print("-" * 60)
        
        # 存储最近邻候选点
        self.nearest = []
        self.visited_nodes = []
        
        self._search(self.root, query_point, k)
        
        # 按距离排序
        self.nearest.sort(key=lambda x: x[1])
        
        return self.nearest[:k]
    
    def _search(self, node, query_point, k, depth=0):
        """
        递归搜索最近邻
        """
        if node is None:
            return
        
        # 记录访问的节点
        self.visited_nodes.append(node.point)
        
        # 计算当前节点与查询点的距离
        dist = np.linalg.norm(np.array(node.point) - np.array(query_point))
        
        print(f"  访问节点: {tuple(node.point)}, 距离: {dist:.4f}")
        
        # 更新最近邻列表
        self.nearest.append((node.point, dist, node.label))
        if len(self.nearest) > k:
            # 保持列表大小，移除最远的点
            self.nearest.sort(key=lambda x: x[1])
            self.nearest.pop()
        
        # 确定搜索方向
        axis = node.axis
        if query_point[axis] < node.point[axis]:
            # 先搜索左子树
            self._search(node.left, query_point, k, depth + 1)
            # 判断是否需要搜索右子树
            if len(self.nearest) < k or abs(query_point[axis] - node.point[axis]) < self.nearest[-1][1]:
                self._search(node.right, query_point, k, depth + 1)
        else:
            # 先搜索右子树
            self._search(node.right, query_point, k, depth + 1)
            # 判断是否需要搜索左子树
            if len(self.nearest) < k or abs(query_point[axis] - node.point[axis]) < self.nearest[-1][1]:
                self._search(node.left, query_point, k, depth + 1)
    
    def print_tree(self, node=None, depth=0, prefix="Root: "):
        """
        打印k-d树结构
        """
        if node is None:
            if depth == 0:
                node = self.root
            else:
                return
        
        print("  " * depth + prefix + f"{tuple(node.point)} (切分轴: x{node.axis+1})")
        
        if node.left:
            self.print_tree(node.left, depth + 1, "L--- ")
        if node.right:
            self.print_tree(node.right, depth + 1, "R--- ")


def plot_kd_tree(X, query_point, nearest_point, visited_nodes, title="k-d树最近邻搜索"):
    """
    可视化k-d树搜索过程
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 左图：所有训练点和查询点
    ax1.scatter(X[:, 0], X[:, 1], c='blue', s=150, marker='o', 
               edgecolors='black', linewidths=2, label='训练点', zorder=3)
    
    # 标注每个训练点
    for i, point in enumerate(X):
        ax1.text(point[0] + 0.2, point[1] + 0.2, f'{tuple(point)}', 
                fontsize=10, fontweight='bold')
    
    # 绘制查询点
    ax1.scatter(query_point[0], query_point[1], c='red', s=200, marker='*', 
               edgecolors='black', linewidths=2, label='查询点', zorder=4)
    ax1.text(query_point[0] + 0.2, query_point[1] + 0.2, f'{tuple(query_point)}', 
            fontsize=10, color='red', fontweight='bold')
    
    # 绘制最近邻点
    ax1.scatter(nearest_point[0], nearest_point[1], c='green', s=200, marker='s', 
               edgecolors='black', linewidths=3, label='最近邻点', zorder=5)
    
    # 绘制连线
    ax1.plot([query_point[0], nearest_point[0]], 
            [query_point[1], nearest_point[1]], 
            'g--', linewidth=2, label='最近邻连线', zorder=2)
    
    ax1.set_xlabel('x₁', fontsize=14)
    ax1.set_ylabel('x₂', fontsize=14)
    ax1.set_title('k-d树数据分布与最近邻', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 右图：搜索过程中访问的节点
    ax2.scatter(X[:, 0], X[:, 1], c='lightgray', s=150, marker='o', 
               edgecolors='black', linewidths=1, label='未访问点', zorder=3, alpha=0.5)
    
    # 标注所有点
    for i, point in enumerate(X):
        ax2.text(point[0] + 0.2, point[1] + 0.2, f'{tuple(point)}', 
                fontsize=10, alpha=0.6)
    
    # 高亮访问过的节点
    if visited_nodes:
        visited_array = np.array(visited_nodes)
        ax2.scatter(visited_array[:, 0], visited_array[:, 1], c='orange', s=150, 
                   marker='o', edgecolors='black', linewidths=2, 
                   label='访问的节点', zorder=4)
        
        # 绘制访问顺序
        for i, point in enumerate(visited_nodes):
            ax2.text(point[0] - 0.3, point[1] - 0.3, f'{i+1}', 
                    fontsize=12, color='red', fontweight='bold',
                    bbox=dict(boxstyle='circle', facecolor='yellow', alpha=0.7))
    
    # 绘制查询点
    ax2.scatter(query_point[0], query_point[1], c='red', s=200, marker='*', 
               edgecolors='black', linewidths=2, label='查询点', zorder=5)
    
    # 绘制最近邻点
    ax2.scatter(nearest_point[0], nearest_point[1], c='green', s=200, marker='s', 
               edgecolors='black', linewidths=3, label='最近邻点', zorder=6)
    
    ax2.set_xlabel('x₁', fontsize=14)
    ax2.set_ylabel('x₂', fontsize=14)
    ax2.set_title('搜索过程（数字表示访问顺序）', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('knn_kdtree_result.png', dpi=300, bbox_inches='tight')
    print("\n图表已保存至: knn_kdtree_result.png")
    plt.show()


def main(query_point=None, k=1):
    """
    主函数
    
    参数:
        query_point: 查询点坐标，格式为 [x, y] 或 (x, y)，默认为 [3, 4.5]
        k: 搜索k个最近邻，默认为1
    """
    # 训练数据
    X = np.array([
        [2, 3],
        [5, 4],
        [9, 6],
        [4, 7],
        [8, 1],
        [7, 2]
    ])
    
    # 如果没有指定查询点，使用默认值
    if query_point is None:
        query_point = np.array([3, 4.5])
    else:
        query_point = np.array(query_point)
    
    print("训练集数据:")
    for i, point in enumerate(X):
        print(f"  点{i+1}: {tuple(point)}")
    print(f"\n查询点: {tuple(query_point)}")
    print(f"搜索 k={k} 个最近邻")
    print()
    
    # 构建k-d树
    kd_tree = KDTree()
    kd_tree.fit(X)
    
    # 打印树结构
    print("\nk-d树结构:")
    print("-" * 60)
    kd_tree.print_tree()
    
    # 搜索最近邻
    nearest = kd_tree.search_nearest(query_point, k=k)
    
    print("\n" + "=" * 60)
    print("搜索结果")
    print("=" * 60)
    
    if nearest:
        print(f"查询点: {tuple(query_point)}")
        print(f"\n找到 {len(nearest)} 个最近邻:")
        for i, (point, dist, label) in enumerate(nearest, 1):
            print(f"\n第 {i} 近邻:")
            print(f"  坐标: {tuple(point)}")
            print(f"  距离: {dist:.4f}")
            print(f"  计算: √[({query_point[0]}-{point[0]})² + ({query_point[1]}-{point[1]})²] = {dist:.4f}")
    
    # 可视化
    if nearest:
        plot_kd_tree(X, query_point, nearest[0][0], kd_tree.visited_nodes)
    
    return nearest


if __name__ == "__main__":
    # 示例1：使用默认查询点 (3, 4.5)
    print("=" * 60)
    print("示例1：搜索点 (3, 4.5) 的最近邻")
    print("=" * 60)
    main()
    
    # 示例2：自定义查询点
    # 取消下面的注释来测试其他查询点
    # print("\n\n")
    # print("=" * 60)
    # print("示例2：搜索点 (7, 5) 的最近邻")
    # print("=" * 60)
    # main(query_point=[7, 5])
    
    # 示例3：搜索k个最近邻
    # print("\n\n")
    # print("=" * 60)
    # print("示例3：搜索点 (5, 5) 的3个最近邻")
    # print("=" * 60)
    # main(query_point=[5, 5], k=3)
