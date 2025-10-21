# 机器学习实验 (Machine Learning Lab)

个人机器学习算法学习与实践项目

> 📖 **学习资料**：本项目基于李航老师的《机器学习方法（第2版）》进行学习，通过代码实现加深对算法原理的理解。

## 📚 项目简介

这是我的机器学习学习笔记和实验代码仓库。跟随《机器学习方法（第2版）》的章节顺序，从零开始实现各种经典的机器学习算法，将理论与实践相结合。

## 🎯 学习目标

- 理解机器学习算法的数学原理
- 从底层实现常见的机器学习算法
- 掌握 NumPy、Pandas、Matplotlib 等数据科学工具
- 培养算法调试和优化能力

## 📂 项目结构

```
machine-learning-lab/
├── linear_regression/          # 线性回归
│   └── gradient_descent.py    # 梯度下降法实现
├── perceptron/                 # 感知机
│   ├── perceptron_primal.py   # 感知机原始形式
│   └── perceptron_dual.py     # 感知机对偶形式
├── knn/                        # k近邻法
│   └── knn_basic.py           # k-d树实现
├── naive_bayes/                # 朴素贝叶斯
│   ├── naive_bayes_mle.py     # 极大似然估计
│   └── naive_bayes_est.py     # 贝叶斯估计(拉普拉斯平滑)
├── venv/                       # Python虚拟环境
├── .gitignore                  # Git忽略文件
└── README.md                   # 项目说明
```

## 🔬 已实现的算法

### 1. 线性回归 (Linear Regression)

#### 梯度下降法 (Gradient Descent)
- **文件**: `linear_regression/gradient_descent.py`
- **功能**: 使用梯度下降算法从头实现线性回归
- **特点**:
  - 生成带高斯噪声的训练数据
  - 实现完整的梯度下降优化过程
  - 可视化拟合结果和损失函数变化
  - 对比学习参数与真实参数

**运行示例**:
```bash
python linear_regression/gradient_descent.py
```

**效果展示**:
- 训练数据: 10个样本，符合 y = 0.5x + 1.5 + 噪声
- 学习率: 0.01
- 迭代次数: 1000次
- 输出: 拟合直线对比图 + 损失曲线图

---

### 2. 感知机 (Perceptron)

感知机是二分类的线性分类模型，是神经网络和支持向量机的基础。

#### 感知机原始形式 (Primal Form)
- **文件**: `perceptron/perceptron_primal.py`
- **模型**: $f(x) = \text{sign}(w \cdot x + b)$
- **算法**: 随机梯度下降
- **特点**:
  - 直接更新权重向量 $w$ 和偏置 $b$
  - 详细输出每次迭代的更新过程
  - 可视化分类结果和分离超平面
  - 适合特征维度较低的情况

**运行示例**:
```bash
python perceptron/perceptron_primal.py
```

#### 感知机对偶形式 (Dual Form)
- **文件**: `perceptron/perceptron_dual.py`
- **模型**: $f(x) = \text{sign}(\sum_{i=1}^{N} \alpha_i y_i x_i \cdot x + b)$
- **算法**: 基于样本更新次数的对偶表示
- **特点**:
  - 通过 $\alpha$ 向量记录每个样本的更新次数
  - 预先计算 Gram 矩阵 $G_{ij} = x_i \cdot x_j$ 提高效率
  - 可以恢复原始形式的参数
  - 适合样本数量较少的情况

**运行示例**:
```bash
python perceptron/perceptron_dual.py
```

**训练数据**:
- 正样本: x₁=(3,3)ᵀ, x₂=(4,3)ᵀ
- 负样本: x₃=(1,1)ᵀ
- 学习率: η = 1

---

### 3. k近邻法 (k-Nearest Neighbors)

k近邻法是一种基本的分类与回归方法，通过找到与待分类点最近的k个训练样本来进行预测。

#### k-d树实现 (k-d Tree)
- **文件**: `knn/knn_basic.py`
- **数据结构**: k-d树（k-dimensional tree）
- **功能**: 构造平衡k-d树，高效搜索最近邻
- **特点**:
  - 实现平衡k-d树的构造算法
  - 支持最近邻搜索和k近邻搜索
  - 详细输出树的结构和搜索过程
  - 可视化训练数据、查询点和搜索路径
  - 支持自定义查询点和k值

**运行示例**:
```bash
python knn/knn_basic.py
```

**使用方法**:
```python
# 使用默认查询点 (3, 4.5)
main()

# 自定义查询点
main(query_point=[7, 5])

# 搜索k个最近邻
main(query_point=[5, 5], k=3)
```

**训练数据**:
- 6个样本点: (2,3), (5,4), (9,6), (4,7), (8,1), (7,2)
- 默认查询点: (3, 4.5)
- 构造平衡k-d树，深度优先搜索最近邻

---

### 4. 朴素贝叶斯 (Naive Bayes)

朴素贝叶斯是基于贝叶斯定理与特征条件独立假设的分类方法，简单高效且适用于多分类问题。

#### 极大似然估计 (MLE)
- **文件**: `naive_bayes/naive_bayes_mle.py`
- **方法**: Maximum Likelihood Estimation
- **特点**:
  - 使用极大似然估计计算先验概率和条件概率
  - 详细输出所有概率计算过程
  - 适用于训练数据充足的情况

**先验概率**:
$$P(Y=c_k) = \frac{N_{c_k}}{N}$$

**条件概率**:
$$P(X^{(j)}=a_{jl}|Y=c_k) = \frac{N_{c_k,jl}}{N_{c_k}}$$

**运行示例**:
```bash
python naive_bayes/naive_bayes_mle.py
```

#### 贝叶斯估计 / 拉普拉斯平滑 (EST)
- **文件**: `naive_bayes/naive_bayes_est.py`
- **方法**: Bayesian Estimation with Laplace Smoothing
- **特点**:
  - 使用贝叶斯估计（加一平滑）避免概率为0
  - $\lambda=1$ 时为拉普拉斯平滑
  - 解决训练数据不足导致的概率估计问题

**先验概率**:
$$P(Y=c_k) = \frac{N_{c_k} + \lambda}{N + K \cdot \lambda}$$

**条件概率**:
$$P(X^{(j)}=a_{jl}|Y=c_k) = \frac{N_{c_k,jl} + \lambda}{N_{c_k} + S_j \cdot \lambda}$$

**运行示例**:
```bash
python naive_bayes/naive_bayes_est.py
```

**训练数据**:
- 特征：X^(1) ∈ {1, 2, 3}, X^(2) ∈ {S, M, L}
- 类别：Y ∈ {1, -1}
- 15个训练样本
- 测试样本：(2, S)
- **预测结果**：两种方法都预测为 Y = -1

## 🛠️ 技术栈

- **Python**: 3.13+
- **NumPy**: 数值计算
- **Pandas**: 数据处理
- **Matplotlib**: 数据可视化
- **Scikit-learn**: 机器学习库（用于对比验证）

## 🚀 快速开始

### 1. 克隆项目
```bash
git clone https://github.com/Kshqsz/machine-learning-lab.git
cd machine-learning-lab
```

### 2. 创建虚拟环境
```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate  # Windows
```

### 3. 安装依赖
```bash
pip install numpy pandas scikit-learn matplotlib
```

### 4. 运行示例
```bash
python linear_regression/gradient_descent.py
```

## 📝 学习笔记

### 线性回归 - 梯度下降
**核心思想**: 通过不断调整参数，使得预测值与真实值之间的误差最小化。

**损失函数**: 均方误差 (MSE)
$$L(w, b) = \frac{1}{n}\sum_{i=1}^{n}(y_i - (wx_i + b))^2$$

**参数更新**:
$$w := w - \alpha \frac{\partial L}{\partial w}$$
$$b := b - \alpha \frac{\partial L}{\partial b}$$

其中 $\alpha$ 是学习率。

---

### 感知机
**核心思想**: 线性可分数据的二分类模型，通过误分类驱动的学习算法找到分离超平面。

**模型**: 
$$f(x) = \text{sign}(w \cdot x + b)$$

**损失函数**: 误分类点到超平面的总距离
$$L(w, b) = -\sum_{x_i \in M} y_i(w \cdot x_i + b)$$

其中 $M$ 是误分类点的集合。

**原始形式更新规则**:
$$w \leftarrow w + \eta y_i x_i$$
$$b \leftarrow b + \eta y_i$$

**对偶形式表示**:
$$f(x) = \text{sign}\left(\sum_{i=1}^{N} \alpha_i y_i x_i \cdot x + b\right)$$

对偶形式的优势是可以预先计算 Gram 矩阵，当样本数远小于特征维度时更高效。

---

### k近邻法
**核心思想**: 给定一个训练数据集，对于新的输入实例，在训练数据集中找到与该实例最近的k个实例，这k个实例的多数属于某个类，就把该输入实例分为这个类。

**距离度量**: 欧氏距离
$$d(x_i, x_j) = \sqrt{\sum_{l=1}^{n}(x_i^{(l)} - x_j^{(l)})^2}$$

**k-d树构造**:
- 选择切分轴：循环选择坐标轴
- 选择切分点：选择该轴坐标的中位数
- 递归构造左右子树

**最近邻搜索**:
1. 从根节点出发，递归地向下访问k-d树
2. 若目标点当前维的坐标小于切分点，则移动到左子节点，否则移动到右子节点
3. 到达叶节点时，计算距离，更新最近点
4. 递归回退，检查是否需要在另一子树中搜索（剪枝）

**时间复杂度**:
- 构造k-d树: $O(n \log n)$
- 搜索: 平均 $O(\log n)$，最坏 $O(n)$

---

### 朴素贝叶斯
**核心思想**: 基于贝叶斯定理与特征条件独立假设，通过训练数据学习先验概率和条件概率，对新实例计算后验概率进行分类。

**贝叶斯定理**:
$$P(Y=c_k|X=x) = \frac{P(X=x|Y=c_k)P(Y=c_k)}{\sum_k P(X=x|Y=c_k)P(Y=c_k)}$$

**条件独立性假设**:
$$P(X=x|Y=c_k) = \prod_{j=1}^{n} P(X^{(j)}=x^{(j)}|Y=c_k)$$

**极大似然估计 (MLE)**:
$$P(Y=c_k) = \frac{N_{c_k}}{N}$$
$$P(X^{(j)}=a_{jl}|Y=c_k) = \frac{N_{c_k,jl}}{N_{c_k}}$$

**贝叶斯估计 (加一平滑)**:
$$P(Y=c_k) = \frac{N_{c_k} + \lambda}{N + K \cdot \lambda}$$
$$P(X^{(j)}=a_{jl}|Y=c_k) = \frac{N_{c_k,jl} + \lambda}{N_{c_k} + S_j \cdot \lambda}$$

其中 $\lambda \geq 0$，常取 $\lambda=1$ (拉普拉斯平滑)。

**分类决策**:
$$y = \arg\max_{c_k} P(Y=c_k) \prod_{j=1}^{n} P(X^{(j)}=x^{(j)}|Y=c_k)$$

## 📈 学习计划与进度

### 监督学习算法

| 状态 | 算法名称 |
|:---:|---------|
| ✅ | **线性回归 - 梯度下降法** |
| ✅ | **感知机 - 原始形式** |
| ✅ | **感知机 - 对偶形式** |
| ✅ | **k近邻法 (k-NN) - k-d树** |
| ✅ | **朴素贝叶斯 - 极大似然估计** |
| ✅ | **朴素贝叶斯 - 贝叶斯估计** |
| ⬜ | 决策树 |
| ⬜ | 逻辑回归与最大熵模型 |
| ⬜ | 支持向量机 (SVM) |
| ⬜ | 提升方法 (AdaBoost) |
| ⬜ | EM算法 |
| ⬜ | 隐马尔可夫模型 |

### 其他算法

| 状态 | 算法名称 |
|:---:|---------|
| ⬜ | 多项式回归 |
| ⬜ | 正则化 (Ridge/Lasso) |
| ⬜ | 聚类算法 (K-Means) |
| ⬜ | 主成分分析 (PCA) |
| ⬜ | 神经网络基础 |

**进度统计**: 已完成 6 / 16 个算法 (37.5%)

## 📖 参考资料

- 《机器学习方法（第2版）》- 李航
- 《机器学习》- 周志华
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [NumPy Documentation](https://numpy.org/doc/)

## 🤝 贡献

这是个人学习项目，欢迎提出建议和改进意见！

## 📧 联系方式

如有问题或建议，欢迎通过 GitHub Issues 联系我。

## 📄 许可证

MIT License

---

⭐ 如果这个项目对你有帮助，欢迎 Star！