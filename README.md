# 机器学习实验室 (Machine Learning Lab)

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

## 📈 学习计划与进度

### 监督学习算法

| 状态 | 算法名称 | 实现文件 |
|:---:|---------|---------|
| ✅ | **线性回归 - 梯度下降法** | `linear_regression/gradient_descent.py` |
| ✅ | **感知机 - 原始形式** | `perceptron/perceptron_primal.py` |
| ✅ | **感知机 - 对偶形式** | `perceptron/perceptron_dual.py` |
| ⬜ | k近邻法 (k-NN) | 待实现 |
| ⬜ | 朴素贝叶斯 | 待实现 |
| ⬜ | 决策树 | 待实现 |
| ⬜ | 逻辑回归与最大熵模型 | 待实现 |
| ⬜ | 支持向量机 (SVM) | 待实现 |
| ⬜ | 提升方法 (AdaBoost) | 待实现 |
| ⬜ | EM算法 | 待实现 |
| ⬜ | 隐马尔可夫模型 | 待实现 |

### 其他算法

| 状态 | 算法名称 | 实现文件 |
|:---:|---------|---------|
| ⬜ | 多项式回归 | 待实现 |
| ⬜ | 正则化 (Ridge/Lasso) | 待实现 |
| ⬜ | 聚类算法 (K-Means) | 待实现 |
| ⬜ | 主成分分析 (PCA) | 待实现 |
| ⬜ | 神经网络基础 | 待实现 |

**进度统计**: 已完成 3 / 16 个算法 (18.8%)

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