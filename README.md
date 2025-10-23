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
├── decision_tree/              # 决策树
│   ├── decision_tree_classifier.py  # 分类树(基尼指数)
│   └── decision_tree_regressor.py   # 回归树(MSE)
├── logistic_regression/        # 逻辑斯谛回归
│   ├── binomial_logistic_regression.py   # 二项逻辑斯谛回归
│   └── multinomial_logisitic_regression.py  # 多项逻辑斯谛回归
├── max_entropy/                # 最大熵模型
│   └── max_entropy_nlp_demo.py  # 中文词性标注Demo
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

---

### 5. 决策树 (Decision Tree)

决策树是一种基本的分类与回归方法，通过树形结构进行决策。

#### 分类树 (Classification Tree)
- **文件**: `decision_tree/decision_tree_classifier.py`
- **分裂准则**: 基尼指数 (Gini Index)
- **特点**:
  - 使用基尼指数选择最优特征和切分点
  - 递归构建决策树
  - 支持类别型特征（自动编码）
  - 输出可读的决策规则
  - 适用于分类问题

**基尼指数**:
$$\text{Gini}(D) = 1 - \sum_{k=1}^{K} p_k^2$$

**分裂后的基尼指数**:
$$\text{Gini}(D, A) = \frac{|D_1|}{|D|}\text{Gini}(D_1) + \frac{|D_2|}{|D|}\text{Gini}(D_2)$$

**运行示例**:
```bash
python decision_tree/decision_tree_classifier.py
```

**训练数据** - 贷款审批数据集:
- 14个样本，4个特征（年龄、有工作、有房子、信贷情况）
- 2个类别（同意贷款、拒绝贷款）
- 训练集准确率：100%
- **决策规则示例**: 
  - 有房子 → 同意贷款
  - 无房子且信贷好 → 同意贷款
  - 无房子且信贷一般 → 拒绝贷款

#### 回归树 (Regression Tree)
- **文件**: `decision_tree/decision_tree_regressor.py`
- **分裂准则**: 均方误差 (MSE - Mean Squared Error)
- **特点**:
  - 使用MSE最小化选择分裂点
  - 叶节点预测值为区域内样本均值
  - 可视化拟合曲线
  - 适用于回归问题

**均方误差**:

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \bar{y})^2$$

**分裂后的MSE**:

$$\text{MSE}_\text{split} = \frac{n_\text{left}}{n}\text{MSE}_\text{left} + \frac{n_\text{right}}{n}\text{MSE}_\text{right}$$

**运行示例**:
```bash
python decision_tree/decision_tree_regressor.py
```

**训练数据**:
- 10个样本点：(1, 4.50), (2, 4.75), ..., (10, 9.00)
- 训练集 MSE: 0.0525
- 训练集 R²: 0.9810

---

### 6. 逻辑斯谛回归 (Logistic Regression)

逻辑斯谛回归是一种广义线性模型，用于解决分类问题，特别是二分类问题。

#### 二项逻辑斯谛回归 (Binomial Logistic Regression)
- **文件**: `logistic_regression/binomial_logistic_regression.py`
- **方法**: 梯度下降法
- **特点**:
  - 使用 Sigmoid 函数将线性组合映射到 [0,1] 区间
  - 基于极大似然估计的损失函数
  - 梯度下降法优化参数
  - 可视化决策边界和训练过程
  - 适用于二分类问题

**Sigmoid 函数**:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**模型**:

$$P(Y=1|x) = \frac{1}{1 + \exp(-(w \cdot x + b))}$$

**损失函数（负对数似然）**:

$$L(w,b) = -\frac{1}{n}\sum_{i=1}^{n}[y_i \log(p_i) + (1-y_i)\log(1-p_i)]$$

**梯度**:

$$\frac{\partial L}{\partial w} = \frac{1}{n}\sum_{i=1}^{n}(p_i - y_i)x_i$$

$$\frac{\partial L}{\partial b} = \frac{1}{n}\sum_{i=1}^{n}(p_i - y_i)$$

**运行示例**:
```bash
python logistic_regression/binomial_logistic_regression.py
```

**训练数据** - 学生考试通过预测:
- 20个样本：学习时长（0.5-5.5小时）与考试结果（通过/未通过）
- 通过人数：10人，未通过人数：10人
- 训练集准确率：80.00%
- **决策边界**: 学习时长约 2.7 小时
- **预测示例**:
  - 学习 1.0 小时 → 通过概率 7% → 未通过
  - 学习 3.0 小时 → 通过概率 61% → 通过
  - 学习 5.0 小时 → 通过概率 97% → 通过

#### 多项逻辑斯谛回归 (Multinomial Logistic Regression)
- **文件**: `logistic_regression/multinomial_logisitic_regression.py`
- **方法**: BFGS拟牛顿法
- **特点**:
  - 使用 Softmax 函数处理多分类问题
  - 基于极大似然估计
  - BFGS优化算法，收敛速度快
  - 可视化多分类决策边界
  - 适用于多分类问题（K ≥ 2）

**Softmax 函数**:

$$P(Y=k|x) = \frac{\exp(w_k \cdot x + b_k)}{\sum_{j=1}^{K}\exp(w_j \cdot x + b_j)}$$

**模型**:

对于 K 个类别，需要学习 K 组参数 $(w_1, b_1), ..., (w_K, b_K)$

**损失函数（交叉熵）**:

$$J(W,b) = -\frac{1}{n}\sum_{i=1}^{n}\sum_{k=1}^{K} \mathbb{1}(y_i=k) \log P(Y=k|x_i)$$

其中 $\mathbb{1}(\cdot)$ 是指示函数。

**梯度**:

$$\frac{\partial J}{\partial w_k} = \frac{1}{n}\sum_{i=1}^{n}(P(Y=k|x_i) - \mathbb{1}(y_i=k))x_i$$

$$\frac{\partial J}{\partial b_k} = \frac{1}{n}\sum_{i=1}^{n}(P(Y=k|x_i) - \mathbb{1}(y_i=k))$$

**运行示例**:
```bash
python logistic_regression/multinomial_logisitic_regression.py
```

**训练数据** - 鸢尾花分类:
- 48个样本：花瓣长度、花瓣宽度 → 3种鸢尾花（Setosa、Versicolor、Virginica）
- 类别分布：Setosa 16个，Versicolor 16个，Virginica 16个
- 训练集准确率：77.08%
- **特点**: 使用BFGS优化，自动计算梯度，收敛快且稳定
- **决策边界**: 可视化展示三个类别的非线性分离超平面

---

### 7. 最大熵模型 (Maximum Entropy Model)

最大熵模型是一种基于最大熵原理的分类模型，属于对数线性模型。在满足已知约束条件的前提下，选择熵最大的模型。

#### 最大熵NLP演示 - 中文词性标注 (POS Tagging)
- **文件**: `max_entropy/max_entropy_nlp_demo.py`
- **方法**: 梯度下降法
- **应用场景**: 中文词性标注（Part-of-Speech Tagging）
- **特点**:
  - 丰富的特征工程（7种特征类型）
  - 手工标注的中文训练数据
  - 支持8种常见词性标签
  - 梯度下降优化，过程可视化
  - 完整的训练、测试和预测流程

**最大熵模型**:

条件概率分布：

$$P(y|x) = \frac{1}{Z(x)}\exp\left(\sum_{i=1}^{n}w_i f_i(x,y)\right)$$

归一化因子：

$$Z(x) = \sum_{y}\exp\left(\sum_{i=1}^{n}w_i f_i(x,y)\right)$$

**与多项逻辑斯谛回归的关系**:

最大熵模型在形式上等价于多项逻辑斯谛回归：
- 都使用 Softmax 进行归一化
- 都是对数线性模型
- 区别在于特征函数的构造方式

**特征工程** - 7种特征类型:

1. **当前词特征**: `word=我`, `word=喜欢`
2. **前一个词**: `prev_word=我`, `prev_word=喜欢`
3. **后一个词**: `next_word=喜欢`, `next_word=中国`
4. **词长度**: `word_len=1`, `word_len=2`
5. **包含数字**: `has_digit=True`
6. **前缀**: `prefix_1=学`, `prefix_2=学习`
7. **后缀**: `suffix_1=习`, `suffix_2=学习`
8. **偏置**: `bias=1`（所有样本）

**词性标签** (8种):

| 标签 | 词性 | 示例 |
|:---:|------|------|
| n | 名词 (Noun) | 中国、音乐、书 |
| v | 动词 (Verb) | 爱、喜欢、学习 |
| a | 形容词 (Adjective) | 好、冷、干净 |
| d | 副词 (Adverb) | 很、非常、都 |
| p | 介词 (Preposition) | 在、从、对 |
| m | 数词 (Numeral) | 一、五、十 |
| q | 量词 (Quantifier) | 本、个、条 |
| r | 代词 (Pronoun) | 我、他、这 |

**损失函数**:

负对数似然 + L2正则化：

$$L(w) = -\sum_{(x,y)}\log P(y|x) + \lambda \|w\|^2$$

**梯度**:

$$\frac{\partial L}{\partial w_i} = \sum_{(x,y)}[P(y|x)f_i(x,y) - f_i(x,y_{\text{true}})] + 2\lambda w_i$$

**优化算法**:

使用梯度下降法：
- 学习率: 0.1
- 最大迭代: 50次
- 收敛阈值: 1e-4
- 监控权重变化和损失变化

**运行示例**:
```bash
python max_entropy/max_entropy_nlp_demo.py
```

**训练数据** - 中文句子标注:
- 训练集：21个句子，涵盖日常用语
- 测试集：5个句子
- 特征总数：252个
- 训练集准确率：100.00%
- 测试集准确率：94.44%

**标注示例**:
```
句子: 我 喜欢 音乐
标注: 我(r) 喜欢(v) 音乐(n)

句子: 这 是 一 本 书
标注: 这(r) 是(v) 一(m) 本(q) 书(n)

句子: 天气 很 冷
标注: 天气(n) 很(d) 冷(a)
```

**预测功能**:
- 对新句子进行词性标注
- 输出每个词的Top-3概率分布
- 可视化预测结果

**优势**:
- 特征灵活，易于添加新特征
- 概率输出，具有可解释性
- 适用于序列标注任务
- 训练过程透明，可监控优化进展

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

---

### 决策树
**核心思想**: 通过树形结构表示决策过程，每个内部节点表示一个特征上的测试，每个分支代表测试结果，每个叶节点存放一个类标记或预测值。

**分类树 - 基尼指数**:

基尼指数表示集合的不纯度：
$$\text{Gini}(D) = 1 - \sum_{k=1}^{K} p_k^2$$

其中 $p_k$ 是样本属于第k类的概率。

特征A条件下的基尼指数：
$$\text{Gini}(D, A) = \frac{|D_1|}{|D|}\text{Gini}(D_1) + \frac{|D_2|}{|D|}\text{Gini}(D_2)$$

选择基尼指数最小的特征及其切分点。

**回归树 - 均方误差**:

划分点s处的平方误差：
$$\min_{s}\left[\min_{c_1}\sum_{x_i \in R_1(s)}(y_i - c_1)^2 + \min_{c_2}\sum_{x_i \in R_2(s)}(y_i - c_2)^2\right]$$

其中 $c_1$ 和 $c_2$ 分别是左右区域的输出值（均值）。

**停止条件**:
- 节点中样本属于同一类别
- 达到最大深度
- 样本数小于最小分裂数
- MSE/基尼指数减少量小于阈值

---

### 逻辑斯谛回归
**核心思想**: 通过 Sigmoid 函数将线性模型的输出映射到 (0,1) 区间，表示样本属于某类的概率，是一种广义线性模型。

**Sigmoid 函数**:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

性质：
- 值域为 (0, 1)，可解释为概率
- 单调递增
- $\sigma(0) = 0.5$

**二项逻辑斯谛回归模型**:

$$P(Y=1|x) = \frac{1}{1 + \exp(-(w \cdot x + b))}$$

$$P(Y=0|x) = 1 - P(Y=1|x)$$

**极大似然估计**:

似然函数：

$$L(w,b) = \prod_{i=1}^{n} [p_i]^{y_i}[1-p_i]^{1-y_i}$$

对数似然函数：

$$\log L(w,b) = \sum_{i=1}^{n}[y_i \log(p_i) + (1-y_i)\log(1-p_i)]$$

**损失函数（负对数似然）**:

$$J(w,b) = -\frac{1}{n}\sum_{i=1}^{n}[y_i \log(p_i) + (1-y_i)\log(1-p_i)]$$

这也称为交叉熵损失（Cross-Entropy Loss）。

**梯度下降更新规则**:

梯度：

$$\frac{\partial J}{\partial w} = \frac{1}{n}\sum_{i=1}^{n}(p_i - y_i)x_i$$

$$\frac{\partial J}{\partial b} = \frac{1}{n}\sum_{i=1}^{n}(p_i - y_i)$$

参数更新：

$$w \leftarrow w - \alpha \frac{\partial J}{\partial w}$$

$$b \leftarrow b - \alpha \frac{\partial J}{\partial b}$$

其中 $\alpha$ 是学习率。

**决策边界**:

当 $P(Y=1|x) = 0.5$ 时，即 $w \cdot x + b = 0$，这就是决策边界。

对于一维特征：

$$x = -\frac{b}{w}$$

**优点**:
- 输出具有概率意义
- 计算代价低，易于实现
- 可解释性强

**局限性**:
- 只能处理线性可分或近似线性可分的问题
- 对特征共线性敏感

---

### 多项逻辑斯谛回归
**核心思想**: 将二项逻辑斯谛回归推广到多分类问题，使用 Softmax 函数将线性输出转换为概率分布。

**Softmax 函数**:
$$P(Y=k|x) = \frac{\exp(w_k \cdot x + b_k)}{\sum_{j=1}^{K}\exp(w_j \cdot x + b_j)}$$

性质：
- 输出K个概率值，和为1
- 单调性：线性得分越高，概率越大
- 当K=2时退化为二项逻辑斯谛回归

**参数**:
对于K个类别，需要学习K组参数：
$$(w_1, b_1), (w_2, b_2), ..., (w_K, b_K)$$

**损失函数（交叉熵）**:

$$J(W,b) = -\frac{1}{n}\sum_{i=1}^{n}\sum_{k=1}^{K} \mathbb{1}(y_i=k) \log P(Y=k|x_i)$$

其中 $\mathbb{1}(\cdot)$ 是指示函数，当 $y_i=k$ 时为1，否则为0。

**梯度计算**:

对于第k类的参数：

$$\frac{\partial J}{\partial w_k} = \frac{1}{n}\sum_{i=1}^{n}(P(Y=k|x_i) - \mathbb{1}(y_i=k))x_i$$

$$\frac{\partial J}{\partial b_k} = \frac{1}{n}\sum_{i=1}^{n}(P(Y=k|x_i) - \mathbb{1}(y_i=k))$$

**优化算法**:
- 梯度下降法：简单但可能较慢
- BFGS拟牛顿法：自动计算近似Hessian矩阵，收敛快
- L-BFGS：BFGS的内存优化版本

**决策规则**:
$$\hat{y} = \arg\max_k P(Y=k|x)$$

选择概率最大的类别作为预测结果。

---

### 最大熵模型
**核心思想**: 在满足约束条件的前提下，选择熵最大的概率分布。熵最大意味着对未知信息不做任何主观假设，是一种最保守的策略。

**熵的定义**:
$$H(P) = -\sum_{x,y} \tilde{P}(x) P(y|x) \log P(y|x)$$

其中 $\tilde{P}(x)$ 是经验分布。

**特征函数**:
$$f_i(x, y) = \begin{cases} 1, & \text{如果}x,y\text{满足某个事实} \\ 0, & \text{否则} \end{cases}$$

**约束条件**:

模型期望 = 经验期望

$$E_P[f_i] = E_{\tilde{P}}[f_i]$$

即：

$$\sum_{x,y} \tilde{P}(x) P(y|x) f_i(x,y) = \sum_{x,y} \tilde{P}(x,y) f_i(x,y)$$

**最大熵模型**:

最优解具有指数形式：

$$P_w(y|x) = \frac{1}{Z_w(x)} \exp\left(\sum_{i=1}^{n} w_i f_i(x,y)\right)$$

归一化因子：

$$Z_w(x) = \sum_y \exp\left(\sum_{i=1}^{n} w_i f_i(x,y)\right)$$

**与多项逻辑斯谛回归的关系**:

最大熵模型在数学形式上等价于多项逻辑斯谛回归：
- 都使用 Softmax 归一化
- 都是对数线性模型
- 都用交叉熵作为损失函数

区别：
- **视角不同**: 最大熵从信息论角度（最大化熵），逻辑回归从概率角度（最大似然）
- **特征构造**: 最大熵强调特征函数 $f_i(x,y)$，逻辑回归强调特征向量 $x$

**极大似然估计**:

对偶问题：

$$\max_w \sum_{x,y} \tilde{P}(x,y) \log P_w(y|x)$$

等价于最小化负对数似然：

$$\min_w -\sum_{x,y} \tilde{P}(x,y) \log P_w(y|x)$$

**梯度**:

$$\frac{\partial L}{\partial w_i} = \sum_{x,y} \tilde{P}(x) [P_w(y|x) f_i(x,y) - \tilde{P}(y|x) f_i(x,y)]$$

简化为：

$$\frac{\partial L}{\partial w_i} = E_P[f_i] - E_{\tilde{P}}[f_i]$$

即模型期望与经验期望的差。

**优化算法**:
- 梯度下降法 (GD)
- 拟牛顿法 (BFGS)
- 改进的迭代尺度法 (IIS)
- 通用迭代尺度法 (GIS)

**应用场景**:
- 自然语言处理（词性标注、命名实体识别）
- 文本分类
- 信息抽取
- 机器翻译

**优势**:
- 特征灵活，可以组合任意特征
- 理论基础扎实（最大熵原理）
- 可解释性强
- 不需要特征独立性假设（相比朴素贝叶斯）

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
| ✅ | **决策树 - 分类树 (基尼指数)** |
| ✅ | **决策树 - 回归树 (MSE)** |
| ✅ | **逻辑斯谛回归 - 二项逻辑斯谛回归** |
| ✅ | **逻辑斯谛回归 - 多项逻辑斯谛回归** |
| ✅ | **最大熵模型 - 中文词性标注** |
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

**进度统计**: 已完成 11 / 17 个算法 (64.7%)

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