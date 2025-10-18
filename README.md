# 机器学习实验室 (Machine Learning Lab)

个人机器学习算法学习与实践项目

## 📚 项目简介

这是我的机器学习学习笔记和实验代码仓库。通过从零开始实现各种经典的机器学习算法，加深对算法原理的理解。

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

## 📈 后续计划

- [ ] 多项式回归
- [ ] 正则化 (Ridge/Lasso)
- [ ] 逻辑回归 (Logistic Regression)
- [ ] 决策树 (Decision Tree)
- [ ] 支持向量机 (SVM)
- [ ] K近邻算法 (KNN)
- [ ] 聚类算法 (K-Means)
- [ ] 神经网络 (Neural Network)
- [ ] 主成分分析 (PCA)

## 📖 参考资料

- 《统计学习方法（第2版）》- 李航
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