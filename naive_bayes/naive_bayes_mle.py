"""
朴素贝叶斯分类器 - 极大似然估计 (Maximum Likelihood Estimation)
参考：《机器学习方法（第2版）》李航 - 第4章
"""

import numpy as np
from collections import defaultdict

# 设置打印选项
np.set_printoptions(precision=4, suppress=True)


class NaiveBayesMLE:
    """
    朴素贝叶斯分类器 - 极大似然估计
    
    使用极大似然估计来估计先验概率和条件概率
    """
    
    def __init__(self):
        self.prior_prob = {}  # 先验概率 P(Y=ck)
        self.conditional_prob = {}  # 条件概率 P(X=x|Y=ck)
        self.classes = None  # 类别集合
        self.features = None  # 特征集合
        
    def fit(self, X, y):
        """
        训练朴素贝叶斯模型 - 极大似然估计
        
        参数:
            X: 训练数据特征, shape (n_samples, n_features)
            y: 训练数据标签
        """
        n_samples = len(y)
        self.classes = np.unique(y)
        n_features = X.shape[1]
        
        print("=" * 70)
        print("朴素贝叶斯分类器 - 极大似然估计 (MLE)")
        print("=" * 70)
        print(f"\n训练样本数: {n_samples}")
        print(f"特征数: {n_features}")
        print(f"类别: {self.classes}")
        print()
        
        # 计算先验概率 P(Y=ck)
        print("1. 计算先验概率 P(Y=ck)")
        print("-" * 70)
        for c in self.classes:
            count = np.sum(y == c)
            self.prior_prob[c] = count / n_samples
            print(f"P(Y={c:2}) = {count}/{n_samples} = {self.prior_prob[c]:.4f}")
        
        # 计算条件概率 P(X^(j)=a_jl | Y=ck)
        print(f"\n2. 计算条件概率 P(X^(j)=a_jl | Y=ck)")
        print("-" * 70)
        
        # 获取每个特征的所有可能取值
        self.features = []
        for j in range(n_features):
            unique_values = np.unique(X[:, j])
            self.features.append(unique_values)
            print(f"特征 X^({j+1}) 的可能取值: {unique_values}")
        print()
        
        # 对每个类别计算条件概率
        for c in self.classes:
            self.conditional_prob[c] = {}
            X_c = X[y == c]  # 类别为c的所有样本
            n_c = len(X_c)
            
            print(f"类别 Y={c} (共 {n_c} 个样本):")
            
            for j in range(n_features):
                self.conditional_prob[c][j] = {}
                
                for value in self.features[j]:
                    # 统计在类别c中，特征j取值为value的样本数
                    count = np.sum(X_c[:, j] == value)
                    # 极大似然估计
                    prob = count / n_c
                    self.conditional_prob[c][j][value] = prob
                    
                    print(f"  P(X^({j+1})={value} | Y={c}) = {count}/{n_c} = {prob:.4f}")
            print()
    
    def predict(self, X):
        """
        预测
        
        参数:
            X: 待预测数据, shape (n_samples, n_features) 或 (n_features,)
        
        返回:
            预测的类别
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        predictions = []
        for x in X:
            predictions.append(self._predict_single(x))
        
        return np.array(predictions) if len(predictions) > 1 else predictions[0]
    
    def _predict_single(self, x):
        """
        预测单个样本
        """
        print("\n" + "=" * 70)
        print(f"对样本 x = {x} 进行分类预测")
        print("=" * 70)
        
        posteriors = {}
        
        for c in self.classes:
            # 计算 P(Y=c) * ∏ P(X^(j)=x^(j) | Y=c)
            prior = self.prior_prob[c]
            likelihood = 1.0
            
            print(f"\n计算 P(Y={c}) * ∏ P(X^(j)=x^(j) | Y={c}):")
            print(f"  P(Y={c}) = {prior:.4f}")
            
            for j, value in enumerate(x):
                cond_prob = self.conditional_prob[c][j].get(value, 0)
                likelihood *= cond_prob
                print(f"  P(X^({j+1})={value} | Y={c}) = {cond_prob:.4f}")
            
            posterior = prior * likelihood
            posteriors[c] = posterior
            
            print(f"  P(Y={c}) * ∏ P(X^(j)=x^(j) | Y={c}) = {prior:.4f} * {likelihood:.4f} = {posterior:.4f}")
        
        # 选择后验概率最大的类别
        predicted_class = max(posteriors, key=posteriors.get)
        
        print(f"\n后验概率比较:")
        for c in self.classes:
            symbol = "← 最大" if c == predicted_class else ""
            print(f"  P(Y={c} | X=x) ∝ {posteriors[c]:.6f} {symbol}")
        
        print(f"\n预测结果: Y = {predicted_class}")
        
        return predicted_class


def main():
    """
    主函数
    """
    # 训练数据
    # A1 ∈ {1, 2, 3}, A2 ∈ {S, M, L}, Y ∈ {1, -1}
    X_train = np.array([
        [1, 'S'],
        [1, 'M'],
        [1, 'M'],
        [1, 'S'],
        [1, 'S'],
        [2, 'S'],
        [2, 'M'],
        [2, 'M'],
        [2, 'L'],
        [2, 'L'],
        [3, 'L'],
        [3, 'M'],
        [3, 'M'],
        [3, 'L'],
        [3, 'L'],
    ])
    
    y_train = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])
    
    print("训练数据集:")
    print("-" * 70)
    print("序号 | X^(1) | X^(2) | Y")
    print("-" * 70)
    for i, (x, y) in enumerate(zip(X_train, y_train), 1):
        print(f"{i:4} | {x[0]:5} | {x[1]:5} | {y:2}")
    print("-" * 70)
    print()
    
    # 创建并训练模型
    model = NaiveBayesMLE()
    model.fit(X_train, y_train)
    
    # 预测
    x_test = np.array([2, 'S'])
    prediction = model.predict(x_test)
    
    print("\n" + "=" * 70)
    print("最终结论")
    print("=" * 70)
    print(f"使用极大似然估计 (MLE)，对样本 {x_test} 的预测类别为: {prediction}")
    print("=" * 70)


if __name__ == "__main__":
    main()
