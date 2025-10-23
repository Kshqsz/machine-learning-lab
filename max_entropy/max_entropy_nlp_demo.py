"""
最大熵模型 - 词性标注 NLP Demo
参考：《机器学习方法（第2版）》李航 - 第6章

使用最大熵模型实现简单的中文词性标注
最大熵模型等价于多项逻辑斯谛回归
使用梯度下降法优化
"""

import numpy as np
from collections import defaultdict, Counter

class MaxEntropyPOSTagger:
    """
    最大熵词性标注器
    
    模型: P(y|x) = exp(Σw_i*f_i(x,y)) / Z(x)
    其中 Z(x) = Σ_y exp(Σw_i*f_i(x,y)) 是归一化因子
    """
    
    def __init__(self, learning_rate=0.1, max_iter=200, tol=1e-4):
        """
        初始化最大熵模型
        
        参数:
            learning_rate: 学习率
            max_iter: 最大迭代次数
            tol: 收敛阈值
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None  # 特征权重
        self.feature_map = {}  # 特征到索引的映射
        self.tag_map = {}  # 标签到索引的映射
        self.tags = []  # 标签列表
        self.loss_history = []
        
    def extract_features(self, words, pos, index):
        """
        提取特征函数
        
        对于位置index的词，提取以下特征：
        1. 当前词
        2. 前一个词（如果存在）
        3. 后一个词（如果存在）
        4. 当前词的长度
        5. 当前词是否包含数字
        6. 当前词的前缀/后缀
        
        参数:
            words: 词序列
            pos: 当前要标注的词性
            index: 当前词的位置
        """
        features = []
        word = words[index]
        
        # 特征1: 当前词 + 词性
        features.append(f"word={word}|pos={pos}")
        
        # 特征2: 前一个词（如果存在）
        if index > 0:
            features.append(f"prev_word={words[index-1]}|pos={pos}")
        else:
            features.append(f"prev_word=<START>|pos={pos}")
        
        # 特征3: 后一个词（如果存在）
        if index < len(words) - 1:
            features.append(f"next_word={words[index+1]}|pos={pos}")
        else:
            features.append(f"next_word=<END>|pos={pos}")
        
        # 特征4: 词长度
        features.append(f"len={len(word)}|pos={pos}")
        
        # 特征5: 是否包含数字
        has_digit = any(c.isdigit() for c in word)
        features.append(f"has_digit={has_digit}|pos={pos}")
        
        # 特征6: 前缀（如果长度>=2）
        if len(word) >= 2:
            features.append(f"prefix={word[0]}|pos={pos}")
            features.append(f"suffix={word[-1]}|pos={pos}")
        
        # 特征7: 词性本身（偏置）
        features.append(f"bias|pos={pos}")
        
        return features
    
    def build_feature_map(self, training_data):
        """
        构建特征映射
        
        参数:
            training_data: [(words, tags), ...]
        """
        feature_count = defaultdict(int)
        tag_set = set()
        
        for words, tags in training_data:
            for idx, (word, tag) in enumerate(zip(words, tags)):
                tag_set.add(tag)
                features = self.extract_features(words, tag, idx)
                for f in features:
                    feature_count[f] += 1
        
        # 只保留出现次数>=1的特征
        self.feature_map = {f: i for i, (f, c) in enumerate(feature_count.items()) if c >= 1}
        
        # 构建标签映射
        self.tags = sorted(list(tag_set))
        self.tag_map = {tag: i for i, tag in enumerate(self.tags)}
        
        print(f"特征总数: {len(self.feature_map)}")
        print(f"标签总数: {len(self.tags)}")
        print(f"标签: {self.tags}")
    
    def feature_vector(self, words, pos, index):
        """
        将特征转换为向量表示
        """
        vector = np.zeros(len(self.feature_map))
        features = self.extract_features(words, pos, index)
        
        for f in features:
            if f in self.feature_map:
                vector[self.feature_map[f]] = 1
        
        return vector
    
    def compute_probabilities(self, words, index):
        """
        计算所有标签的概率 P(y|x)
        
        返回:
            概率数组 (n_tags,)
        """
        scores = np.zeros(len(self.tags))
        
        for tag_idx, tag in enumerate(self.tags):
            fv = self.feature_vector(words, tag, index)
            scores[tag_idx] = np.dot(self.weights, fv)
        
        # Softmax归一化
        scores = scores - np.max(scores)  # 数值稳定
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores)
        
        return probs
    
    def compute_loss_and_gradient(self, training_data):
        """
        计算损失和梯度
        
        返回:
            loss: 损失值
            grad: 梯度向量
        """
        loss = 0.0
        grad = np.zeros(len(self.weights))
        
        for words, tags in training_data:
            for idx, tag in enumerate(tags):
                # 计算概率
                probs = self.compute_probabilities(words, idx)
                tag_idx = self.tag_map[tag]
                
                # 累加负对数似然
                loss -= np.log(probs[tag_idx] + 1e-10)
                
                # 计算梯度
                # 真实标签的特征
                true_fv = self.feature_vector(words, tag, idx)
                
                # 期望特征（所有可能标签的加权平均）
                expected_fv = np.zeros(len(self.weights))
                for t_idx, t in enumerate(self.tags):
                    fv = self.feature_vector(words, t, idx)
                    expected_fv += probs[t_idx] * fv
                
                # 梯度 = 期望特征 - 真实特征
                grad += (expected_fv - true_fv)
        
        # L2正则化
        loss += 0.01 * np.sum(self.weights ** 2)
        grad += 0.02 * self.weights
        
        return loss, grad
    
    def fit(self, training_data, verbose=True):
        """
        使用梯度下降训练最大熵模型
        
        参数:
            training_data: [(words, tags), ...]
        """
        if verbose:
            print("="*70)
            print("最大熵词性标注器 - 梯度下降优化")
            print("="*70)
            print(f"训练样本数: {len(training_data)}")
        
        # 构建特征映射
        self.build_feature_map(training_data)
        
        # 初始化权重
        n_features = len(self.feature_map)
        self.weights = np.zeros(n_features)
        
        self.loss_history = []
        
        if verbose:
            print(f"学习率: {self.learning_rate}")
            print(f"最大迭代次数: {self.max_iter}")
            print(f"收敛阈值: {self.tol}")
            print("-"*70)
            print("开始梯度下降优化...")
            print()
        
        # 梯度下降迭代
        for iteration in range(self.max_iter):
            # 计算损失和梯度
            loss, grad = self.compute_loss_and_gradient(training_data)
            self.loss_history.append(loss)
            
            # 保存旧权重用于检查收敛
            weights_old = self.weights.copy()
            
            # 更新权重
            self.weights -= self.learning_rate * grad
            
            # 打印进度
            if verbose and (iteration + 1) % 20 == 0:
                print(f"迭代 {iteration + 1:4d} | 损失: {loss:.6f} | "
                      f"梯度范数: {np.linalg.norm(grad):.6f}")
            
            # 检查收敛
            if iteration > 0:
                weight_diff = np.linalg.norm(self.weights - weights_old)
                loss_diff = abs(self.loss_history[-1] - self.loss_history[-2]) if len(self.loss_history) > 1 else float('inf')
                
                if weight_diff < self.tol and loss_diff < self.tol:
                    if verbose:
                        print(f"\n✓ 在第 {iteration + 1} 次迭代时收敛！")
                        print(f"  权重变化: {weight_diff:.2e}")
                        print(f"  损失变化: {loss_diff:.2e}")
                    break
        
        if verbose:
            if iteration == self.max_iter - 1:
                print(f"\n⚠ 达到最大迭代次数 {self.max_iter}")
            print("-"*70)
            print(f"训练完成！最终损失: {self.loss_history[-1]:.4f}")
            print("="*70)
    
    def predict(self, words):
        """
        预测词序列的词性标注
        
        参数:
            words: 词列表
        返回:
            tags: 词性列表
        """
        tags = []
        for idx in range(len(words)):
            probs = self.compute_probabilities(words, idx)
            tag_idx = np.argmax(probs)
            tags.append(self.tags[tag_idx])
        
        return tags
    
    def predict_with_probs(self, words):
        """
        预测词性并返回概率
        """
        results = []
        for idx in range(len(words)):
            probs = self.compute_probabilities(words, idx)
            tag_idx = np.argmax(probs)
            results.append({
                'word': words[idx],
                'tag': self.tags[tag_idx],
                'prob': probs[tag_idx],
                'all_probs': {tag: prob for tag, prob in zip(self.tags, probs)}
            })
        return results
    
    def evaluate(self, test_data):
        """
        评估模型准确率
        """
        correct = 0
        total = 0
        
        for words, true_tags in test_data:
            pred_tags = self.predict(words)
            for true_tag, pred_tag in zip(true_tags, pred_tags):
                if true_tag == pred_tag:
                    correct += 1
                total += 1
        
        return correct / total if total > 0 else 0


def create_training_data():
    """
    创建简单的中文词性标注训练数据
    
    词性标签:
    - n: 名词
    - v: 动词
    - a: 形容词
    - d: 副词
    - p: 介词
    - m: 数词
    - q: 量词
    - r: 代词
    """
    training_data = [
        # 句子1
        (["我", "爱", "中国"], ["r", "v", "n"]),
        (["他", "喜欢", "音乐"], ["r", "v", "n"]),
        (["她", "学习", "英语"], ["r", "v", "n"]),
        (["我们", "热爱", "祖国"], ["r", "v", "n"]),
        
        # 句子2
        (["这", "是", "一", "本", "书"], ["r", "v", "m", "q", "n"]),
        (["那", "是", "两", "个", "苹果"], ["r", "v", "m", "q", "n"]),
        (["这里", "有", "三", "只", "猫"], ["r", "v", "m", "q", "n"]),
        
        # 句子3
        (["天气", "很", "好"], ["n", "d", "a"]),
        (["花儿", "很", "美"], ["n", "d", "a"]),
        (["房子", "很", "大"], ["n", "d", "a"]),
        (["孩子", "很", "聪明"], ["n", "d", "a"]),
        
        # 句子4
        (["他", "在", "学校", "学习"], ["r", "p", "n", "v"]),
        (["我", "在", "家里", "休息"], ["r", "p", "n", "v"]),
        (["她", "在", "公园", "散步"], ["r", "p", "n", "v"]),
        
        # 句子5
        (["小明", "看", "书"], ["n", "v", "n"]),
        (["老师", "教", "学生"], ["n", "v", "n"]),
        (["妈妈", "做", "饭"], ["n", "v", "n"]),
        (["爸爸", "开", "车"], ["n", "v", "n"]),
        
        # 句子6 - 更复杂的
        (["我", "非常", "喜欢", "这个", "城市"], ["r", "d", "v", "r", "n"]),
        (["他们", "认真", "学习", "新", "知识"], ["r", "d", "v", "a", "n"]),
        (["今天", "天气", "真", "好"], ["n", "n", "d", "a"]),
    ]
    
    return training_data


def create_test_data():
    """
    创建测试数据
    """
    test_data = [
        (["我", "学习", "数学"], ["r", "v", "n"]),
        (["这", "是", "五", "本", "书"], ["r", "v", "m", "q", "n"]),
        (["天气", "很", "冷"], ["n", "d", "a"]),
        (["她", "在", "图书馆", "看书"], ["r", "p", "n", "v"]),
        (["老师", "讲", "课"], ["n", "v", "n"]),
    ]
    return test_data


def main():
    """
    主函数
    """
    print("\n" + "="*70)
    print("最大熵模型 - 中文词性标注 Demo")
    print("="*70)
    
    # 词性标签说明
    pos_description = {
        'n': '名词 (Noun)',
        'v': '动词 (Verb)',
        'a': '形容词 (Adjective)',
        'd': '副词 (Adverb)',
        'p': '介词 (Preposition)',
        'm': '数词 (Numeral)',
        'q': '量词 (Quantifier)',
        'r': '代词 (Pronoun)'
    }
    
    print("\n词性标签说明:")
    print("-"*70)
    for tag, desc in pos_description.items():
        print(f"  {tag}: {desc}")
    print("-"*70)
    
    # 创建训练和测试数据
    training_data = create_training_data()
    test_data = create_test_data()
    
    print(f"\n训练集: {len(training_data)} 个句子")
    print(f"测试集: {len(test_data)} 个句子")
    print()
    
    # 训练模型
    model = MaxEntropyPOSTagger(max_iter=50)
    model.fit(training_data, verbose=True)
    
    # 评估模型
    print("\n模型评估:")
    print("="*70)
    train_acc = model.evaluate(training_data)
    test_acc = model.evaluate(test_data)
    print(f"训练集准确率: {train_acc*100:.2f}%")
    print(f"测试集准确率: {test_acc*100:.2f}%")
    print("="*70)
    
    # 显示训练集上的预测结果
    print("\n训练集预测示例 (前5个句子):")
    print("-"*70)
    for i, (words, true_tags) in enumerate(training_data[:5]):
        pred_tags = model.predict(words)
        correct = sum(1 for t, p in zip(true_tags, pred_tags) if t == p)
        
        print(f"\n句子 {i+1}: {' '.join(words)}")
        print(f"真实标注: {' '.join(true_tags)}")
        print(f"预测标注: {' '.join(pred_tags)}")
        print(f"准确率: {correct}/{len(words)} ({correct/len(words)*100:.1f}%)")
    
    # 显示测试集上的预测结果
    print("\n" + "="*70)
    print("测试集预测结果:")
    print("-"*70)
    for i, (words, true_tags) in enumerate(test_data):
        pred_tags = model.predict(words)
        correct = sum(1 for t, p in zip(true_tags, pred_tags) if t == p)
        
        print(f"\n句子 {i+1}: {' '.join(words)}")
        print(f"真实标注: {' '.join(true_tags)}")
        print(f"预测标注: {' '.join(pred_tags)}")
        
        # 标记错误
        if correct < len(words):
            errors = []
            for j, (w, t, p) in enumerate(zip(words, true_tags, pred_tags)):
                if t != p:
                    errors.append(f"{w}({t}→{p})")
            print(f"错误: {', '.join(errors)}")
        
        print(f"准确率: {correct}/{len(words)} ({correct/len(words)*100:.1f}%)")
    
    # 新句子预测（带概率）
    print("\n" + "="*70)
    print("新句子预测 (带概率分布):")
    print("="*70)
    
    new_sentences = [
        ["我", "喜欢", "猫"],
        ["这", "是", "六", "个", "杯子"],
        ["房间", "很", "干净"],
    ]
    
    for i, words in enumerate(new_sentences):
        print(f"\n句子 {i+1}: {' '.join(words)}")
        results = model.predict_with_probs(words)
        
        print("\n详细分析:")
        for res in results:
            print(f"  词: {res['word']:4s} | 词性: {res['tag']} ({res['prob']:.3f})")
            # 显示Top-3概率
            top3 = sorted(res['all_probs'].items(), key=lambda x: x[1], reverse=True)[:3]
            probs_str = ", ".join([f"{tag}:{prob:.3f}" for tag, prob in top3])
            print(f"           Top-3: {probs_str}")
        
        pred_tags = [res['tag'] for res in results]
        print(f"\n  完整标注: {' / '.join([f'{w}({t})' for w, t in zip(words, pred_tags)])}")
    
    print("\n" + "="*70)
    print("Demo完成！")
    print("="*70)


if __name__ == "__main__":
    main()
