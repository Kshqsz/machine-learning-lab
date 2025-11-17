"""
概率潜在语义分析 (Probabilistic Latent Semantic Analysis, PLSA)
================================================================

PLSA是一种生成式概率模型，通过EM算法学习文档-主题和主题-词的概率分布。
与LSA相比，PLSA提供了概率解释；与NMF相比，PLSA有严格的概率框架。

理论基础
--------

1. **生成模型**:
   PLSA假设文档中的每个词都是通过以下过程生成的：
   
   (1) 选择一个文档 d，概率为 P(d)
   (2) 在文档d中选择一个隐含主题 z，概率为 P(z|d)
   (3) 在主题z下选择一个词 w，概率为 P(w|z)
   
   因此，词w在文档d中出现的概率为：
   $$P(w|d) = \sum_{z=1}^K P(w|z) P(z|d)$$

2. **联合概率**:
   $$P(d, w) = P(d) P(w|d) = P(d) \sum_{z=1}^K P(z|d) P(w|z)$$
   
   或等价地：
   $$P(d, w) = \sum_{z=1}^K P(z) P(d|z) P(w|z)$$

3. **对数似然函数**:
   给定观测到的词-文档共现矩阵 n(d,w)，对数似然为：
   $$L = \sum_{d=1}^D \sum_{w=1}^W n(d,w) \log P(w|d)$$
   $$= \sum_{d,w} n(d,w) \log \sum_{z=1}^K P(w|z) P(z|d)$$

4. **EM算法**:
   
   **E步** - 计算后验概率（责任度）:
   $$P(z|d,w) = \frac{P(w|z) P(z|d)}{\sum_{z'=1}^K P(w|z') P(z'|d)}$$
   
   **M步** - 更新参数:
   $$P(w|z) = \frac{\sum_{d=1}^D n(d,w) P(z|d,w)}{\sum_{w'=1}^W \sum_{d=1}^D n(d,w') P(z|d,w')}$$
   
   $$P(z|d) = \frac{\sum_{w=1}^W n(d,w) P(z|d,w)}{\sum_{w'=1}^W n(d,w')}$$

5. **模型参数**:
   - P(w|z): 主题-词分布，K×W矩阵（K个主题，W个词）
   - P(z|d): 文档-主题分布，D×K矩阵（D个文档，K个主题）
   - 约束条件：∑_w P(w|z) = 1, ∑_z P(z|d) = 1

6. **PLSA vs LSA vs NMF**:

   | 特性 | PLSA | LSA | NMF |
   |------|------|-----|-----|
   | 模型类型 | 概率生成模型 | 线性代数 | 矩阵分解 |
   | 参数解释 | 概率分布 | 向量空间 | 非负因子 |
   | 学习算法 | EM算法 | SVD | 乘法更新 |
   | 可解释性 | 强（概率语义） | 弱 | 强（部分表示） |
   | 过拟合处理 | 需要正则化 | 降维天然正则 | 非负约束 |

算法步骤
--------

1. 初始化P(w|z)和P(z|d)为随机概率分布
2. E步：计算P(z|d,w)（后验概率）
3. M步：更新P(w|z)和P(z|d)
4. 计算对数似然
5. 检查收敛，否则回到步骤2

优势
----
- 严格的概率框架，易于扩展
- 可以计算困惑度（perplexity）评估模型
- 支持贝叶斯推断和正则化
- 可以处理新文档（折叠推断）

"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
from collections import Counter
import re
import warnings

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class PLSA:
    """
    概率潜在语义分析 (Probabilistic Latent Semantic Analysis)
    
    使用EM算法学习文档-主题和主题-词的概率分布
    
    参数
    ----
    n_topics : int
        主题数量（K）
    max_iter : int
        最大迭代次数
    tol : float
        收敛容差（对数似然变化）
    random_state : int
        随机种子
    verbose : bool
        是否显示训练信息
    """
    
    def __init__(
        self,
        n_topics: int = 10,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        verbose: bool = False
    ):
        self.n_topics = n_topics
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        
        # 模型参数
        self.P_w_z = None  # P(w|z): 主题-词分布 (n_topics, n_words)
        self.P_z_d = None  # P(z|d): 文档-主题分布 (n_docs, n_topics)
        
        # 词汇表
        self.vocabulary_ = {}  # 词 -> 索引
        self.idx_to_word_ = {}  # 索引 -> 词
        
        # 训练历史
        self.loglikelihood_history_ = []
        self.perplexity_history_ = []
        self.n_iter_ = 0
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _tokenize(self, text: str) -> List[str]:
        """简单的分词器"""
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def _build_vocabulary(self, documents: List[str]) -> Dict[str, int]:
        """构建词汇表"""
        vocab = {}
        idx = 0
        for doc in documents:
            words = self._tokenize(doc)
            for word in words:
                if word not in vocab:
                    vocab[word] = idx
                    idx += 1
        return vocab
    
    def _build_word_doc_matrix(self, documents: List[str]) -> np.ndarray:
        """
        构建词-文档共现矩阵 n(d,w)
        
        返回
        ----
        n_dw : np.ndarray
            形状 (n_docs, n_words)，元素n_dw[d,w]表示词w在文档d中出现的次数
        """
        n_docs = len(documents)
        n_words = len(self.vocabulary_)
        
        n_dw = np.zeros((n_docs, n_words))
        
        for d, doc in enumerate(documents):
            words = self._tokenize(doc)
            word_counts = Counter(words)
            
            for word, count in word_counts.items():
                if word in self.vocabulary_:
                    w = self.vocabulary_[word]
                    n_dw[d, w] = count
        
        return n_dw
    
    def _initialize_parameters(self, n_docs: int, n_words: int):
        """
        随机初始化模型参数
        
        P(w|z): 主题-词分布
        P(z|d): 文档-主题分布
        """
        # P(w|z): (n_topics, n_words)
        # 每行是一个主题的词分布，和为1
        self.P_w_z = np.random.rand(self.n_topics, n_words)
        self.P_w_z = self.P_w_z / self.P_w_z.sum(axis=1, keepdims=True)
        
        # P(z|d): (n_docs, n_topics)
        # 每行是一个文档的主题分布，和为1
        self.P_z_d = np.random.rand(n_docs, self.n_topics)
        self.P_z_d = self.P_z_d / self.P_z_d.sum(axis=1, keepdims=True)
    
    def _e_step(self, n_dw: np.ndarray) -> np.ndarray:
        """
        E步：计算后验概率 P(z|d,w)
        
        P(z|d,w) = P(w|z) * P(z|d) / sum_z' P(w|z') * P(z'|d)
        
        参数
        ----
        n_dw : np.ndarray
            词-文档共现矩阵 (n_docs, n_words)
        
        返回
        ----
        P_z_dw : np.ndarray
            后验概率 (n_docs, n_words, n_topics)
        """
        n_docs, n_words = n_dw.shape
        
        # P_z_dw[d, w, z] = P(z|d,w)
        P_z_dw = np.zeros((n_docs, n_words, self.n_topics))
        
        for d in range(n_docs):
            for w in range(n_words):
                if n_dw[d, w] > 0:
                    # 计算分子：P(w|z) * P(z|d) for all z
                    numerator = self.P_w_z[:, w] * self.P_z_d[d, :]  # (n_topics,)
                    
                    # 计算分母：sum_z P(w|z) * P(z|d)
                    denominator = numerator.sum()
                    
                    if denominator > 0:
                        P_z_dw[d, w, :] = numerator / denominator
        
        return P_z_dw
    
    def _m_step(self, n_dw: np.ndarray, P_z_dw: np.ndarray):
        """
        M步：更新参数 P(w|z) 和 P(z|d)
        
        P(w|z) = sum_d n(d,w) * P(z|d,w) / sum_w' sum_d n(d,w') * P(z|d,w')
        
        P(z|d) = sum_w n(d,w) * P(z|d,w) / sum_w n(d,w)
        
        参数
        ----
        n_dw : np.ndarray
            词-文档共现矩阵 (n_docs, n_words)
        P_z_dw : np.ndarray
            后验概率 (n_docs, n_words, n_topics)
        """
        n_docs, n_words = n_dw.shape
        
        # 更新 P(w|z)
        # P_w_z[z, w] = sum_d n(d,w) * P(z|d,w)
        for z in range(self.n_topics):
            numerator = (n_dw * P_z_dw[:, :, z]).sum(axis=0)  # (n_words,)
            denominator = numerator.sum()
            
            if denominator > 0:
                self.P_w_z[z, :] = numerator / denominator
            else:
                # 避免除零，重新初始化
                self.P_w_z[z, :] = 1.0 / n_words
        
        # 更新 P(z|d)
        # P_z_d[d, z] = sum_w n(d,w) * P(z|d,w) / sum_w n(d,w)
        for d in range(n_docs):
            numerator = (n_dw[d, :, np.newaxis] * P_z_dw[d, :, :]).sum(axis=0)  # (n_topics,)
            denominator = n_dw[d, :].sum()
            
            if denominator > 0:
                self.P_z_d[d, :] = numerator / denominator
            else:
                # 空文档，均匀分布
                self.P_z_d[d, :] = 1.0 / self.n_topics
    
    def _compute_loglikelihood(self, n_dw: np.ndarray) -> float:
        """
        计算对数似然
        
        L = sum_d sum_w n(d,w) * log P(w|d)
        其中 P(w|d) = sum_z P(w|z) * P(z|d)
        """
        n_docs, n_words = n_dw.shape
        loglikelihood = 0.0
        
        for d in range(n_docs):
            for w in range(n_words):
                if n_dw[d, w] > 0:
                    # P(w|d) = sum_z P(w|z) * P(z|d)
                    p_w_d = (self.P_w_z[:, w] * self.P_z_d[d, :]).sum()
                    
                    if p_w_d > 0:
                        loglikelihood += n_dw[d, w] * np.log(p_w_d)
        
        return loglikelihood
    
    def _compute_perplexity(self, n_dw: np.ndarray) -> float:
        """
        计算困惑度（Perplexity）
        
        Perplexity = exp(-L / N)
        其中 L 是对数似然，N 是总词数
        
        困惑度越低，模型越好
        """
        loglikelihood = self._compute_loglikelihood(n_dw)
        total_words = n_dw.sum()
        
        if total_words > 0:
            perplexity = np.exp(-loglikelihood / total_words)
        else:
            perplexity = float('inf')
        
        return perplexity
    
    def fit(self, documents: List[str]) -> 'PLSA':
        """
        在文档集合上训练PLSA模型
        
        参数
        ----
        documents : List[str]
            文档列表
            
        返回
        ----
        self : PLSA
        """
        # 构建词汇表
        self.vocabulary_ = self._build_vocabulary(documents)
        self.idx_to_word_ = {idx: word for word, idx in self.vocabulary_.items()}
        
        n_docs = len(documents)
        n_words = len(self.vocabulary_)
        
        if self.verbose:
            print(f"PLSA模型训练开始:")
            print(f"  文档数: {n_docs}")
            print(f"  词汇量: {n_words}")
            print(f"  主题数: {self.n_topics}")
        
        # 构建词-文档矩阵
        n_dw = self._build_word_doc_matrix(documents)
        
        # 初始化参数
        self._initialize_parameters(n_docs, n_words)
        
        # 初始对数似然
        initial_ll = self._compute_loglikelihood(n_dw)
        initial_perplexity = self._compute_perplexity(n_dw)
        self.loglikelihood_history_ = [initial_ll]
        self.perplexity_history_ = [initial_perplexity]
        
        if self.verbose:
            print(f"  初始对数似然: {initial_ll:.2f}")
            print(f"  初始困惑度: {initial_perplexity:.2f}")
        
        # EM迭代
        for iter_num in range(self.max_iter):
            # E步
            P_z_dw = self._e_step(n_dw)
            
            # M步
            self._m_step(n_dw, P_z_dw)
            
            # 计算对数似然
            loglikelihood = self._compute_loglikelihood(n_dw)
            perplexity = self._compute_perplexity(n_dw)
            
            self.loglikelihood_history_.append(loglikelihood)
            self.perplexity_history_.append(perplexity)
            
            # 检查收敛
            ll_change = loglikelihood - self.loglikelihood_history_[-2]
            
            if self.verbose and (iter_num + 1) % 10 == 0:
                print(f"  迭代 {iter_num + 1}/{self.max_iter}: "
                      f"对数似然={loglikelihood:.2f}, "
                      f"困惑度={perplexity:.2f}, "
                      f"变化={ll_change:.4f}")
            
            # 收敛判断
            if abs(ll_change) < self.tol:
                if self.verbose:
                    print(f"  迭代 {iter_num + 1}: 收敛！变化={ll_change:.6f} < tol={self.tol}")
                break
        
        self.n_iter_ = iter_num + 1
        
        if self.verbose:
            print(f"\nPLSA训练完成:")
            print(f"  迭代次数: {self.n_iter_}")
            print(f"  最终对数似然: {self.loglikelihood_history_[-1]:.2f}")
            print(f"  最终困惑度: {self.perplexity_history_[-1]:.2f}")
        
        return self
    
    def transform(self, documents: List[str], max_iter: int = 20) -> np.ndarray:
        """
        对新文档进行主题推断（折叠推断）
        
        固定 P(w|z)，只更新新文档的 P(z|d)
        
        参数
        ----
        documents : List[str]
            新文档列表
        max_iter : int
            推断的最大迭代次数
            
        返回
        ----
        P_z_d : np.ndarray
            新文档的主题分布 (n_new_docs, n_topics)
        """
        if self.P_w_z is None:
            raise ValueError("模型未训练，请先调用fit()")
        
        n_new_docs = len(documents)
        n_words = len(self.vocabulary_)
        
        # 构建新文档的词-文档矩阵
        n_dw_new = np.zeros((n_new_docs, n_words))
        
        for d, doc in enumerate(documents):
            words = self._tokenize(doc)
            word_counts = Counter(words)
            
            for word, count in word_counts.items():
                if word in self.vocabulary_:
                    w = self.vocabulary_[word]
                    n_dw_new[d, w] = count
        
        # 初始化新文档的主题分布
        P_z_d_new = np.random.rand(n_new_docs, self.n_topics)
        P_z_d_new = P_z_d_new / P_z_d_new.sum(axis=1, keepdims=True)
        
        # 折叠推断：固定P(w|z)，只更新P(z|d)
        for _ in range(max_iter):
            # E步
            P_z_dw = np.zeros((n_new_docs, n_words, self.n_topics))
            
            for d in range(n_new_docs):
                for w in range(n_words):
                    if n_dw_new[d, w] > 0:
                        numerator = self.P_w_z[:, w] * P_z_d_new[d, :]
                        denominator = numerator.sum()
                        
                        if denominator > 0:
                            P_z_dw[d, w, :] = numerator / denominator
            
            # M步（只更新P(z|d)）
            for d in range(n_new_docs):
                numerator = (n_dw_new[d, :, np.newaxis] * P_z_dw[d, :, :]).sum(axis=0)
                denominator = n_dw_new[d, :].sum()
                
                if denominator > 0:
                    P_z_d_new[d, :] = numerator / denominator
                else:
                    P_z_d_new[d, :] = 1.0 / self.n_topics
        
        return P_z_d_new
    
    def get_top_words(self, topic_idx: int, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        获取主题的top词
        
        参数
        ----
        topic_idx : int
            主题索引
        top_n : int
            返回前n个词
            
        返回
        ----
        top_words : List[Tuple[str, float]]
            (词, 概率)的列表
        """
        if self.P_w_z is None:
            raise ValueError("模型未训练")
        
        if topic_idx >= self.n_topics:
            raise ValueError(f"主题索引超出范围 [0, {self.n_topics-1}]")
        
        # 获取该主题的词概率
        word_probs = self.P_w_z[topic_idx, :]
        
        # 获取top-n词
        top_indices = np.argsort(word_probs)[::-1][:top_n]
        
        top_words = [(self.idx_to_word_[idx], float(word_probs[idx])) 
                     for idx in top_indices]
        
        return top_words
    
    def get_document_topics(self, doc_idx: int) -> np.ndarray:
        """
        获取文档的主题分布
        
        参数
        ----
        doc_idx : int
            文档索引
            
        返回
        ----
        topic_dist : np.ndarray
            主题分布向量 (n_topics,)
        """
        if self.P_z_d is None:
            raise ValueError("模型未训练")
        
        return self.P_z_d[doc_idx, :]


# ============================================================================
# 示例应用
# ============================================================================

def demo_1_simple_example():
    """
    示例1: 简单的PLSA主题模型
    """
    print("=" * 70)
    print("示例1: 简单的PLSA主题模型")
    print("=" * 70)
    
    # 小型文档集合（3个主题：体育、科技、艺术）
    documents = [
        # 体育主题
        "the team won the game with great players",
        "the player scored in the match",
        "the coach leads the team to victory",
        # 科技主题
        "the computer runs the software program",
        "the algorithm processes the data efficiently",
        "the system executes the code",
        # 艺术主题
        "the artist paints beautiful pictures",
        "the musician plays wonderful music",
        "the gallery displays amazing art",
    ]
    
    print(f"\n文档集合 ({len(documents)}个文档):")
    for i, doc in enumerate(documents):
        print(f"  [{i}] {doc}")
    
    # 训练PLSA模型
    plsa = PLSA(n_topics=3, max_iter=100, tol=1e-4, random_state=42, verbose=True)
    plsa.fit(documents)
    
    # 显示每个主题的top词
    print("\n" + "=" * 70)
    print("主题的关键词:")
    print("=" * 70)
    
    for topic_idx in range(plsa.n_topics):
        print(f"\n主题 {topic_idx}:")
        top_words = plsa.get_top_words(topic_idx, top_n=5)
        for word, prob in top_words:
            print(f"  {word:15s} P(w|z)={prob:.4f}")
    
    # 显示每个文档的主题分布
    print("\n" + "=" * 70)
    print("文档的主题分布:")
    print("=" * 70)
    
    for doc_idx in range(len(documents)):
        topic_dist = plsa.get_document_topics(doc_idx)
        main_topic = np.argmax(topic_dist)
        print(f"\n文档 {doc_idx}: {documents[doc_idx][:40]}...")
        print(f"  主题分布: {topic_dist}")
        print(f"  主要主题: {main_topic} (概率={topic_dist[main_topic]:.4f})")


def demo_2_convergence():
    """
    示例2: 收敛性分析
    """
    print("\n" + "=" * 70)
    print("示例2: PLSA收敛性分析")
    print("=" * 70)
    
    documents = [
        "machine learning algorithms for data analysis",
        "deep learning neural networks",
        "artificial intelligence and machine learning",
        "natural language processing with deep learning",
        "computer vision image recognition",
        "data science and statistical modeling",
        "python programming for data analysis",
        "software development and coding",
    ]
    
    print(f"\n使用 {len(documents)} 个文档训练PLSA")
    
    # 训练模型
    plsa = PLSA(n_topics=2, max_iter=100, tol=1e-5, random_state=42, verbose=False)
    plsa.fit(documents)
    
    print(f"\n训练结果:")
    print(f"  迭代次数: {plsa.n_iter_}")
    print(f"  最终对数似然: {plsa.loglikelihood_history_[-1]:.2f}")
    print(f"  最终困惑度: {plsa.perplexity_history_[-1]:.2f}")
    
    # 可视化收敛曲线
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 对数似然曲线
    axes[0].plot(plsa.loglikelihood_history_, 'b-', linewidth=2)
    axes[0].set_xlabel('迭代次数')
    axes[0].set_ylabel('对数似然')
    axes[0].set_title('PLSA收敛曲线：对数似然')
    axes[0].grid(True, alpha=0.3)
    
    # 困惑度曲线
    axes[1].plot(plsa.perplexity_history_, 'r-', linewidth=2)
    axes[1].set_xlabel('迭代次数')
    axes[1].set_ylabel('困惑度 (Perplexity)')
    axes[1].set_title('PLSA收敛曲线：困惑度')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plsa/plsa_convergence.png', dpi=150, bbox_inches='tight')
    print("\n图片已保存: plsa/plsa_convergence.png")
    plt.close()


def demo_3_topic_inference():
    """
    示例3: 新文档的主题推断
    """
    print("\n" + "=" * 70)
    print("示例3: 新文档的主题推断（折叠推断）")
    print("=" * 70)
    
    # 训练文档
    train_documents = [
        "machine learning algorithms",
        "deep learning neural networks",
        "data science and statistics",
        "computer programming python",
        "software development coding",
        "web development javascript",
    ]
    
    print(f"\n训练集 ({len(train_documents)}个文档):")
    for i, doc in enumerate(train_documents):
        print(f"  [{i}] {doc}")
    
    # 训练模型
    plsa = PLSA(n_topics=2, max_iter=50, random_state=42, verbose=False)
    plsa.fit(train_documents)
    
    print("\n训练完成，主题关键词:")
    for topic_idx in range(plsa.n_topics):
        print(f"\n主题 {topic_idx}:")
        top_words = plsa.get_top_words(topic_idx, top_n=3)
        words = ", ".join([f"{word}({prob:.3f})" for word, prob in top_words])
        print(f"  {words}")
    
    # 新文档
    test_documents = [
        "machine learning with python programming",
        "web development with javascript frameworks",
    ]
    
    print(f"\n测试集（新文档）:")
    for i, doc in enumerate(test_documents):
        print(f"  [{i}] {doc}")
    
    # 推断主题
    test_topics = plsa.transform(test_documents, max_iter=20)
    
    print("\n新文档的主题分布:")
    for i, doc in enumerate(test_documents):
        print(f"\n文档: {doc}")
        print(f"  主题分布: {test_topics[i]}")
        main_topic = np.argmax(test_topics[i])
        print(f"  主要主题: {main_topic} (概率={test_topics[i, main_topic]:.4f})")


def demo_4_different_k():
    """
    示例4: 不同主题数的对比
    """
    print("\n" + "=" * 70)
    print("示例4: 不同主题数的PLSA对比")
    print("=" * 70)
    
    documents = [
        "machine learning and data science",
        "deep learning neural networks",
        "natural language processing",
        "computer vision image recognition",
        "python programming language",
        "javascript web development",
        "database management systems",
        "cloud computing services",
        "mobile app development",
        "cybersecurity and encryption",
    ]
    
    print(f"\n使用 {len(documents)} 个文档")
    print("测试不同的主题数: K = 2, 3, 4, 5")
    
    k_values = [2, 3, 4, 5]
    results = []
    
    for k in k_values:
        plsa = PLSA(n_topics=k, max_iter=50, random_state=42, verbose=False)
        plsa.fit(documents)
        
        final_ll = plsa.loglikelihood_history_[-1]
        final_perplexity = plsa.perplexity_history_[-1]
        
        results.append({
            'k': k,
            'loglikelihood': final_ll,
            'perplexity': final_perplexity,
            'n_iter': plsa.n_iter_
        })
        
        print(f"\nK={k}:")
        print(f"  对数似然: {final_ll:.2f}")
        print(f"  困惑度: {final_perplexity:.2f}")
        print(f"  迭代次数: {plsa.n_iter_}")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    k_list = [r['k'] for r in results]
    ll_list = [r['loglikelihood'] for r in results]
    perp_list = [r['perplexity'] for r in results]
    
    axes[0].plot(k_list, ll_list, 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('主题数 K')
    axes[0].set_ylabel('对数似然')
    axes[0].set_title('主题数 vs 对数似然')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(k_list)
    
    axes[1].plot(k_list, perp_list, 'o-', linewidth=2, markersize=8, color='red')
    axes[1].set_xlabel('主题数 K')
    axes[1].set_ylabel('困惑度')
    axes[1].set_title('主题数 vs 困惑度')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(k_list)
    
    plt.tight_layout()
    plt.savefig('plsa/plsa_k_comparison.png', dpi=150, bbox_inches='tight')
    print("\n图片已保存: plsa/plsa_k_comparison.png")
    plt.close()


def demo_5_visualization():
    """
    示例5: 主题-词分布可视化
    """
    print("\n" + "=" * 70)
    print("示例5: PLSA主题-词分布可视化")
    print("=" * 70)
    
    documents = [
        "sports team player game win match",
        "basketball football soccer team",
        "computer software program code algorithm",
        "technology machine learning data",
        "music artist song concert performance",
        "art painting gallery exhibition",
    ]
    
    print(f"\n训练 PLSA 模型（{len(documents)}个文档，3个主题）")
    
    plsa = PLSA(n_topics=3, max_iter=100, random_state=42, verbose=False)
    plsa.fit(documents)
    
    print("\n主题关键词:")
    topic_labels = []
    for topic_idx in range(plsa.n_topics):
        top_words = plsa.get_top_words(topic_idx, top_n=3)
        words = ", ".join([word for word, _ in top_words])
        topic_labels.append(f"主题{topic_idx}\n({words})")
        print(f"  主题 {topic_idx}: {words}")
    
    # 可视化主题-词分布
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for topic_idx in range(plsa.n_topics):
        top_words = plsa.get_top_words(topic_idx, top_n=8)
        words = [word for word, _ in top_words]
        probs = [prob for _, prob in top_words]
        
        axes[topic_idx].barh(range(len(words)), probs, color=f'C{topic_idx}')
        axes[topic_idx].set_yticks(range(len(words)))
        axes[topic_idx].set_yticklabels(words)
        axes[topic_idx].set_xlabel('P(w|z)')
        axes[topic_idx].set_title(topic_labels[topic_idx])
        axes[topic_idx].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('plsa/plsa_topic_words.png', dpi=150, bbox_inches='tight')
    print("\n图片已保存: plsa/plsa_topic_words.png")
    plt.close()
    
    # 可视化文档-主题分布
    fig, ax = plt.subplots(figsize=(10, 6))
    
    doc_topic_matrix = plsa.P_z_d  # (n_docs, n_topics)
    
    im = ax.imshow(doc_topic_matrix.T, cmap='YlOrRd', aspect='auto')
    ax.set_xlabel('文档')
    ax.set_ylabel('主题')
    ax.set_title('文档-主题分布矩阵 P(z|d)')
    ax.set_xticks(range(len(documents)))
    ax.set_xticklabels([f'D{i}' for i in range(len(documents))])
    ax.set_yticks(range(plsa.n_topics))
    ax.set_yticklabels([f'主题{i}' for i in range(plsa.n_topics)])
    
    # 添加数值标注
    for i in range(plsa.n_topics):
        for j in range(len(documents)):
            text = ax.text(j, i, f'{doc_topic_matrix[j, i]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im, ax=ax, label='P(z|d)')
    plt.tight_layout()
    plt.savefig('plsa/plsa_doc_topics.png', dpi=150, bbox_inches='tight')
    print("图片已保存: plsa/plsa_doc_topics.png")
    plt.close()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("概率潜在语义分析 (Probabilistic Latent Semantic Analysis, PLSA)")
    print("=" * 70)
    
    # 运行所有示例
    demo_1_simple_example()
    demo_2_convergence()
    demo_3_topic_inference()
    demo_4_different_k()
    demo_5_visualization()
    
    print("\n" + "=" * 70)
    print("所有示例运行完成!")
    print("=" * 70)
    print("\n总结:")
    print("1. 简单示例: 展示了PLSA的基本主题建模能力")
    print("2. 收敛分析: 可视化对数似然和困惑度的收敛过程")
    print("3. 主题推断: 演示了对新文档的折叠推断")
    print("4. 主题数对比: 评估不同K值对模型的影响")
    print("5. 可视化: 展示主题-词和文档-主题分布")
    print("\n核心特点:")
    print("- EM算法保证对数似然单调递增")
    print("- 概率解释清晰，易于理解")
    print("- 可以用困惑度评估模型质量")
    print("- 支持新文档的主题推断")
