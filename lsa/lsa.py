"""
潜在语义分析 (Latent Semantic Analysis, LSA)
============================================

LSA是一种用于分析文本语料库和概念间关系的自然语言处理技术。
它使用奇异值分解(SVD)来降低词-文档矩阵的维度，发现词汇和文档之间的潜在语义结构。

理论基础
--------

1. **词-文档矩阵 (Term-Document Matrix)**:
   - 行：词汇表中的词
   - 列：文档集合中的文档
   - 元素 A[i,j]：词i在文档j中的权重（通常使用TF-IDF）

2. **TF-IDF权重**:
   - TF (Term Frequency): 词频，词在文档中出现的频率
     $$\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}$$
   
   - IDF (Inverse Document Frequency): 逆文档频率
     $$\text{IDF}(t) = \log \frac{N}{|\{d: t \in d\}|}$$
   
   - TF-IDF = TF × IDF

3. **奇异值分解 (SVD)**:
   对词-文档矩阵A进行分解：
   $$A_{m \times n} = U_{m \times k} \Sigma_{k \times k} V^T_{k \times n}$$
   
   其中：
   - U: 词的k维语义表示（词-主题矩阵）
   - Σ: 对角矩阵，包含k个奇异值（主题的重要性）
   - V^T: 文档的k维语义表示（文档-主题矩阵）
   - k: 保留的主题数（通常k << min(m, n)）

4. **降维和语义空间**:
   - 只保留前k个最大的奇异值及对应的向量
   - 新的k维空间捕获了主要的语义信息
   - 相似的词和文档在这个空间中距离更近

5. **查询投影**:
   对于新查询q，投影到k维语义空间：
   $$q_k = q^T U_k \Sigma_k^{-1}$$
   
   然后计算与文档的相似度（通常使用余弦相似度）

算法步骤
--------

1. 构建词-文档矩阵（TF-IDF加权）
2. 对矩阵进行SVD分解
3. 保留前k个奇异值和对应向量
4. 将文档投影到k维语义空间
5. 查询时，将查询投影到同一空间并计算相似度

优势
----
- 发现词汇间的潜在语义关系
- 处理一词多义和多词一义问题
- 降维减少噪声，提高计算效率
- 可用于信息检索、文档聚类、文本分类等

日期: 2025-01-13
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import re
from collections import Counter
import warnings

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class LSA:
    """
    潜在语义分析 (Latent Semantic Analysis)
    
    使用TF-IDF和SVD进行文本语义分析
    
    参数
    ----
    n_components : int
        保留的主题/语义维度数量
    min_df : int
        词汇必须出现的最小文档数
    max_df : float
        词汇可以出现的最大文档比例
    use_idf : bool
        是否使用IDF权重
    sublinear_tf : bool
        是否使用对数TF：log(1 + tf)
    """
    
    def __init__(
        self,
        n_components: int = 100,
        min_df: int = 1,
        max_df: float = 1.0,
        use_idf: bool = True,
        sublinear_tf: bool = True
    ):
        self.n_components = n_components
        self.min_df = min_df
        self.max_df = max_df
        self.use_idf = use_idf
        self.sublinear_tf = sublinear_tf
        
        # 将在fit时设置
        self.vocabulary_ = {}  # 词 -> 索引
        self.idf_ = None
        self.U_ = None  # 词-主题矩阵
        self.Sigma_ = None  # 奇异值
        self.Vt_ = None  # 文档-主题矩阵的转置
        self.doc_embeddings_ = None  # 文档在语义空间的表示
        self.documents_ = []
        
    def _tokenize(self, text: str) -> List[str]:
        """简单的分词器"""
        # 转小写，提取单词
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def _build_vocabulary(self, documents: List[str]) -> Dict[str, int]:
        """构建词汇表"""
        # 统计词频
        doc_freq = Counter()
        for doc in documents:
            words = set(self._tokenize(doc))
            doc_freq.update(words)
        
        # 过滤词汇
        n_docs = len(documents)
        max_doc_count = int(self.max_df * n_docs)
        
        vocab = {}
        idx = 0
        for word, freq in sorted(doc_freq.items()):
            if self.min_df <= freq <= max_doc_count:
                vocab[word] = idx
                idx += 1
        
        return vocab
    
    def _compute_tf(self, documents: List[str]) -> np.ndarray:
        """计算TF矩阵"""
        n_docs = len(documents)
        n_terms = len(self.vocabulary_)
        
        tf_matrix = np.zeros((n_terms, n_docs))
        
        for j, doc in enumerate(documents):
            words = self._tokenize(doc)
            word_counts = Counter(words)
            doc_length = len(words)
            
            if doc_length == 0:
                continue
            
            for word, count in word_counts.items():
                if word in self.vocabulary_:
                    i = self.vocabulary_[word]
                    # 计算TF
                    if self.sublinear_tf:
                        tf = 1 + np.log(count)
                    else:
                        tf = count / doc_length
                    tf_matrix[i, j] = tf
        
        return tf_matrix
    
    def _compute_idf(self, documents: List[str]) -> np.ndarray:
        """计算IDF向量"""
        n_docs = len(documents)
        n_terms = len(self.vocabulary_)
        
        df = np.zeros(n_terms)
        
        for doc in documents:
            words = set(self._tokenize(doc))
            for word in words:
                if word in self.vocabulary_:
                    i = self.vocabulary_[word]
                    df[i] += 1
        
        # IDF: log((N + 1) / (df + 1)) + 1
        # 加1平滑，避免除零
        idf = np.log((n_docs + 1) / (df + 1)) + 1
        
        return idf
    
    def _build_tfidf_matrix(self, documents: List[str]) -> np.ndarray:
        """构建TF-IDF矩阵"""
        # 计算TF
        tf_matrix = self._compute_tf(documents)
        
        # 计算IDF
        if self.use_idf:
            self.idf_ = self._compute_idf(documents)
            # TF-IDF = TF * IDF
            tfidf_matrix = tf_matrix * self.idf_[:, np.newaxis]
        else:
            tfidf_matrix = tf_matrix
        
        return tfidf_matrix
    
    def fit(self, documents: List[str]) -> 'LSA':
        """
        在文档集合上训练LSA模型
        
        参数
        ----
        documents : List[str]
            文档列表
            
        返回
        ----
        self : LSA
        """
        self.documents_ = documents
        
        # 构建词汇表
        self.vocabulary_ = self._build_vocabulary(documents)
        
        if len(self.vocabulary_) == 0:
            raise ValueError("词汇表为空，请检查文档内容和参数设置")
        
        # 构建TF-IDF矩阵
        tfidf_matrix = self._build_tfidf_matrix(documents)
        
        # SVD分解
        k = min(self.n_components, tfidf_matrix.shape[0], tfidf_matrix.shape[1])
        
        # 使用numpy的SVD
        U, Sigma, Vt = np.linalg.svd(tfidf_matrix, full_matrices=False)
        
        # 保留前k个成分
        self.U_ = U[:, :k]
        self.Sigma_ = Sigma[:k]
        self.Vt_ = Vt[:k, :]
        
        # 文档在语义空间的表示: V * Sigma
        self.doc_embeddings_ = self.Vt_.T * self.Sigma_
        
        print(f"LSA模型训练完成:")
        print(f"  - 文档数量: {len(documents)}")
        print(f"  - 词汇量: {len(self.vocabulary_)}")
        print(f"  - 语义维度: {k}")
        print(f"  - 保留的方差比例: {np.sum(self.Sigma_**2) / np.sum(Sigma**2):.2%}")
        
        return self
    
    def transform_query(self, query: str) -> np.ndarray:
        """
        将查询转换到语义空间
        
        参数
        ----
        query : str
            查询文本
            
        返回
        ----
        query_embedding : np.ndarray
            查询在语义空间的表示
        """
        if self.U_ is None:
            raise ValueError("模型未训练，请先调用fit()方法")
        
        # 将查询转换为TF-IDF向量
        words = self._tokenize(query)
        word_counts = Counter(words)
        
        query_vector = np.zeros(len(self.vocabulary_))
        
        for word, count in word_counts.items():
            if word in self.vocabulary_:
                i = self.vocabulary_[word]
                if self.sublinear_tf:
                    tf = 1 + np.log(count)
                else:
                    tf = count / len(words) if len(words) > 0 else 0
                
                if self.use_idf:
                    query_vector[i] = tf * self.idf_[i]
                else:
                    query_vector[i] = tf
        
        # 投影到语义空间: q_k = q^T * U_k * Sigma_k^-1
        query_embedding = (query_vector @ self.U_) / (self.Sigma_ + 1e-10)
        
        return query_embedding
    
    def similarity(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        找到与查询最相似的文档
        
        参数
        ----
        query : str
            查询文本
        top_k : int
            返回前k个最相似的文档
            
        返回
        ----
        results : List[Tuple[int, float]]
            (文档索引, 相似度分数)的列表，按相似度降序排列
        """
        # 获取查询的语义表示
        query_embedding = self.transform_query(query)
        
        # 计算余弦相似度
        # cosine_sim = (q · d) / (||q|| * ||d||)
        query_norm = np.linalg.norm(query_embedding)
        doc_norms = np.linalg.norm(self.doc_embeddings_, axis=1)
        
        if query_norm == 0:
            return []
        
        similarities = (self.doc_embeddings_ @ query_embedding) / (doc_norms * query_norm + 1e-10)
        
        # 获取top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [(int(idx), float(similarities[idx])) for idx in top_indices]
        
        return results
    
    def get_topic_terms(self, topic_idx: int, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        获取主题的关键词
        
        参数
        ----
        topic_idx : int
            主题索引
        top_n : int
            返回前n个关键词
            
        返回
        ----
        terms : List[Tuple[str, float]]
            (词, 权重)的列表
        """
        if self.U_ is None:
            raise ValueError("模型未训练")
        
        if topic_idx >= self.U_.shape[1]:
            raise ValueError(f"主题索引超出范围 [0, {self.U_.shape[1]-1}]")
        
        # 获取该主题的词权重
        topic_weights = self.U_[:, topic_idx]
        
        # 创建词 -> 权重的映射
        idx_to_word = {idx: word for word, idx in self.vocabulary_.items()}
        
        # 获取top-n词（按绝对值排序）
        top_indices = np.argsort(np.abs(topic_weights))[::-1][:top_n]
        
        terms = [(idx_to_word[int(idx)], float(topic_weights[idx])) for idx in top_indices]
        
        return terms
    
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
            主题分布向量
        """
        if self.doc_embeddings_ is None:
            raise ValueError("模型未训练")
        
        return self.doc_embeddings_[doc_idx]


# ============================================================================
# 示例应用
# ============================================================================

def demo_1_simple_example():
    """
    示例1: 简单的文档检索
    """
    print("=" * 70)
    print("示例1: 简单的文档检索示例")
    print("=" * 70)
    
    # 小型文档集合
    documents = [
        "The cat sat on the mat.",
        "The dog sat on the log.",
        "Cats and dogs are common pets.",
        "I love my pet cat.",
        "Dogs are loyal animals.",
        "The cat is sleeping on the bed.",
        "My dog likes to play in the park.",
        "Pets need love and care.",
    ]
    
    print(f"\n文档集合 ({len(documents)}个文档):")
    for i, doc in enumerate(documents):
        print(f"  [{i}] {doc}")
    
    # 训练LSA模型
    lsa = LSA(n_components=3, min_df=1, sublinear_tf=True)
    lsa.fit(documents)
    
    # 测试查询
    queries = [
        "cat sleeping",
        "dog playing",
        "pet care",
    ]
    
    print("\n" + "=" * 70)
    print("查询结果:")
    print("=" * 70)
    
    for query in queries:
        print(f"\n查询: '{query}'")
        results = lsa.similarity(query, top_k=3)
        
        for rank, (doc_idx, score) in enumerate(results, 1):
            print(f"  {rank}. [文档{doc_idx}] 相似度={score:.4f}: {documents[doc_idx]}")


def demo_2_topic_analysis():
    """
    示例2: 主题分析
    """
    print("\n" + "=" * 70)
    print("示例2: 主题分析")
    print("=" * 70)
    
    documents = [
        "machine learning algorithms are used for data analysis",
        "deep learning neural networks can recognize patterns",
        "natural language processing enables text understanding",
        "computer vision helps machines see and interpret images",
        "reinforcement learning agents learn from environment interaction",
        "data science combines statistics and programming",
        "artificial intelligence includes machine learning and expert systems",
        "big data requires distributed computing frameworks",
    ]
    
    print(f"\n文档集合 ({len(documents)}个文档):")
    for i, doc in enumerate(documents):
        print(f"  [{i}] {doc}")
    
    # 训练LSA模型
    lsa = LSA(n_components=3, min_df=1, sublinear_tf=True)
    lsa.fit(documents)
    
    # 显示主题的关键词
    print("\n" + "=" * 70)
    print("主题的关键词:")
    print("=" * 70)
    
    for topic_idx in range(min(3, lsa.U_.shape[1])):
        print(f"\n主题 {topic_idx}:")
        terms = lsa.get_topic_terms(topic_idx, top_n=5)
        for term, weight in terms:
            print(f"  {term:20s} {weight:8.4f}")
    
    # 可视化文档在主题空间的分布
    if lsa.doc_embeddings_.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        embeddings_2d = lsa.doc_embeddings_[:, :2]
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=100, alpha=0.6)
        
        for i, (x, y) in enumerate(embeddings_2d):
            ax.annotate(f'Doc{i}', (x, y), xytext=(5, 5), 
                       textcoords='offset points', fontsize=9)
        
        ax.set_xlabel('主题维度 1')
        ax.set_ylabel('主题维度 2')
        ax.set_title('文档在2维主题空间的分布')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('lsa/lsa_topic_space.png', dpi=150, bbox_inches='tight')
        print("\n图片已保存: lsa/lsa_topic_space.png")
        plt.close()


def demo_3_dimensionality_reduction():
    """
    示例3: 降维效果分析
    """
    print("\n" + "=" * 70)
    print("示例3: 降维效果分析")
    print("=" * 70)
    
    # 创建一个更大的文档集
    documents = [
        "Python is a popular programming language for data science",
        "Java is widely used in enterprise applications",
        "JavaScript runs in web browsers and servers",
        "C++ offers high performance for system programming",
        "Machine learning requires understanding of algorithms and statistics",
        "Deep learning uses neural networks with many layers",
        "Natural language processing deals with text and speech",
        "Computer vision enables image and video analysis",
        "Data analysis involves exploring and visualizing datasets",
        "Statistical modeling helps make predictions from data",
        "Web development includes frontend and backend technologies",
        "Mobile apps are built for iOS and Android platforms",
        "Cloud computing provides scalable infrastructure",
        "Database systems store and retrieve information efficiently",
        "Software engineering practices improve code quality",
    ]
    
    print(f"\n使用 {len(documents)} 个文档进行降维分析")
    
    # 测试不同的主题数
    n_components_list = [2, 3, 5, 8, 10]
    variance_ratios = []
    
    for n_comp in n_components_list:
        lsa = LSA(n_components=n_comp, min_df=1, sublinear_tf=True)
        lsa.fit(documents)
        
        # 计算保留的方差比例
        # 通过重新进行完整SVD来计算
        tfidf = lsa._build_tfidf_matrix(documents)
        _, full_sigma, _ = np.linalg.svd(tfidf, full_matrices=False)
        
        variance_ratio = np.sum(lsa.Sigma_**2) / np.sum(full_sigma**2)
        variance_ratios.append(variance_ratio)
        
        print(f"  n_components={n_comp:2d}: 保留方差比例={variance_ratio:.2%}")
    
    # 可视化
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(n_components_list, variance_ratios, 'o-', linewidth=2, markersize=8)
    ax.axhline(y=0.9, color='r', linestyle='--', label='90% 方差阈值')
    ax.set_xlabel('主题数量 (n_components)')
    ax.set_ylabel('保留的方差比例')
    ax.set_title('LSA降维: 主题数量 vs 保留的方差比例')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('lsa/lsa_variance_ratio.png', dpi=150, bbox_inches='tight')
    print("\n图片已保存: lsa/lsa_variance_ratio.png")
    plt.close()


def demo_4_semantic_similarity():
    """
    示例4: 语义相似度发现
    """
    print("\n" + "=" * 70)
    print("示例4: 语义相似度发现（同义词和相关概念）")
    print("=" * 70)
    
    documents = [
        "The car is fast and efficient.",
        "This automobile has great speed.",
        "The vehicle runs very quickly.",
        "A dog is a loyal pet.",
        "Canines are faithful companions.",
        "The book is interesting and informative.",
        "This novel is engaging and educational.",
        "The film was entertaining and fun.",
        "The movie was enjoyable to watch.",
    ]
    
    print(f"\n文档集合 ({len(documents)}个文档):")
    for i, doc in enumerate(documents):
        print(f"  [{i}] {doc}")
    
    # 训练LSA模型
    lsa = LSA(n_components=4, min_df=1, sublinear_tf=True)
    lsa.fit(documents)
    
    # 测试语义相似的查询
    print("\n" + "=" * 70)
    print("语义相似度测试:")
    print("=" * 70)
    
    test_cases = [
        ("fast car", "应该匹配关于汽车速度的文档"),
        ("loyal dog", "应该匹配关于狗的文档"),
        ("good book", "应该匹配关于书籍的文档"),
        ("fun movie", "应该匹配关于电影的文档"),
    ]
    
    for query, description in test_cases:
        print(f"\n查询: '{query}' ({description})")
        results = lsa.similarity(query, top_k=3)
        
        for rank, (doc_idx, score) in enumerate(results, 1):
            print(f"  {rank}. [文档{doc_idx}] 相似度={score:.4f}: {documents[doc_idx]}")


def demo_5_comparison():
    """
    示例5: LSA与关键词匹配的对比
    """
    print("\n" + "=" * 70)
    print("示例5: LSA与简单关键词匹配的对比")
    print("=" * 70)
    
    documents = [
        "Python programming language",
        "Java programming language",
        "Machine learning with Python",
        "Deep learning algorithms",
        "Natural language processing",
        "Computer vision applications",
        "Data science and analytics",
        "Web development with JavaScript",
    ]
    
    print(f"\n文档集合 ({len(documents)}个文档):")
    for i, doc in enumerate(documents):
        print(f"  [{i}] {doc}")
    
    # 训练LSA
    lsa = LSA(n_components=3, min_df=1, sublinear_tf=True)
    lsa.fit(documents)
    
    # 测试查询
    query = "coding in Python"
    
    print(f"\n查询: '{query}'")
    print("\n" + "-" * 70)
    print("LSA结果 (考虑语义相似度):")
    print("-" * 70)
    
    lsa_results = lsa.similarity(query, top_k=5)
    for rank, (doc_idx, score) in enumerate(lsa_results, 1):
        print(f"  {rank}. [文档{doc_idx}] 相似度={score:.4f}: {documents[doc_idx]}")
    
    # 简单关键词匹配
    print("\n" + "-" * 70)
    print("关键词匹配结果 (只看词的直接匹配):")
    print("-" * 70)
    
    query_words = set(lsa._tokenize(query))
    keyword_scores = []
    
    for idx, doc in enumerate(documents):
        doc_words = set(lsa._tokenize(doc))
        # Jaccard相似度
        intersection = len(query_words & doc_words)
        union = len(query_words | doc_words)
        score = intersection / union if union > 0 else 0
        keyword_scores.append((idx, score))
    
    keyword_scores.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (doc_idx, score) in enumerate(keyword_scores[:5], 1):
        print(f"  {rank}. [文档{doc_idx}] Jaccard={score:.4f}: {documents[doc_idx]}")
    
    print("\n说明: LSA能够发现'coding'和'programming'、'Python'之间的语义关系，")
    print("      而简单的关键词匹配只能找到包含完全相同词的文档。")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("潜在语义分析 (Latent Semantic Analysis, LSA)")
    print("=" * 70)
    
    # 运行所有示例
    demo_1_simple_example()
    demo_2_topic_analysis()
    demo_3_dimensionality_reduction()
    demo_4_semantic_similarity()
    demo_5_comparison()
    
    print("\n" + "=" * 70)
    print("所有示例运行完成!")
    print("=" * 70)
    print("\n总结:")
    print("1. 简单检索: 展示了基本的文档相似度搜索")
    print("2. 主题分析: 提取和可视化主题结构")
    print("3. 降维分析: 评估不同主题数的效果")
    print("4. 语义相似: 发现同义词和相关概念")
    print("5. 方法对比: LSA vs 关键词匹配")
    print("\n关键优势:")
    print("- 发现潜在语义关系（同义词、相关概念）")
    print("- 降维减少噪声，提高效率")
    print("- 适用于信息检索、文档聚类、推荐系统等")
