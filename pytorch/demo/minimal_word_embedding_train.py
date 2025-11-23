import torch
import torch.nn as nn
import torch.optim as optim

sentences = [
    "i love cats", 
    "i love dogs", 
    "you love animals",
    "dogs are cute", 
    "cats are cute"
]

words = set(" ".join(sentences).split())
print(words)
word2id = {w : i for i, w in enumerate(words)}
print(word2id)
id2word = {i : w for w, i in word2id.items()}
print(id2word)
vocab_size = len(words)
print("vocab_size:", vocab_size)


train_x = []
train_y = []

for s in sentences:
    tokens = s.split()
    print(tokens)
    for i in range(len(tokens) - 1):
        train_x.append(word2id[tokens[i]])
        train_y.append(word2id[tokens[i + 1]])

train_x = torch.tensor(train_x)
train_y = torch.tensor(train_y)

print(train_x)
print(train_y)

embed_dim = 8
embedding = nn.Embedding(vocab_size, embed_dim)
predictor = nn.Linear(embed_dim, vocab_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(embedding.parameters()) + list(predictor.parameters()), lr=0.05)

for epoch in range(1000):
    optimizer.zero_grad()

    embedded = embedding(train_x)

    logits = predictor(embedded)

    loss = criterion(logits, train_y)

    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
        print("love 的词向量: ", embedding(torch.tensor(word2id["love"])))
        print("cats 的词向量: ", embedding(torch.tensor(word2id["cats"])))
        print("dogs 的词向量: ", embedding(torch.tensor(word2id["dogs"])))
        print("-" * 40)


def cosine_sim(a, b):
    return torch.cosine_similarity(a, b).item()
w1 = "cats"
w2 = "dogs"

print(f"{w1} 和 {w2} 的余弦相似度 = {cosine_sim(embedding(torch.tensor([word2id[w1]])), embedding(torch.tensor([word2id[w2]]))):.3f}")