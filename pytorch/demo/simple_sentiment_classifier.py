import torch
import torch.nn as nn
import torch.optim as optim

sentences = [
    "i love this movie",
    "this is amazing",
    "i really like this film",
    "the story is great",

    "i hate this movie",
    "this is bad",
    "i really dislike this film",
    "the story is terrible",
]

labels = [
    1, 1, 1, 1,   
    0, 0, 0, 0, 
]

words = set(" ".join(sentences).split())
print(words)
word2id = {w : i for i, w in enumerate(words)}
vocab_size = len(words)
print(word2id)
print("vocab_size: ", vocab_size)

def encode_sentence(sentence):
    return [word2id[w] for w in sentence.split()]
encoded_sentences = [encode_sentence(s) for s in sentences]

print(encoded_sentences)

embed_dim = 8
hidden_dim = 16
num_classes = 2

class SimpleSentimentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        emb = self.embedding(x)
        avg_emb = emb.mean(dim = 0)
        h = self.relu(self.fc1(avg_emb))
        logits = self.fc2(h)
        return logits

model = SimpleSentimentNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

for epoch in range(300):
    total_loss = 0
    for i, sent in enumerate(encoded_sentences):
        x = torch.tensor(sent)
        y = torch.tensor(labels[i])

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.unsqueeze(0), y.unsqueeze(0))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    if epoch % 50 == 0:
        print(f"Epoch {epoch} Loss = {total_loss:.4f}")

test = "i love this"
x = torch.tensor(encode_sentence(test))
logits = model(x)
pred = logits.argmax().item()
print("预测结果: ", "正面" if pred == 1 else "负面")