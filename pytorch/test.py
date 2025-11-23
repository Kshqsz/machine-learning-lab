import torch
import torch.nn as nn

vocab_size = 5000
embed_dim = 128

embedding = nn.Embedding(vocab_size, embed_dim)

rnn = nn.RNN(input_size=embed_dim, hidden_size=256, batch_first=True)

x = torch.tensor([[10, 200, 301, 8, 5]])

embed = embedding(x)

output, h_n = rnn(embed)


print("\nRNN 每个时间步的输出 (output) shape:", output.shape)
print("RNN 最后一个时间步的隐藏状态 (h_n) shape:", h_n.shape)

print("\nRNN 输出数据：\n", output)
print("\n最后一个隐藏层状态：\n", h_n)



