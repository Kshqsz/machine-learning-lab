import torch

list_of_lists = [
    [1, 2, 3],
    [4, 5, 6],
]

print(list_of_lists)

data = torch.tensor(list_of_lists)
print(data)