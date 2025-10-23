import torch

# -------------------------------
# 1. 什么是张量（Tensor）
# -------------------------------
# 张量是 PyTorch 中最基础的数据结构，可以看作是一个多维数组。
# 例如：
# 0 维张量：标量（scalar）
# 1 维张量：向量（vector）
# 2 维张量：矩阵（matrix）
# n 维张量：n 阶张量

# 创建一个标量
scalar = torch.tensor(5)
print("标量（0D Tensor）:", scalar)
print("维度:", scalar.ndim)
print("形状:", scalar.shape, "\n")

# 创建一个向量
vector = torch.tensor([1, 2, 3])
print("向量（1D Tensor）:", vector)
print("维度:", vector.ndim)
print("形状:", vector.shape, "\n")

# 创建一个矩阵
matrix = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])
print("矩阵（2D Tensor）:\n", matrix)
print("维度:", matrix.ndim)
print("形状:", matrix.shape, "\n")

# 创建一个三维张量
tensor3d = torch.tensor([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])
print("三维张量（3D Tensor）:\n", tensor3d)
print("维度:", tensor3d.ndim)
print("形状:", tensor3d.shape, "\n")

# -------------------------------
# 2. 张量的基本操作
# -------------------------------
print("张量加法:", vector + 10)
print("张量乘法:", vector * 2)
print("矩阵相加:\n", matrix + torch.ones_like(matrix), "\n")

# -------------------------------
# 3. 张量的类型与设备（CPU/GPU）
# -------------------------------
print("默认数据类型:", scalar.dtype)
print("是否在GPU上:", scalar.device, "\n")

# 如果有GPU可用，可以把张量放到GPU
if torch.cuda.is_available():
    gpu_tensor = matrix.to("cuda")
    print("张量已移动到GPU上:", gpu_tensor.device)
else:
    print("当前未检测到GPU设备。")

# -------------------------------
# 4. 张量的用途
# -------------------------------
# 张量是深度学习中所有计算的基础，用于存储输入数据、权重、梯度等。
# 例如，神经网络的输入、卷积核、权重参数等本质上都是张量。
