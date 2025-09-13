import torch
print(torch.cuda.is_available())  # 应输出 True
print(torch.cuda.device_count())  # 应输出 1
print(torch.cuda.get_device_name(0))  # 应输出 NVIDIA GeForce RTX 4060 Laptop GPU
print(torch.version.cuda)  # 应输出 11.8
x = torch.randn(1000, 1000, device='cuda')
y = x @ x
print(y.sum())  # 如果正常运行，说明 CUDA 没问题