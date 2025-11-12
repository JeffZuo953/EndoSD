import torch
import time

# 定义要占用的GPU列表
gpus = [0, 1]

# 设置每个GPU占用约5000MB内存 
memory_per_gpu = 5000  # MB

# 占用内存的大小（以字节为单位）
bytes_per_gpu = memory_per_gpu * 1024 * 1024  # 转换为字节

# 持续占用每个GPU的内存
tensors = []
for gpu in gpus:
    tensor = torch.empty((bytes_per_gpu // 4,), device=f'cuda:{gpu}')  # 每个float占4字节
    tensors.append(tensor)


while True:
    time.sleep(3600)
