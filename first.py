# CUDA TEST
from __future__ import print_function
import torch

# x = torch.empty(5,3)
# print(x)
x = torch.Tensor([1.0])
xx = x.cuda()
print(xx)

# CUDNN TEST
from torch.backends import cudnn
print(cudnn.is_acceptable(xx))