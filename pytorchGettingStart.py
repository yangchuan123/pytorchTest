from __future__ import print_function
import torch
import numpy as np

def test():
    # x = torch.empty(5, 3)
    # print(x)
    #
    # x = torch.rand(5, 3)
    # print(x)
    #
    # x = torch.zeros(5, 3, dtype=torch.long)
    # print(x)
    #
    # x = torch.tensor([5.5, 3])
    # print(x)
    # x = x.new_ones(5, 3, dtype=torch.double)  # new_* methods take in sizes
    # print(x)
    #
    # x = torch.randn_like(x, dtype=torch.float)  # override dtype!
    # print(x)
    #
    # print(x.size())
    #
    # y = torch.rand(5, 3)
    # print(x + y)
    # print(torch.add(x, y))
    #
    # result = torch.empty(5, 3)
    # torch.add(x, y, out=result)
    # print(result)
    #
    # y.add_(x)
    # print(y)
    #
    # print(x[:, 1])
    #
    # x = torch.randn(1)
    # print(x)
    # print(x.item())
    # a = torch.ones(5)
    # print(a)
    # b = a.numpy()
    # print(b)
    # a.add_(1)
    # print(a)
    # print(b)
    # a = np.ones(5)
    # b = torch.from_numpy(a)
    # np.add(a, 1, out=a)
    # print(a)
    # print(b)
    x = torch.randn(1)
    if torch.cuda.is_available():
        device = torch.device("cuda")  # a CUDA device object
        y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
        x = x.to(device)  # or just use strings ``.to("cuda")``
        z = x + y
        print(z)
        print(z.to("cpu", torch.double))  # ``.to`` can also change dtype together!
    return


test()