import torch

a = torch.randn(1, 4, 4)
b = torch.randn(1)

print(a)
print(b)
print(torch.div(a, b).shape)
