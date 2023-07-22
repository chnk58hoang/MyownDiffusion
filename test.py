import torch

x = torch.rand((3, 5, 5))
eps = torch.randn_like(x)

alpha = torch.rand(10)
t = (torch.ones(2) * 5).int()

print(alpha[t][:, None, None])
