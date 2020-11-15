from __future__ import print_function
import torch

x = torch.rand(5,3)
print(x)

y = torch.ones(5,3)
print(y)

result = (x+y)

print(result)
print(result[:,1])

z = x.view(15)

print(z)
