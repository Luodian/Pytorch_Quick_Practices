from __future__ import print_function
import torch as t
from torch.autograd import Variable as V

# 从tensor中创建variable，指定需要求导
a = V(t.ones(3,4), requires_grad = True)

b = V(t.zeros(3,4))

c = a.add(b)

d = c.sum()

d.backward()

print (c.data.sum(),c.sum())

print (a.grad)