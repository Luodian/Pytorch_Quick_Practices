import torch as t

x = t.rand(5,3)

print (x)

print (x.size())

print (x.size(0),x.size(1))

y = t.rand(5,3)

print (y)

result = t.Tensor(5,3)

result = t.add(x,y)

print (result)

a = t.ones(5)
# create a tensor has size of 5

b = a.numpy()
# tensor -> numpy

# Tensor可通过.cuda 方法转为GPU的Tensor，从而享受GPU带来的加速运算。

if t.cuda.is_available():
	x = x.cuda()
	y = y.cuda()
	x + y
	
# 深度学习的算法本质上是通过反向传播求导数，而PyTorch的Autograd模块则实现了此功能。在Tensor上的所有操作，Autograd都能为它们自动提供微分，避免了手动计算导数的复杂过程。
#
# autograd.Variable是Autograd中的核心类，它简单封装了Tensor，并支持几乎所有Tensor有的操作。Tensor在被封装为Variable之后，可以调用它的.backward实现反向传播，自动计算所有梯度。Variable的数据结构如图2-6所示。

