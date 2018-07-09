import torch as t
from matplotlib import pyplot as plt
from torch.autograd import Variable as V

t.manual_seed ( 1000 )


def get_fake_data ( batch_size = 20 ) :
	x = t.rand ( batch_size , 1 ) * 20
	# 产生随机数据
	y = x * 2 + (1 + t.randn ( batch_size , 1 )) * 3
	return x , y


x , y = get_fake_data ( )

x = V ( x )
y = V ( y )



w = V ( t.rand ( 1 , 1 ) , requires_grad = True )
b = V ( t.zeros ( 1 , 1 ) , requires_grad = True )

plt.scatter ( x.squeeze ( ).numpy ( ) , y.squeeze ( ).numpy ( ) )
plt.plot ( x.numpy ( ) , (x.mm ( w.data ) + b.data.expand_as ( x )).numpy ( ) )
plt.show ( )

print ( x )

print ( y )

lr = 0.0001

for ii in range ( 8000 ) :
	# forward: 计算loss
	y_pred = x.mm ( w ) + b.expand_as ( y )
	
	loss = 0.5 * (y_pred - y) ** 2
	loss = loss.sum ( )
	
	loss.backward ( )
	
	w.data.sub_ ( lr * w.grad.data )
	
	b.data.sub_ ( lr * b.grad.data )
	
	w.grad.data.zero_ ( )
	
	b.grad.data.zero_ ( )

print ( w.data.squeeze ( ) [ 0 ] , b.data.squeeze ( ) [ 0 ] )


plt.scatter ( x.squeeze ( ).numpy ( ) , y.squeeze ( ).numpy ( ) )
plt.plot ( x.numpy ( ) , (x.mm ( w.data ) + b.data.expand_as ( x )).numpy ( ) )
plt.show ( )

# 下面的这段代码里

# # 随机初始化参数
# w = V ( t.rand ( 1 , 1 ) , requires_grad = True )
# b = V ( t.zeros ( 1 , 1 ) , requires_grad = True )
#
# lr = 0.001  # 学习率
#
# for ii in range ( 8000 ) :
# 	x , y = get_fake_data ( )
# 	x , y = V ( x ) , V ( y )
#
# 	# forward：计算loss
# 	y_pred = x.mm ( w ) + b.expand_as ( y )
# 	loss = 0.5 * (y_pred - y) ** 2
# 	loss = loss.sum ( )
#
# 	# backward：手动计算梯度
# 	loss.backward ( )
#
# 	# 更新参数
# 	w.data.sub_ ( lr * w.grad.data )
# 	b.data.sub_ ( lr * b.grad.data )
#
# 	# 梯度清零
# 	w.grad.data.zero_ ( )
# 	b.grad.data.zero_ ( )
#
# 	if ii % 1000 == 0 :
# 		# 画图
# 		display.clear_output ( wait = True )
# 		x = t.arange ( 0 , 20 ).view ( -1 , 1 )
# 		y = x.mm ( w.data ) + b.data.expand_as ( x )
# 		plt.plot ( x.numpy ( ) , y.numpy ( ) )  # predicted
#
# 		x2 , y2 = get_fake_data ( batch_size = 20 )
# 		plt.scatter ( x2.numpy ( ) , y2.numpy ( ) )  # true data
#
# 		plt.xlim ( 0 , 20 )
# 		plt.ylim ( 0 , 41 )
# 		plt.show ( )
# 		plt.pause ( 0.5 )
#
# print ( w.data.squeeze ( ) [ 0 ] , b.data.squeeze ( ) [ 0 ] )