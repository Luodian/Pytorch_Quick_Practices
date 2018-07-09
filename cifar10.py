import torch as t
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage

show = ToPILImage ( )

transform = transforms.Compose (
	[ transforms.ToTensor ( ) , transforms.Normalize ( (0.5 , 0.5 , 0.5) , (0.5 , 0.5 , 0.5) ) ] )

trainset = tv.datasets.CIFAR10(root = './train/', train = True, download = True, transform = transform)

tranloader = t.utils.data.DataLoader(trainset,batch_size = 4,shuffle = True,num_workers = 2)

testset = tv.datasets.CIFAR10('./test/',train = False, download = True, transform = transform)

testloader = t.utils.data.DataLoader(testset,batch_size = 4, shuffle = False, num_workers = 2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

(data, label) = trainset[100]
print(classes[label])

# (data + 1) / 2是为了还原被归一化的数据
I = show((data + 1) / 2).resize((100, 100))

# I.show()




