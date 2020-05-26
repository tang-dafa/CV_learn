# Task3 字符识别模型

卷积神经网络（CNN）

## 1.目标

- CNN基础和原理
- 使用pytorch框架构建CNN模型，完成训练

## 2.CNN理解

![image-20200520162331405](C:\Users\dafa\AppData\Roaming\Typora\typora-user-images\image-20200520162331405.png)



**卷积神经网络**（简称CNN）是一类特殊的人工神经网络，是深度学习中重要的一个分支。CNN在很多领域都表现优异，精度和速度比传统计算学习算法高很多。特别是在计算机视觉领域，CNN是解决图像分类、图像检索、物体检测和语义分割的主流模型。 

CNN是一种层次模型，输入的是原始的像素数据。**CNN**通过**卷积（convolution）**、**00池化（pooling）**、**非线性激活函数（non-linear activation function）**和**全连接层（fully connected layer）**构成。



- **局部连接** 这个是最容易想到的，每个神经元不再和上一层的所有神经元相连，而只和一小部分神经元相连。这样就减少了很多参数。

- **权值共享** 一组连接可以共享同一个权重，而不是每个连接有一个不同的权重，这样又减少了很多参数。

- **下采样** 可以使用Pooling来减少每层的样本数，进一步减少参数数量，同时还可以提升模型的鲁棒性。

  

### 2.1 网络架构

常用架构模式**为：
$$
INPUT -> [[CONV]*N -> POOL?]*M -> [FC]*K
$$
也就是N个卷积层叠加，然后(可选)叠加一个Pooling层，重复这个结构M次，最后叠加K个全连接层。

图中 的模式架构可以表示为：
$$
INPUT -> [[CONV]*1 -> POOL]*2 -> [FC]*2
$$
也就是：`N=1, M=2, K=2`。

![image-20200520164139854](C:\Users\dafa\AppData\Roaming\Typora\typora-user-images\image-20200520164139854.png)

![image-20200520164155066](C:\Users\dafa\AppData\Roaming\Typora\typora-user-images\image-20200520164155066.png)

### 2.2 层结构

![image-20200520163248560](C:\Users\dafa\AppData\Roaming\Typora\typora-user-images\image-20200520163248560.png)

第一个卷积层对这幅图像进行了卷积操作，得到了三个Feature Map。

可以这样理解，这个卷积层包含三个Filter，也就是三套参数，每个Filter都可以把原始输入图像卷积得到一个Feature Map，三个Filter就可以得到三个Feature Map。至于一个卷积层可以有多少个Filter，那是可以自由设定的。也就是说，卷积层的Filter个数也是一个**超参数**。我们可以把Feature Map可以看做是通过卷积变换提取到的图像特征，三个Filter就对原始图像提取出三组不同的特征，也就是得到了三个Feature Map，也称做三个**通道(channel)**。

![image-20200520163528425](C:\Users\dafa\AppData\Roaming\Typora\typora-user-images\image-20200520163528425.png)

在第一个卷积层之后，Pooling层对三个Feature Map做了**下采样**(后面我们会讲如何计算下采样)，得到了三个更小的Feature Map。接着，是第二个**卷积层**，它有5个Filter。每个Fitler都把前面**下采样**之后的**3个\**Feature Map**卷积**在一起，得到一个新的Feature Map。这样，5个Filter就得到了5个Feature Map。接着，是第二个Pooling，继续对5个Feature Map进行**下采样**，得到了5个更小的Feature Map。

### 2.3  卷积神经网络输出值计算

#### 2.3.1 卷积层输出值计算

举例：

假设有一个5 * 5的图像，使用一个3 * 3的filter进行卷积，想得到一个3 * 3的Feature Map，如下所示：

![image-20200520164625771](C:\Users\dafa\AppData\Roaming\Typora\typora-user-images\image-20200520164625771.png)

![image-20200520171700164](C:\Users\dafa\AppData\Roaming\Typora\typora-user-images\image-20200520171700164.png)

![image-20200520171437778](C:\Users\dafa\AppData\Roaming\Typora\typora-user-images\image-20200520171437778.png)

![图2 卷积计算](http://upload-images.jianshu.io/upload_images/2256672-19110dee0c54c0b2.gif)

当深度大于1时，扩展公式1

![image-20200520171740514](C:\Users\dafa\AppData\Roaming\Typora\typora-user-images\image-20200520171740514.png)

每个卷积层可以有多个filter。每个filter和原始图像进行卷积后，都可以得到一个Feature Map。因此，卷积后Feature Map的深度(个数)和卷积层的filter个数是相同的。

下面的动画显示了包含两个filter的卷积层的计算。我们可以看到7*7*3输入，经过两个3*3*3filter的卷积(步幅为2)，得到了3*3*2的输出。另外我们也会看到下图的**Zero padding**是1，也就是在输入元素的周围补了一圈0。**Zero padding**对于图像边缘部分的特征提取是很有帮助的。![点击查看大图](http://upload-images.jianshu.io/upload_images/2256672-958f31b01695b085.gif)

以上就是卷积层的计算方法。这里面体现了**局部连接**和**权值共享**：每层神经元只和上一层部分神经元相连(卷积计算规则)，且filter的权值对于上一层所有神经元都是一样的。对于包含两个3*3*3的fitler的卷积层来说，其参数数量仅有(3*3*3+1)*2=56个，且参数数量与上一层神经元个数无关。与**全连接神经网络**相比，其参数数量大大减少了。

#### 2.3.2 Pooling层的输出值计算

**下采样**：通过去掉Feature Map中不重要的样本，进一步减少参数数量，主要使用Max Pooling，在n*n的样本中取最大值，作为采样后的样本值。

举例：

![点击查看大图](http://upload-images.jianshu.io/upload_images/2256672-03bfc7683ad2e3ad.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/640)

此外还有，**Mean Pooling**——取各样本的平均值

### 2.4 卷积神经网络的训练

训练原理：利用链式求导计算损失函数对每个权重的偏导数（梯度），然后根据梯度下降公式更新权重。训练算法依然是反向传播算法。

![image-20200520174744520](C:\Users\dafa\AppData\Roaming\Typora\typora-user-images\image-20200520174744520.png)

### 2.5 卷积神经网络的实现

构件一个简单的CNN模型，完成字符识别功能，包括两个卷积层，最后并联6个全连接层进行分类。



1、在实现自己的某层的时候基础了nn.Module,在构造函数中要调用Module的构造函数

super(Linear,self).init()

2、可学习参数放在构造函数中，并且通过nn.Parameter()使参数以parameters（一种tensor,默认是自动求导）的形式存在Module中，并且通过parameters()或者named_parameters()以迭代器的方式返回可学习参数的值

3、因为parameters是自动求导，所以调用forward()后，不用自己写和调用backward()函数。而且一般不是显式的调用forward(layer.farword)，而是layer(input)，会自执行forward()。

4、forward函数实现前向传播过程，其输入可以是一个或多个tensor。

5、module对象可以包含子module，Module能够自动检测到自己的Parameter并将其作为学习参数。除了parameter之外，Module还包含子Module，主Module能够递归查找子Modul中parameters。构造函数中，可利用前面自定义的Linear层(module)，作为当前module对象的一个子module，它的可学习参数，也会成为当前module的可学习参数。

```python
# 导入包
import torch
torch.manual_seed(0) #为CPU设置种子用于生成随机数，以使得结果是确定的
#cuDNN使用非确定性算法，并且可以使用torch.backends.cudnn.enabled = False来进行禁用
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#Autograd: 自动微分，为基于tensor的的所有操作提供自动微分的功能, 这是一个逐个运行的框架, 意味着反向传播是根据你的代码来运行的, 并且每一次的迭代运行都可能不同.autograd.Variable 是包的中央类, 它包裹着Tensor, 支持几乎所有Tensor的操作,并附加额外的属性, 在进行操作以后, 通过调用.backward()来计算梯度, 通过.data来访问原始raw data (tensor), 并将变量梯度累加到.grad
from torch.autograd import Variable

from torch.utils.data.dataset import Dataset

# 定义模型
class SVHN_Model1(nn.Module):
    def __init__(self):
        super(SVHN_Model1, self).__init__()
        # CNN提取特征模块
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),  
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(), 
            nn.MaxPool2d(2),
        )
        # 
        self.fc1 = nn.Linear(32*3*7, 11)
        self.fc2 = nn.Linear(32*3*7, 11)
        self.fc3 = nn.Linear(32*3*7, 11)
        self.fc4 = nn.Linear(32*3*7, 11)
        self.fc5 = nn.Linear(32*3*7, 11)
        self.fc6 = nn.Linear(32*3*7, 11)
    
    def forward(self, img):        
        feat = self.cnn(img)
        feat = feat.view(feat.shape[0], -1)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        c5 = self.fc5(feat)
        c6 = self.fc6(feat)
        return c1, c2, c3, c4, c5, c6
    
model = SVHN_Model1()

```

训练代码：

```python
# 损失函数
criterion = nn.CrossEntropyLoss()
# 优化器
optimizer = torch.optim.Adam(model.parameters(), 0.005)

loss_plot, c0_plot = [], []
# 迭代10个Epoch
for epoch in range(10):
    for data in train_loader:
        c0, c1, c2, c3, c4, c5 = model(data[0])
        loss = criterion(c0, data[1][:, 0]) + \
                criterion(c1, data[1][:, 1]) + \
                criterion(c2, data[1][:, 2]) + \
                criterion(c3, data[1][:, 3]) + \
                criterion(c4, data[1][:, 4]) + \
                criterion(c5, data[1][:, 5])
        loss /= 6
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_plot.append(loss.item())
        c0_plot.append((c0.argmax(1) == data[1][:, 0]).sum().item()*1.0 / c0.shape[0])
        
    print(epoch)
```

也可以使用在ImageNet数据集上的预训练模型，具体方法如下：   

```python
class SVHN_Model2(nn.Module):
	def __init__(self):
        super(SVHN_Model1, self).__init__()
        
        #使用预训练模型
        model_conv = models.resnet18(pretrained=True)
        model_conv.avgpool = nn.AdaptiveAvgPool2d(1)
        model_conv = nn.Sequential(*list(model_conv.children())[:-1])
        self.cnn = model_conv
        
        self.fc1 = nn.Linear(512, 11)
        self.fc2 = nn.Linear(512, 11)
        self.fc3 = nn.Linear(512, 11)
        self.fc4 = nn.Linear(512, 11)
        self.fc5 = nn.Linear(512, 11)
    
    def forward(self, img):        
        feat = self.cnn(img)
        # print(feat.shape)
        feat = feat.view(feat.shape[0], -1)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        c5 = self.fc5(feat)
        return c1, c2, c3, c4, c5
```

