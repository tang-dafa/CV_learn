# Task2数据读取与数据扩增

## 1.目标

1. Pillow与Pytorch中的图像读取
2. 学习扩增方法
3. 使用Pytorch读取赛题数据

## 2.图像读取

主要使用Pillow与Pytorch

### 2.1 Pillow

对照教学文档使用

```python
from PIL import Image,ImageFilter

im = Image.open('5b567748cf10.jpg')
im.show()
print(im)

# 模糊滤镜
im2 = im.filter(ImageFilter.BLUR)
im2.save('blur.jpg','jpeg')
im2.show()
```

结果

<img src="C:\Users\dafa\AppData\Roaming\Typora\typora-user-images\image-20200520122942777.png" alt="image-20200520122942777" style="zoom: 50%;" />

Pillow的官方文档：https://pillow.readthedocs.io/en/stable/

### 2.2 OpenCV

对照教学文档使用

```python
import cv2
img = cv2.imread('5b567748cf10.jpg')

# 转换颜色通道，默认为 BRG
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# 转换为灰度图
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('img_gray.jpg',img_gray)
# Canny边缘检测
edges = cv2.Canny(img,30,70)
cv2.imwrite('canny.jpg',edges)
# BGR2HLS
img_BGR2HLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
cv2.imwrite('img_BGR2HLS.jpg',img_BGR2HLS)
```

结果

![image-20200520123946945](C:\Users\dafa\AppData\Roaming\Typora\typora-user-images\image-20200520123946945.png)

![image-20200520124010778](C:\Users\dafa\AppData\Roaming\Typora\typora-user-images\image-20200520124010778.png)

![image-20200520124242484](C:\Users\dafa\AppData\Roaming\Typora\typora-user-images\image-20200520124242484.png)

OpenCV官网：https://opencv.org/       
OpenCV Github：https://github.com/opencv/opencv      
OpenCV 扩展算法库：https://github.com/opencv/opencv_contrib

## 3.数据扩增方法

### 31数据扩增的作用：

增加训练集的样本，有效缓解过拟合，拥有更强的繁华能力。

### 3.2**常见的数据扩增的方法**：

- 从颜色空间、尺度空间到样本空间，同时根据不同任务数据扩增都有相应的区别。        
- 对于图像分类，数据扩增一般不会改变标签；
- 对于物体检测，数据扩增会改变物体坐标位置；
- 对于图像分割，数据扩增会改变像素标签。

一般会从图像颜色、尺寸、形态、空间和像素等角度进行变换。当然不同的数据扩增方法可以自由进行组合，得到更加丰富的数据扩增方法。

**以torchvision为例，常见的数据扩增方法包括**：

- transforms.CenterCrop      对图片中心进行裁剪      

- transforms.ColorJitter      对图像颜色的对比度、饱和度和零度进行变换      

- transforms.FiveCrop     对图像四个角和中心进行裁剪得到五分图像     

- transforms.Grayscale      对图像进行灰度变换    

- transforms.Pad        使用固定值进行像素填充     

- transforms.RandomAffine      随机仿射变换    

- transforms.RandomCrop      随机区域裁剪     

- transforms.RandomHorizontalFlip      随机水平翻转     

- transforms.RandomRotation     随机旋转     

- transforms.RandomVerticalFlip     随机垂直翻转

  ### 3.3常用的库

1. **torchvision**      

   https://github.com/pytorch/vision      
   pytorch官方提供的数据扩增库，提供了基本的数据数据扩增方法，可以无缝与torch进行集成；但数据扩增方法种类较少，且速度中等；       

2. #### imgaug         

   https://github.com/aleju/imgaug      
   imgaug是常用的第三方数据扩增库，提供了多样的数据扩增方法，且组合起来非常方便，速度较快；      

3. #### albumentations       

   https://albumentations.readthedocs.io      
   是常用的第三方数据扩增库，提供了多样的数据扩增方法，对图像分类、语义分割、物体检测和关键点检测都支持，速度较快。   

## 4.Pytorch读取数据

在Pytorch中数据是通过Dataset进行封装，并通过DataLoder进行并行读取。

from torch.utila.data.dataset import Dataset # 用这个类构造pytorch的数据集

以这个类构造的子类，一定要定义两个函数一个是__len__，另一个是__getitem__，前者提供数据集size，而后者通过给定索引获取数据和标签。__getitem__一次只能获取一个数据（不知道是不是强制性的），所以通过torch.utils.data.DataLoader来定义一个新的迭代器，实现batch读取。

```python
# 导入模块
import os,sys,glob,shutil,json
import cv2
from PIL import Image
import numpy as np
import torch
from torch.utils.data.dataset import Dataset # 用这个类构造pytorch的数据集
import torchvision.transforms as transforms
```

**Dataset：对数据集的封装，提供索引方式的对数据样本进行读取**

```python
class SVHNDataset(Dataset):
    def __init__(self,img_path, img_label , transform=None):
        self.img_path = img_path
        self.img_label= img_label
        if transform is not None :
            self.transorm = transform
        else:
            self.transform = None
            
    def __getitem__ (self , index):
        img =Image.open(self.img_path[index]).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
            
        # 原始SVHN中类别10为数字0
        lbl = np.array(self.iimg_label[index] , dtype=np.int)
        lbl = list(lbl) + (5 - len(lbl)) * [10]
        
        return img , torch.from_numpy(np.array(lbl[:5]))
    
    def __len__(self):
        return len(self.img_path)
    
train_path = glob.glob('D:\\PartGit\\team-learning\\cv_data\\train\\train\\*.png')
train_path.sort()
train_json = json.load(open('D:\\PartGit\\team-learning\\cv_data\\train.json'))
train_label = [train_json[x]['label'] for x in train_json]

data = SVHNDataset(train_path , train_label,
                  transforms.Compose([
                      # 缩放固定尺寸
                      transforms.Resize((64,128)),
                      # 随机颜色变换
                      transforms.ColorJitter(0.2,0.2,0.2),
                      # 加入随机旋转
                      transforms.RandomRotation(5),
                      # 将图片转换为pytorch的tensor
                      # transforms.ToTensor(),
                      
                      # 图像像素归一化
                      # transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])   
                  ]))
```

通过torch.utils.data.DataLoader来定义一个新的迭代器，实现batch读取

**DataLoder：对Dataset进行封装，提供批量读取的迭代读取 **

加入DataLoder之后

```python
class SVHNDataset(Dataset):
    def __init__(self,img_path, img_label , transform=None):
        self.img_path = img_path
        self.img_label= img_label
        if transform is not None :
            self.transorm = transform
        else:
            self.transform = None
            
    def __getitem__ (self , index):
        img =Image.open(self.img_path[index]).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
            
        # 原始SVHN中类别10为数字0
        lbl = np.array(self.img_label[index] , dtype=np.int)
        lbl = list(lbl) + (5 - len(lbl)) * [10]
        
        return img , torch.from_numpy(np.array(lbl[:5]))
    
    def __len__(self):
        return len(self.img_path)
    
train_path = glob.glob('D:\\PartGit\\team-learning\\cv_data\\train\\train\\*.png')
train_path.sort()
train_json = json.load(open('D:\\PartGit\\team-learning\\cv_data\\train.json'))
train_label = [train_json[x]['label'] for x in train_json]

# 这一部分不一样
train_loader = torch.utils.data.DataLoader(
    SVHNDataset(train_path , train_label,
                  transforms.Compose([
                      # 缩放固定尺寸
                      transforms.Resize((64,128)),
                      # 随机颜色变换
                      transforms.ColorJitter(0.2,0.2,0.2),
                      # 加入随机旋转
                      transforms.RandomRotation(5),
                      # 将图片转换为pytorch的tensor
                      transforms.ToTensor(),
                      # 图像像素归一化
                      transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])   
                  ])),
    batch_size   = 10,    # 每批样本的个数
    shuffle      = False, # 是否打乱顺序
    num_workers  =10      # 读取线程的个数
)

for data in train_loader:
    break
    
```

在加入DataLoder后，数据按照批次获取，每批次调用Dataset读取单个样本进行拼接。此时data的格式为：       
                ``` torch.Size([10, 3, 64, 128]), torch.Size([10, 6]) ```          
前者为图像文件，为batchsize * chanel * height * width次序；后者为字符标签。