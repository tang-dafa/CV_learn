# Task4 模型训练与验证

## 1.学习目标

- 理解验证集的作用，并使用训练集和验证集完成训练         
- 学会使用Pytorch环境下的模型读取和加载，并了解调参流程  

## 2.构造验证集

- 深度学习的过拟合

  ​	深度学习模型在不断的训练过程中训练误差会逐渐降低，但测试误差的走势则不一定。

  ​	在模型的训练过程中，模型只能利用训练数据来进行训练，模型并不能接触到测试集上的样本。因此模型如果将训练集学的过好，模型就会记住训练样本的细节，导致模型在测试集的泛化效果较差，这种现象称为过拟合（Overfitting）。与过拟合相对应的是欠拟合（Underfitting），即模型在训练集上的拟合效果较差。  

![image-20200525124012611](C:\Users\dafa\AppData\Roaming\Typora\typora-user-images\image-20200525124012611.png)

**导致过拟合的原因**：模型复杂度（Model Complexity ）太高，导致模型学习到了训练数据的方方面面，学习到了一些细枝末节的规律。 

**解决方法**：构建一个与测试集尽可能分布一致的样本集（可称为验证集），在训练过程中不断验证模型在验证集上的精度，并以此控制模型的训练。

- ### **如何划分验证集**

  - 训练集（Train Set）：模型用于训练和调整模型参数；
  - 验证集（Validation Set）：用来验证模型精度和调整模型超参数；
  - 测试集（Test Set）：验证模型的泛化能力。

- **留出法（Hold-Out）**           

  直接将训练集划分成两部分，新的训练集和验证集。这种划分方式的优点是最为直接简单；缺点是只得到了一份验证集，有可能导致模型在验证集上过拟合。留出法应用场景是数据量比较大的情况。     

- **交叉验证法（Cross Validation，CV）**      

  将训练集划分成K份，将其中的K-1份作为训练集，剩余的1份作为验证集，循环K训练。这种划分方式是所有的训练集都是验证集，最终模型验证精度是K份平均得到。这种方式的优点是验证集精度比较可靠，训练K次可以得到K个有多样性差异的模型；CV验证的缺点是需要训练K次，不适合数据量很大的情况。     

- **自助采样法（BootStrap）**      

  通过有放回的采样方式得到新的训练集和验证集，每次的训练集和验证集都是有区别的。这种划分方式一般适用于数据量较小的情况。      
  ![image-20200525124516024](C:\Users\dafa\AppData\Roaming\Typora\typora-user-images\image-20200525124516024.png)

## 3.模型训练与验证

- 构造训练集和验证集；
- 每轮进行训练和验证，并根据最优验证集精度保存模型

```python
# 训练集
# torch.utils.data.DataLoader（
#                            batch_size = 1，        #批处理大小设置
#                            shuffle = False，		#是否进项洗牌操作
#                            sampler = None，		#指定数据加载中使用的索引/键的序列
#                            batch_sampler = None，	#和sampler类似
#                            num_workers = 0，		#是否进行多进程加载数据设置
#                            collat​​e_fn = None，	#是否合并样本列表以形成一小批Tensor
#                            pin_memory = False，	#如果True，数据加载器会在返回之前将Tensors复制到CUDA固定内存
#                            drop_last = False，		#True如果数据集大小不能被批处理大小整除，则设置为删除最后一个不完整的批处理。
#                            timeout = 0，			#如果为正，则为从工作人员收集批处理的超时值
#                            worker_init_fn = None ）


train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=10, 
    shuffle=True, 
    num_workers=10, 
)

#验证集
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=10, 
    shuffle=False, 
    num_workers=10, 
)

#模型
model = SVHN_Model1()
#损失函数
criterion = nn.CrossEntropyLoss (size_average=False)
#params (iterable) – 待优化参数的iterable或者是定义了参数组的dict
#lr (float, 可选) – 学习率（默认：1e-3）
#betas (Tuple[float, float], 可选) – 用于计算梯度以及梯度平方的运行平均值的系数（默认：0.9，0.999）
#eps (float, 可选) – 为了增加数值计算的稳定性而加到分母里的项（默认：1e-8）
#weight_decay (float, 可选) – 权重衰减（L2惩罚）（默认: 0）

# 优化器
optimizer = torch.optim.Adam(model.parameters(), 0.001)
best_loss = 1000.0
for epoch in range(20):
    print('Epoch: ', epoch)

    train(train_loader, model, criterion, optimizer, epoch)
    val_loss = validate(val_loader, model, criterion)
    
    # 记录下验证集精度
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), './model.pt')
```



每个Epoch的训练代码：

```python
def train(train_loader,model,criterion , optimizer , epoch) :
	# 切换模式为训练模式
	model.train()
	
	for i,(input,target) in enumerate(train_loader):
		c0,c1,c2,c3,c4,c5 = model(data[0])
        loss = criterion(c,0,data[1][:,0]) + \
       			criterion(c1, data[1][:, 1]) + \
                criterion(c2, data[1][:, 2]) + \
                criterion(c3, data[1][:, 3]) + \
                criterion(c4, data[1][:, 4]) + \
                criterion(c5, data[1][:, 5])
                
        loss /=6
        optimizer.zero_grad()  #清空过往梯度
        loss.backward()       # 反向传播，计算当前梯度
        optimizer.step()       # 根据梯度更新网络参数
```

每个Epoch的验证代码：

```python
def validate(val_loader, model, criterion):
    # 切换模型为预测模型
    model.eval()
    val_loss = []

    # 不记录模型梯度信息
    # with torch.no_grad()在eval阶段了，即使不更新，但是在模型中所使用的dropout或者batch norm也就失效了，直接都会进行预测，而使用no_grad则设置让梯度Autograd设置为False(因为在训练中我们默认是True)，这样保证了反向过程为纯粹的测试，而不变参数。另外，参考文档说这样避免每一个参数都要设置，解放了GPU底层的时间开销，在测试阶段统一梯度设置为False
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            c0, c1, c2, c3, c4, c5 = model(data[0])
            loss = criterion(c0, data[1][:, 0]) + \
                    criterion(c1, data[1][:, 1]) + \
                    criterion(c2, data[1][:, 2]) + \
                    criterion(c3, data[1][:, 3]) + \
                    criterion(c4, data[1][:, 4]) + \
                    criterion(c5, data[1][:, 5])
            loss /= 6
            val_loss.append(loss.item())
    return np.mean(val_loss)
```



```python
# model.train() 与model.eval()
# 分这两个模式是针对模型在评价和训练时不同的Batch Normalization 和 Dropout方法模式
# eval()时，pytorch会自动把BN和Dropout固定住，不会取平均，而是用训练好的值。
# 不然如果test的batch_size过小，很容易被BN层导致图片颜色失真过大。
```

## 4.模型保存和加载

```python
#在Pytorch中模型的保存和加载非常简单，比较常见的做法是保存和加载模型参数：        
torch.save(model_object.state_dict(), 'model.pt')            
model.load_state_dict(torch.load(' model.pt')) 
       
```

 ## 5. 模型调参流程     

深度学习原理少但实践性非常强，基本上很多的模型的验证只能通过训练来完成。同时深度学习有众多的网络结构和超参数，因此需要反复尝试。训练深度学习模型需要GPU的硬件支持，也需要较多的训练时间，如何有效的训练深度学习模型逐渐成为了一门学问。
             
深度学习有众多的训练技巧，比较推荐的阅读链接有：          

- http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html      
- http://karpathy.github.io/2019/04/25/recipe/     
         

本节挑选了常见的一些技巧来讲解，并针对本次赛题进行具体分析。与传统的机器学习模型不同，深度学习模型的精度与模型的复杂度、数据量、正则化、数据扩增等因素直接相关。所以当深度学习模型处于不同的阶段（欠拟合、过拟合和完美拟合）的情况下，大家可以知道可以什么角度来继续优化模型。                 
                
在参加本次比赛的过程中，我建议大家以如下逻辑完成：      
        

- 1.初步构建简单的CNN模型，不用特别复杂，跑通训练、验证和预测的流程；    
- 2.简单CNN模型的损失会比较大，尝试增加模型复杂度，并观察验证集精度；            
- 3.在增加模型复杂度的同时增加数据扩增方法，直至验证集精度不变。



![image-20200529110332747](C:\Users\dafa\AppData\Roaming\Typora\typora-user-images\image-20200529110332747.png)



## 补充：

这里首先解释一下bias和variance的概念。模型的Error = Bias + Variance，Error反映的是整个模型的准确度，Bias反映的是模型在样本上的输出与真实值之间的误差，即模型本身的精准度，Variance反映的是模型每一次输出结果与模型输出期望之间的误差，即模型的稳定性。

我们可以根据j_cv 与 j_train两个来判断是处于欠拟合还是过拟合。
![image-20200525200608749](C:\Users\dafa\AppData\Roaming\Typora\typora-user-images\image-20200525200608749.png)



当观察到 J_cv 很大时，可能处在途中蓝色圆圈中的两个位置，虽然观察到的现象很相似(J_cv都很大)，但这两个位置的状态是非常不同的，处理方法也完全不同。

当cross validation error (Jcv) 跟training error(Jtrain)差不多，且Jtrain较大时，即图中标出的bias，此时 high bias low variance，当前模型更可能存在欠拟合。
当Jcv >> Jtrain且Jtrain较小时，即图中标出的variance时，此时 low bias high variance，当前模型更可能存在过拟合。





***\*1. 欠拟合\****

首先欠拟合就是模型没有很好地捕捉到数据特征，不能够很好地拟合数据，例如下面的例子：

![image-20200525200456372](C:\Users\dafa\AppData\Roaming\Typora\typora-user-images\image-20200525200456372.png)

左图表示size与prize关系的数据，中间的图就是出现欠拟合的模型，不能够很好地拟合数据，如果在中间的图的模型后面再加一个二次项，就可以很好地拟合图中的数据了，如右面的图所示。



解决方法：

1）添加其他特征项，有时候我们模型出现欠拟合的时候是因为特征项不够导致的，可以添加其他特征项来很好地解决。例如，“组合”、“泛化”、“相关性”三类特征是特征添加的重要手段，无论在什么场景，都可以照葫芦画瓢，总会得到意想不到的效果。除上面的特征之外，“上下文特征”、“平台特征”等等，都可以作为特征添加的首选项。

2）添加多项式特征，这个在机器学习算法里面用的很普遍，例如将线性模型通过添加二次项或者三次项使模型泛化能力更强。例如上面的图片的例子。

3）减少正则化参数，正则化的目的是用来防止过拟合的，但是现在模型出现了欠拟合，则需要减少正则化参数。



***\*2. 过拟合\****

通俗一点地来说过拟合就是模型把数据学习的太彻底，以至于把噪声数据的特征也学习到了，这样就会导致在后期测试的时候不能够很好地识别数据，即不能正确的分类，模型泛化能力太差。例如下面的例子。

![image-20200525200437586](C:\Users\dafa\AppData\Roaming\Typora\typora-user-images\image-20200525200437586.png)

解决方法：

1）重新清洗数据，导致过拟合的一个原因也有可能是数据不纯导致的，如果出现了过拟合就需要我们重新清洗数据。

2）增大数据的训练量，还有一个原因就是我们用于训练的数据量太小导致的，训练数据占总数据的比例过小。

3）采用正则化方法。正则化方法包括L0正则、L1正则和L2正则，而正则一般是在目标函数之后加上对于的范数。但是**在机器学习中一般使用L2正则**，下面看具体的原因。

L0范数是指向量中非0的元素的个数。L1范数是指向量中各个元素绝对值之和，也叫“稀疏规则算子”（Lasso regularization）。两者都可以实现稀疏性，既然L0可以实现稀疏，为什么不用L0，而要用L1呢？个人理解一是因为L0范数很难优化求解（NP难问题），二是L1范数是L0范数的最优凸近似，而且它比L0范数要容易优化求解。所以大家才把目光和万千宠爱转于L1范数。

L2范数是指向量各元素的平方和然后求平方根。可以使得W的每个元素都很小，都接近于0，但与L1范数不同，它不会让它等于0，而是接近于0。L2正则项起到使得参数w变小加剧的效果，但是为什么可以防止过拟合呢？一个通俗的理解便是：更小的参数值w意味着模型的复杂度更低，对训练数据的拟合刚刚好（奥卡姆剃刀），不会过分拟合训练数据，从而使得不会过拟合，以提高模型的泛化能力。还有就是看到有人说L2范数有助于处理 condition number不好的情况下矩阵求逆很困难的问题（具体这儿我也不是太理解）。

4）采用dropout方法。这个方法在神经网络里面很常用。dropout方法是ImageNet中提出的一种方法，通俗一点讲就是。具体看下图：
![image-20200525200345599](C:\Users\dafa\AppData\Roaming\Typora\typora-user-images\image-20200525200345599.png)

如上图所示，左边a图是没用用dropout方法的标准神经网络，右边b图是在训练过程中使用了dropout方法的神经网络，即在训练时候以一定的概率p来跳过一定的神经元。

