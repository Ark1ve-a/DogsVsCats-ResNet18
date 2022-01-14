# 猫狗大战

# 猫狗大战

### 题目来源

**2014年的Kaggle竞赛**[Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/overview)

### 数据集来源

[Dogs vs. Cats | Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data)

**训练集**：训练集由标记为**cat**和**dog**的猫狗图片组成，各 `12500`张，总共 `25000`张，图片为24位jpg格式，即RGB三通道图像，**图片尺寸不一**。

**测试集**：测试集由 `12500`张的cat或dog图片组成，**未标记**，图片也为24位jpg格式，RGB三通道图像，**图像尺寸不一**。

**使用ResNet18二分类网络实现，**

### 导入数据

**训练数据和测试数据存在** `CatsVSDogsDataset`类中，初始化时根据数据集类别，train数据集导入图片的完整相对地址 `list_img` 和 图片的标签 `list_label` ，标签0表示猫，1表示狗，test数据集只导入图片的完整相对地址。

**图片的转换操作包括：图片大小统一、中心裁剪、转换为Tensor、标准化。并将数据的label转换为LongTensor。**

```
 IMAGE_SIZE = 200
 
 dataTransform = transforms.Compose([
     transforms.Resize(IMAGE_SIZE), # 将图片resize成统一大小
     transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
 ])
```

`DataLoader()`的作用是对定义好的数据集类做一次封装，有以下作用：

* **可以定义为打乱数据集分布，使各个类型的样本均匀地参与网络训练；**
* **可以设定多线程数据读取(指数据载入内存)，提高训练效率，因为训练过程中文件的读取是比较耗时的**
* **可以一次获得batch size大小的数据，并且是Tensor形式，比如读取4个训练数据，需要调用** `__getitem__()`4次，但是封装好后，在for img, label in dataloader:中，一次img可以获得4个(假设batch size为4)数据，这也就是为什么代码中img的size是[16×3×200×200]，而直接调用 `__getitem__()`是[3×200×200]的原因，因为DataLoader封装过程中加入了一维数据个数。

```
 dataloader = DataLoader(datafile, 
                         batch_size=opt.batch_size, 
                         shuffle=True, 
                         num_workers=opt.workers)
```

### 训练过程

**使用** `torchvision.models`下的 `resnet18`模型，使用Adam优化方法，loss计算方法为交叉熵，可以理解为两者数值越接近其值越小。

**使用** `tqdm` 完成训练进度以及loss的可视化

### 题目来源

2014年的Kaggle竞赛[Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/overview)

### 数据集来源

[Dogs vs. Cats | Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data)

**训练集**：训练集由标记为**cat**和**dog**的猫狗图片组成，各 `12500`张，总共 `25000`张，图片为24位jpg格式，即RGB三通道图像，**图片尺寸不一**。

**测试集**：测试集由 `12500`张的cat或dog图片组成，**未标记**，图片也为24位jpg格式，RGB三通道图像，**图像尺寸不一**。

使用ResNet18二分类网络实现，

### 导入数据

训练数据和测试数据存在 `CatsVSDogsDataset`类中，初始化时根据数据集类别，train数据集导入图片的完整相对地址 `list_img` 和 图片的标签 `list_label` ，标签0表示猫，1表示狗，test数据集只导入图片的完整相对地址。

图片的转换操作包括：图片大小统一、中心裁剪、转换为Tensor、标准化。并将数据的label转换为LongTensor。

```python
IMAGE_SIZE = 200

dataTransform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE), # 将图片resize成统一大小
    transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

`DataLoader()`的作用是对定义好的数据集类做一次封装，有以下作用：

- 可以定义为打乱数据集分布，使各个类型的样本均匀地参与网络训练；
- 可以设定多线程数据读取(指数据载入内存)，提高训练效率，因为训练过程中文件的读取是比较耗时的
- 可以一次获得batch size大小的数据，并且是Tensor形式。

```python
dataloader = DataLoader(datafile, 
                        batch_size=opt.batch_size, 
                        shuffle=True, 
                        num_workers=opt.workers)
```



### 训练过程

使用 `torchvision.models`下的 `resnet18`模型，使用Adam优化方法，loss计算方法为交叉熵，可以理解为两者数值越接近其值越小。

使用 `tqdm` 完成训练进度以及loss的可视化。

>  `Variable()`的理解，在代码出现了这个，可以把它理解为定义一个符号，一个未知数，然后网络中的所有计算方式都用这个符号来表示，最终网络计算形成一个由Variable()组成的复杂计算公式，只要将实际数据代入Variable，便可快速求出结果，并且，求导也十分方便，因为都是符号，可以调用求导公式，当然这些都是内部计算过程，外部看不到。

需要注意的是代码`loss = criterion(out, label.squeeze())`中的`.squeeze()`方法，因为从dataloader中获取的label是一个[batch_size × 1]的Tensor，而实际输入的应是1维Tensor，所以需要做一个维度变换。

最后保存模型即可。
