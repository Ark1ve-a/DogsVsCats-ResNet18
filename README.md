# 猫狗大战

### 题目来源

2014年的Kaggle竞赛[Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/overview)

### 数据集来源

[Dogs vs. Cats | Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data)

**训练集**：训练集由标记为**cat**和**dog**的猫狗图片组成，各`12500`张，总共`25000`张，图片为24位jpg格式，即RGB三通道图像，**图片尺寸不一**。

**测试集**：测试集由`12500`张的cat或dog图片组成，**未标记**，图片也为24位jpg格式，RGB三通道图像，**图像尺寸不一**。

使用ResNet18二分类网络实现，

### 导入数据

训练数据和测试数据存在`CatsVSDogsDataset`类中，初始化时根据数据集类别，train数据集导入图片的完整相对地址`list_img` 和 图片的标签 `list_label` ，标签0表示猫，1表示狗，test数据集只导入图片的完整相对地址。

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

### 训练过程

使用`torchvision.models`下的`resnet18`模型，使用Adam优化方法，loss计算方法为交叉熵，可以理解为两者数值越接近其值越小。

