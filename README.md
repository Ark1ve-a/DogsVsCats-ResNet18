# 猫狗大战

### 题目来源

2014年的Kaggle竞赛[Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/overview)

### 数据集来源

[Dogs vs. Cats | Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data)

**训练集**：训练集由标记为**cat**和**dog**的猫狗图片组成，各`12500`张，总共`25000`张，图片为24位jpg格式，即RGB三通道图像，**图片尺寸不一**。

**测试集**：测试集由`12500`张的cat或dog图片组成，**未标记**，图片也为24位jpg格式，RGB三通道图像，**图像尺寸不一**。

使用ResNet18二分类网络实现，

