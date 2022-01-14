import os
import torch.utils.data as data
from PIL import Image
import torch
import torchvision.transforms as transforms

IMAGE_SIZE = 200

dataTransform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE), # 将图片resize成统一大小
    transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class CatsVSDogsDataset(data.Dataset):
    def __init__(self, mode, dir): # 默认构造函数，传入数据集类别（train 或 test），以及数据集路径
        # 只导入数据文件名，而不是数据的全部内容
        self.mode = mode
        self.list_img = []  # 存储图片的完整相对地址
        self.list_label = []  # 存储图片标签，0表示猫，1表示狗
        self.data_size = 0 # 样本数量
        self.transform = dataTransform

        if self.mode == 'train':
            dir += 'train/'
            for file in os.listdir(dir): # cat.0.jpg
                self.list_img.append(dir + file)
                # label采用one-hot编码，
                # "1,0"表示猫，"0,1"表示狗，任何情况只有一个位置为"1"，
                # 在采用CrossEntropyLoss()计算Loss情况下，label只需要输入"1"的索引，
                # 即猫应输入0，狗应输入1
                self.list_label.append(0 if file.split(sep='.')[0] == 'cat' else 1)
                self.data_size += 1
                if self.data_size == 5000: break
        elif self.mode == 'test':
            dir += 'test/'
            for file in os.listdir(dir):
                self.list_img.append(dir + file)
                self.list_label.append(2) # 添加2作为label，实际未用到，也无意义
                self.data_size += 1
        else:
            print('Undefined Dataset!')

    def __getitem__(self, item): # 重载data.Dataset父类方法，获取数据集中数据内容
        if self.mode == 'train':
            img = Image.open(self.list_img[item])
            label = self.list_label[item]
            return self.transform(img), torch.LongTensor([label])
        elif self.mode == 'test':
            img = Image.open(self.list_img[item])
            return self.transform(img)
        else:
            print('None')

    def __len__(self): # 数据集大小
        return self.data_size 

    def get_item(self, index): # 获取list里的元素
        if self.mode == 'train':
            return self.list_img[index], self.list_label[index]
        elif self.mode == 'test':
            return self.list_img[index]
        else:
            print('None')
