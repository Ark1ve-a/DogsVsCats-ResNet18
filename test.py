from config import opt
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
import dataset

np.set_printoptions(suppress=True)

def test():
    # setting model
    model = models.resnet18(num_classes=2) # 实例化网络
    if opt.use_gpu: # gpu优化
        model = model.cuda()
    model = nn.DataParallel(model) # 多线程
    model.load_state_dict(torch.load(opt.model_file)) # 加载训练模型
    model.eval() # 设定为评估模式，计算过程不要dropout

    # get data
    files = random.sample(os.listdir(opt.test_data_dir), opt.N)
    imgs = []  # 原始图片，用来plt.imshow(imgs[index])
    imgs_data = []  # 训练图片，dataTransform(img)
    for file in files:
        img = Image.open(opt.test_data_dir + file)
        imgs.append(img)
        imgs_data.append(dataset.dataTransform(img))
    imgs_data = torch.stack(imgs_data) # 把Tensor list 拼接成一个Tensor
    if opt.use_gpu: # gpu优化
        imgs_data = imgs_data.cuda()

    # calculation
    # 对资源闭合访问，进行必要的清理（close等操作）
    with torch.no_grad(): # 不自动求导
        out = model(imgs_data)
    out = F.softmax(out, dim=1) # 将分类结果转换为[0, 1]的概率
    out = out.data.cpu().numpy()

    # 展示测试图片以及预测结果
    for idx in range(opt.N):
        plt.figure()
        if out[idx, 0] > out[idx, 1]:
            plt.suptitle('cat:{:.1%},dog:{:.1%}'.format(out[idx, 0], out[idx, 1]))
        else:
            plt.suptitle('dog:{:.1%},cat:{:.1%}'.format(out[idx, 1], out[idx, 0]))
        plt.imshow(imgs[idx])
    plt.show()


if __name__ == '__main__':
    test()
