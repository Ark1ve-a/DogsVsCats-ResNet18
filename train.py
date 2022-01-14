from config import opt
from dataset import CatsVSDogsDataset as CVDD
from torch.utils.data import DataLoader as DataLoader
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.models as models


def train():
    datafile = CVDD('train', opt.dataset_dir)
    # 用PyTorch的DataLoader类封装
    # 实现数据集顺序打乱，多线程读取，一次取多个数据等效果
    dataloader = DataLoader(datafile, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)

    print('Dataset loaded! length of train set is {0}'.format(len(datafile)))

    model = models.resnet18(num_classes=2)
    if opt.use_gpu:
        model = model.cuda()
    model = nn.DataParallel(model)
    model.train()
    # 实例化一个优化器，即调整网络参数，优化方式为adam方法
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    # 定义loss计算方法，cross entropy，交叉熵，可以理解为两者数值越接近其值越小
    criterion = nn.CrossEntropyLoss()

    cnt = 0
    for epoch in range(1, opt.nepoch + 1):
        for img, label in dataloader:  # 循环读取封装后的数据集，其实就是调用了数据集中的__getitem__()方法，只是返回数据格式进行了一次封装
            img, label = Variable(img), Variable(label)  # 将数据放置在PyTorch的Variable节点中
            if opt.use_gpu:
                img, label = img.cuda(), label.cuda()
            out = model(img)  # 计算网络输出值，就是输入网络一个图像数据，输出猫和狗的概率，调用了网络中的forward()方法
            loss = criterion(out, label.squeeze())  # 计算损失，也就是网络输出值和实际label的差异，显然差异越小说明网络拟合效果越好，此处需要注意的是第二个参数，必须是一个1维Tensor
            loss.backward()  # 误差反向传播，采用求导的方式，计算网络中每个节点参数的梯度，显然梯度越大说明参数设置不合理，需要调整
            optimizer.step()  # 优化采用设定的优化方法对网络中的各个参数进行调整
            optimizer.zero_grad()  # 清除优化器中的梯度以便下一次计算，因为优化器默认会保留，不清除的话，每次计算梯度都回累加
            cnt += 1

            print('Epoch:{0}, Frame:{1}, train_loss {2}'.format(epoch, cnt * opt.batch_size, loss / opt.batch_size))  # 打印一个batch size的训练结果

    torch.save(model.state_dict(), '{0}/model.pth'.format(opt.model_cp)) # 训练所有数据后，保存网络的参数


if __name__ == '__main__':
    train()
