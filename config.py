class DefaultConfig(object):
    dataset_dir = './data/'
    model_cp = './checkpoint/'
    test_data_dir = './data/test/'
    model_file = './checkpoint/model.pth'
    workers = 8 # 用来加载数据的进程数
    batch_size = 16 # 批处理数量
    lr = 0.0001 # 学习率
    nepoch = 10 # 训练次数
    use_gpu = False # gpu优化
    N = 10 # 随机测试N张图片

opt = DefaultConfig()