class DefaultConfig(object):
    dataset_dir = './data/'
    model_cp = './checkpoint/'
    test_data_dir = './data/test/'
    model_file = './checkpoint/model.pth'
    workers = 8 
    batch_size = 16 # 批处理数量
    lr = 0.0001 # 学习率
    nepoch = 10
    use_gpu = False

opt = DefaultConfig()