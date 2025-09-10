class hparams:
    device_id = 0  # 指定使用的GPU ID
    model_name = 'resnet3d_101'  # 训练用的模型
    total_epochs = 10  # 训练轮数
    batch_size = 8  # 批次大小
    num_classes = 2  # 分类数
    log_dir = 'logs/experiment_10'  # 日志保存路径
    ckpt = None  # 模型保存路径
    save_by_metric = 'val_acc'  # 保存模型时使用的指标
    init_lr = 0.0002  # 初始学习率
    scheduer_step_size = 20  # 学习率衰减间隔
    scheduer_gamma = 0.8  # 学习率衰减因子
    is_3d = True
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    data_dir = '/workdir1/echo_dataset/MI-DATA/CAMUS/pt_data/'
    metadata_path = 'metadata/label_select160.csv'
