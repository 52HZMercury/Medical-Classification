class hparams:
    device_id = 0  # 指定使用的GPU ID
    model_name = 'resnet18' # 训练用的模型
    total_epochs = 60 # 训练轮数
    batch_size = 16 # 批次大小
    num_classes = 5 # 分类数
    log_dir = 'logs/experiment_9' # 日志保存路径
    ckpt = None # 模型保存路径
    save_by_metric = 'val_acc' # 保存模型时使用的指标
    init_lr = 0.0002 # 初始学习率
    scheduer_step_size = 20 # 学习率衰减间隔
    scheduer_gamma = 0.8 # 学习率衰减因子

    is_3d = False
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    data_dir = 'E:/DataSet/flower_photos_5/'
