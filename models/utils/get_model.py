import torch.nn as nn
from ..alexnet import alexnet
from ..densenet import densenet121
from ..densenet import densenet161
from ..densenet import densenet169
from ..densenet import densenet201
from ..googlenet import googlenet
from ..resnet import resnet18
from ..resnet3d import generate_model as resnet3d


def get_model(model_name, num_classes=2, pretrained=False):
    """
    根据模型名称获取对应的模型
    
    Args:
        model_name (str): 模型名称
        num_classes (int): 分类数量
        pretrained (bool): 是否使用预训练模型
        model_path (str): 本地模型文件路径
        **kwargs: 其他参数
    
    Returns:
        nn.Module: 对应的模型
    """
    if model_name == 'alexnet':
        model = alexnet(num_classes=num_classes)

    elif model_name == 'densenet121':
        model = densenet121(num_classes=num_classes)

    elif model_name == 'densenet161':
        model = densenet161(num_classes=num_classes)

    elif model_name == 'densenet169':
        model = densenet169(num_classes=num_classes)

    elif model_name == 'densenet201':
        model = densenet201(num_classes=num_classes)

    elif model_name == 'googlenet':
        model = googlenet(num_classes=num_classes)

    elif model_name == 'resnet18':
        model = resnet18(num_classes=num_classes)

    elif model_name == 'resnet3d_18':
        model = resnet3d(18,n_input_channels=3,n_classes=2)

    elif model_name == 'resnet3d_101':
        model = resnet3d(101,n_input_channels=3,n_classes=2)

    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return model
