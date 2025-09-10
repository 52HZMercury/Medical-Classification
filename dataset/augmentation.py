# 替换原有的数据增强定义部分
from torchvision import transforms
import albumentations as A
# 定义2D和3D数据增强转换
def get_transforms(is_3d=True):
    if is_3d:
        # 暂时没有使用3D数据增强
        # 3D数据增强转换
        data_transform = {
            "train":
                A.Compose([
                A.RandomCrop(width=128, height=128),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.RandomGamma(p=0.2),
                ])
            ,
            "val":
                A.Compose([
                A.RandomCrop(width=128, height=128),
                ])
        }

    # 2D数据增强转换（原有的transforms）
    data_transform = {
        "train":
            transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        "val":
            transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }
    return data_transform


