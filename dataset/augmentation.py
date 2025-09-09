# 替换原有的数据增强定义部分
import monai.transforms as monai_transforms
from torchvision import transforms
# 定义2D和3D数据增强转换
def get_transforms(is_3d=False):
    if is_3d:
        # 3D数据增强转换
        data_transform = {
            "train": monai_transforms.Compose([
                monai_transforms.EnsureChannelFirstd(keys=["image"]),
                monai_transforms.Resized(keys=["image"], spatial_size=(224, 224, 224)),
                monai_transforms.RandRotated(keys=["image"], range_x=0.1, range_y=0.1, range_z=0.1, prob=0.5),
                monai_transforms.RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                monai_transforms.ScaleIntensityd(keys=["image"]),
                monai_transforms.ToTensord(keys=["image"])
            ]),
            "val": monai_transforms.Compose([
                monai_transforms.EnsureChannelFirstd(keys=["image"]),
                monai_transforms.Resized(keys=["image"], spatial_size=(224, 224, 224)),
                monai_transforms.ScaleIntensityd(keys=["image"]),
                monai_transforms.ToTensord(keys=["image"])
            ])
        }
    else:
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


