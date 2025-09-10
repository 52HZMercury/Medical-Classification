from glob import glob
from os.path import dirname, join, basename, isfile
import sys
sys.path.append('../')
import csv
import torch
import random
from PIL import Image


class MedDataSet(torch.utils.data.Dataset):
    """
    自定义数据集类，用于处理2d数据集
    """
    def __init__(self, data, transform=None):
        """
        初始化数据集
        Args:
            data: 从dataSplitter获取的包含图像路径和标签的字典列表
            transform: 数据增强转换
        """
        self.data = data
        self.transform = transform or self.default_transform()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        获取指定索引的数据项
        Args:
            index: 数据索引
        Returns:
            tuple: (图像数据, 标签)
        """
        subject = self.data[index]
        
        # 从文件路径加载图像
        img_path = subject['source']
        img = Image.open(img_path)
        
        # 确保图像是RGB格式
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # 获取标签
        label = subject['label']
        
        # 应用数据增强
        if self.transform:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
