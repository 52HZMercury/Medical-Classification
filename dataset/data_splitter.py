from pathlib import Path
import random
import sys
sys.path.append('../')
from config import hparams as hp


def read_2d_dataset(data_dir):
    """
    读取数据目录中的所有图像数据并创建subjects列表

    Args:
        data_dir: 数据根目录，其中包含各个类别的子目录

    Returns:
        list: 包含所有数据subjects的列表
    """
    # 支持的图像扩展名
    supported_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}

    # 获取所有类别目录
    data_path = Path(data_dir)
    class_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    
    # 按类别收集所有图像路径
    data = []
    for class_idx, class_dir in enumerate(class_dirs):
        print(f"Processing class {class_idx}: {class_dir.name}")
        # 获取所有文件并根据扩展名过滤
        all_files = list(class_dir.iterdir())
        image_files = [f for f in all_files if f.is_file() and f.suffix in supported_extensions]
        image_files = sorted(image_files)
        print(f"  Found {len(image_files)} images")

        for image_file in image_files:
            # 使用字典存储图片路径source和标签label
            subject = {
                'source': str(image_file),
                'label': class_idx
            }
            # 将subject添加到data列表中
            data.append(subject)
    
    print(f"Dataset reading completed:")
    print(f"Total samples: {len(data)}")
    
    return data

def split_dataset(data, train_ratio=hp.train_ratio, val_ratio=hp.val_ratio, test_ratio=hp.test_ratio):
    """
    将数据集划分为训练集、验证集和测试集

    Args:
        data: 包含所有数据subjects的列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例

    Returns:
        tuple: (train_subjects, val_subjects, test_subjects) 三个数据集的subjects列表
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "数据集划分比例之和必须为1"
    
    # 打乱数据
    random.shuffle(data)
    
    # 按比例划分数据集
    n_total = len(data)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # 划分数据集
    train_subjects = data[:n_train]
    val_subjects = data[n_train:n_train + n_val]
    test_subjects = data[n_train + n_val:]

    print(f"Dataset split completed:")
    print(f"  Train set: {len(train_subjects)} samples")
    print(f"  Validation set: {len(val_subjects)} samples")
    print(f"  Test set: {len(test_subjects)} samples")
    
    return train_subjects, val_subjects, test_subjects

def read_split_dataset(data_dir, train_ratio=hp.train_ratio, val_ratio=hp.val_ratio, test_ratio=hp.test_ratio):
    """
    将数据集划分为训练集、验证集和测试集

    Args:
        data_dir: 数据根目录，其中包含各个类别的子目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例

    Returns:
        tuple: (train_subjects, val_subjects, test_subjects) 三个数据集的subjects列表
    """
    # 读取数据
    data = read_2d_dataset(data_dir)
    
    # 划分数据集
    return split_dataset(data, train_ratio, val_ratio, test_ratio)


class MedDataSplitter:
    def __init__(self, data_dir, train_ratio=hp.train_ratio, val_ratio=hp.val_ratio, test_ratio=hp.test_ratio):
        self.data_dir = data_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def read_2d_dataset(self):
        return read_2d_dataset(self.data_dir)

    def split_dataset(self, data):
        return split_dataset(data, self.train_ratio, self.val_ratio, self.test_ratio)

    def read_split_dataset(self):
        return read_split_dataset(self.data_dir, self.train_ratio, self.val_ratio, self.test_ratio)
