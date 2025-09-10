import csv
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class BaseDataset(Dataset):
    """
    超声心动图数据集的基类.
    处理数据集划分 (train/test) 和交叉验证折叠 (fold) 的通用逻辑.
    """
    def __init__(self, data_dir, metadata_path):
        super().__init__()
        self.data_dir = data_dir
        self.patients = []
        self._load_metadata(metadata_path)


    def _load_metadata(self, metadata_path):
        """从CSV文件中加载元数据."""
        with open(metadata_path) as mfile:
            reader = csv.DictReader(mfile)
            for row in reader:
                self.patients.append(row)


    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        raise NotImplementedError("子类必须实现 __getitem__ 方法")