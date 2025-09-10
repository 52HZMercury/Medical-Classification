import torch
from pathlib import Path
from .base import BaseDataset


class CAMUSDataset(BaseDataset):
    """CAMUS 数据集"""

    def __init__(self, data_dir, metadata_path, view, fold=[5]):
        """添加fold参数用于过滤患者数据"""
        super().__init__(data_dir, metadata_path)
        self.view = view

        # 支持单个fold值或fold值列表
        if not isinstance(fold, list):
            fold = [fold]
        # 过滤出fold在指定列表中的患者数据
        self.patients = [patient for patient in self.patients if int(patient['fold']) in fold]

    def __getitem__(self, idx):
        patient_info = self.patients[idx]
        name = patient_info['Number']

        if self.view == "both":
            a2c_path = Path(self.data_dir) / 'A2C' / f"{name}.pt"
            a4c_path = Path(self.data_dir) / 'A4C' / f"{name}.pt"
            a2c_tensor = torch.load(a2c_path)
            a4c_tensor = torch.load(a4c_path)
            label = int(patient_info["Both"])
            # 返回样本ID
            return a2c_tensor, a4c_tensor, label
        else:
            view_path = Path(self.data_dir) / self.view / f"{name}.pt"
            video_tensor = torch.load(view_path, weights_only=True)
            label = int(patient_info[self.view])
            # 返回样本ID
            return video_tensor, label
