import os
import torch
import sys

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.utils.get_model import get_model
from tqdm import tqdm
from loss import Classification_Loss, FocalLoss


class MedicalClassificationTrainer:
    def __init__(self, args, hparams):
        self.args = args
        self.hp = hparams
        self.device = torch.device(f"cuda:{self.hp.device_id}" if torch.cuda.is_available() else "cpu")

        # 初始化模型
        self.model = self._init_model(args.model, args.num_classes)
        # 初始化优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.init_lr)
        # 初始化学习率调度器
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.hp.scheduer_step_size,
            gamma=self.hp.scheduer_gamma
        )

        # 初始化损失函数
        self.criterion = self._init_criterion()

        # 设置CUDA相关参数
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = args.cudnn_enabled
        torch.backends.cudnn.benchmark = args.cudnn_benchmark

    def _init_model(self, model_name, num_classes):
        """
        根据参数初始化模型
        """
        model = get_model(model_name,num_classes=num_classes)  # 假设get_model根据hp中的参数返回对应模型
        model = torch.nn.DataParallel(model, device_ids=[self.hp.device_id])
        return model.to(self.device)

    def _init_criterion(self):
        """
        根据参数初始化损失函数
        这里可以选择不同的损失函数
        """

        # 交叉熵损失
        return torch.nn.CrossEntropyLoss().to(self.device)
        # focal损失
        # return FocalLoss().to(self.device)
        # 二分类交叉熵损失
        # return torch.nn.BCEWithLogitsLoss().to(self.device)


    def load_checkpoint(self):
        """
        加载检查点
        """
        if self.args.ckpt is not None:
            print("load model:", self.args.ckpt)
            ckpt_path = os.path.join(self.args.output_dir, self.args.latest_checkpoint_file)
            print(ckpt_path)

            ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)

            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optim"])

            # 将优化器状态转移到GPU
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)

            elapsed_epochs = ckpt.get("epoch", 0)
            if "scheduler" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler"])
        else:
            elapsed_epochs = 0

        return elapsed_epochs

    def train_one_epoch(self, model, optimizer, criterion, data_loader, device, epoch):
        model.train()
        loss_function = criterion
        accu_loss = torch.zeros(1).to(device)  # 累计损失
        accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
        optimizer.zero_grad()

        sample_num = 0
        data_loader = tqdm(data_loader, file=sys.stdout)
        for step, data in enumerate(data_loader):
            images, labels = data
            sample_num += images.shape[0]

            pred = model(images.to(device))
            pred_classes = torch.max(pred, dim=1)[1]
            accu_num += torch.eq(pred_classes, labels.to(device)).sum()

            loss = loss_function(pred, labels.to(device))
            loss.backward()
            accu_loss += loss.detach()

            data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                                   accu_loss.item() / (step + 1),
                                                                                   accu_num.item() / sample_num)

            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss)
                sys.exit(1)

            optimizer.step()
            optimizer.zero_grad()

        return accu_loss.item() / (step + 1), accu_num.item() / sample_num


# 使用示例：
# trainer = MedicalClassificationTrainer(args, hp)
# trainer.train(train_loader, val_loader)
# trainer.test(test_loader)
