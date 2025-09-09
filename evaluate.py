import torch
import sys
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, classification_report
import numpy as np


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    # 用于计算precision, recall, f1
    all_preds = []
    all_labels = []

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        # 收集预测结果和真实标签
        all_preds.extend(pred_classes.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        # 判断是验证还是测试（通过数据集大小或其他特征）
        # 这里使用一个简单的方法：如果epoch为0，则认为是测试
        if epoch == 0:
            prefix = "test"
        else:
            prefix = "valid"

        data_loader.desc = "[{} epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            prefix, epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num
        )

    # 计算loss accuracy precision recall f1
    loss = accu_loss.item() / (step + 1)
    accuracy = accu_num.item() / sample_num
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')


    print(f"Loss: {loss:.3f}, Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

    return loss, accuracy, precision, recall, f1
