import os
import argparse
import torch
from evaluate import evaluate
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from config import hparams as hp
from torchvision import transforms
from dataset.data_splitter import read_split_dataset
from dataset.med_2d_dataset import MedDataSet
from dataset.med_3d_dataset import CAMUSDataset
from classification_trainer import MedicalClassificationTrainer
from dataset.augmentation import get_transforms

device_id = hp.device_id
os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
devicess = [device_id]
device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")


def train(args, trainer, train_loader, val_loader=None):
    # 初始化TensorBoard writer
    tb_writer = SummaryWriter(args.log_dir)
    """
    训练模型
    """
    # 初始化最佳指标值和模型路径
    best_metric_value = 0.0
    best_model_path = None

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = (trainer.train_one_epoch(model=trainer.model,
                                                         optimizer=trainer.optimizer,
                                                         criterion=trainer.criterion,
                                                         data_loader=train_loader,
                                                         device=device,
                                                         epoch=epoch))

        trainer.scheduler.step()

        # 每隔一个epoch进行验证
        if epoch % 1 == 0:
            # validate
            val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate(model=trainer.model,
                                                                            data_loader=val_loader,
                                                                            device=device,
                                                                            epoch=epoch)

            # 根据指定的指标确定当前值
            metric_values = {
                'val_acc': val_acc,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_f1': val_f1
            }
            current_metric_value = metric_values.get(hp.save_by_metric)  # 默认使用val_acc

            # 如果当前指标达到新高，则保存该模型
            if current_metric_value > best_metric_value:
                best_metric_value = current_metric_value

                # 删除之前指标最高的模型
                if best_model_path and os.path.exists(best_model_path):
                    os.remove(best_model_path)

                # 保存当前最佳模型
                best_model_path = os.path.join(args.log_dir, f"best_model_acc{val_acc:.4f}_epoch{epoch}.pth")
                torch.save({
                    'epoch': epoch,
                    'model': trainer.model.state_dict(),
                    'optimizer': trainer.optimizer.state_dict(),
                    'val_acc': val_acc
                }, best_model_path)
                print(f"Saved new best model with val_acc: {val_acc:.4f} at {best_model_path}")

            # 保存最新的模型
            latest_model_path = os.path.join(args.log_dir, 'checkpoint_latest.pt')
            torch.save({
                'epoch': epoch,
                'model': trainer.model.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
                'val_acc': val_acc
            }, latest_model_path)

            tags = ["train_loss", "train_acc", "learning_rate", "val_loss", "val_acc", "val_precision", "val_recall",
                    "val_f1"]
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], trainer.optimizer.param_groups[0]["lr"], epoch)
            tb_writer.add_scalar(tags[3], val_loss, epoch)
            tb_writer.add_scalar(tags[4], val_acc, epoch)
            tb_writer.add_scalar(tags[5], val_precision, epoch)
            tb_writer.add_scalar(tags[6], val_recall, epoch)
            tb_writer.add_scalar(tags[7], val_f1, epoch)

    tb_writer.close()


def test(args, model, test_loader):
    # 初始化TensorBoard writer用于测试日志
    tb_writer = SummaryWriter(args.log_dir)
    # 查找最佳模型文件
    best_model_files = [f for f in os.listdir(args.log_dir) if f.startswith("best_model_acc")]
    if best_model_files:
        # 按照准确率排序，选择最佳的模型
        best_model_file = sorted(best_model_files,
                                 key=lambda x: float(x.split("acc")[1].split("_")[0]),
                                 reverse=True)[0]
        best_model_path = os.path.join(args.log_dir, best_model_file)
        print("load best model:", best_model_path)
        ckpt = torch.load(best_model_path, map_location=lambda storage, loc: storage)
    else:
        # 如果没有找到最佳模型文件，回退到使用最新模型
        print("No best model found, loading latest checkpoint:", args.ckpt)
        print(os.path.join(args.log_dir, args.latest_checkpoint_file))
        ckpt = torch.load(os.path.join(args.log_dir, args.latest_checkpoint_file),
                          map_location=lambda storage, loc: storage)

    model.load_state_dict(ckpt["model"])

    model.cuda()
    model.eval()

    test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate(model=model,
                                                                         data_loader=test_loader,
                                                                         device=device,
                                                                         epoch=0)
    tags = ["test_loss", "test_acc", "test_precision", "test_recall", "test_f1"]
    tb_writer.add_scalar(tags[0], test_loss, 0)
    tb_writer.add_scalar(tags[1], test_acc, 0)
    tb_writer.add_scalar(tags[2], test_precision, 0)
    tb_writer.add_scalar(tags[3], test_recall, 0)
    tb_writer.add_scalar(tags[4], test_f1, 0)
    print("-----------------------------------------------------------------------")
    print("The final performance of the best_model on the test set is as follows:")
    print("Test loss:", test_loss)
    print("Test acc:", test_acc)
    print("Test precision:", test_precision)
    print("Test recall:", test_recall)
    print("Test f1:", test_f1)


def main(args):
    # 定义数据增强转换
    data_transform = get_transforms(args.is_3d)

    # 使用dataSplitter读取并分割数据集
    # train_data, val_data, test_data = read_split_dataset(args.data)
    # # 实例化2d的三个数据集 分别是训练 验证 测试
    # train_dataset = MedDataSet(data=train_data,transform=data_transform["train"])
    # val_dataset = MedDataSet(data=val_data,transform=data_transform["val"])
    # #测试集和验证数据集采取一样的处理方式
    # test_dataset = MedDataSet(data=test_data,transform=data_transform["val"])

    # 实例化3d的三个数据集 分别是训练 验证 测试
    train_dataset = CAMUSDataset(args.data, args.metadata_path, "A2C", [1, 2, 3, 4])
    val_dataset = CAMUSDataset(args.data, args.metadata_path, "A2C", [5])
    # 测试集和验证数据集采取一样的处理方式
    test_dataset = CAMUSDataset(args.data, args.metadata_path, "A2C", [5])

    batch_size = args.batch
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=nw)

    # 初始化模型 优化器 损失函数
    Trainer = MedicalClassificationTrainer(args, hp)
    train(args, Trainer, train_loader, val_loader)
    test(args, Trainer.model, test_loader)


if __name__ == '__main__':
    """
    Parse commandline arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=hp.data_dir, required=False, help='数据集文件夹')
    parser.add_argument('--is_3d', type=str, default=hp.is_3d, required=False, help='根据是否3d数据应用数据增强')
    parser.add_argument('--metadata_path', type=str, default=hp.metadata_path, required=False, help='元数据文件路径')
    parser.add_argument('--num_classes', type=str, default=hp.num_classes, required=False, help='分几类')
    parser.add_argument('--log_dir', type=str, default=hp.log_dir, required=False, help='Directory to save logs')

    # training
    training = parser.add_argument_group('training setup')
    training.add_argument('--epochs', type=int, default=hp.total_epochs, help='Number of total epochs to run')
    training.add_argument('--batch', type=int, default=hp.batch_size, help='batch-size')
    training.add_argument('--model', type=str, default=hp.model_name, help='训练模型类型')

    parser.add_argument("--ckpt", type=str, default=hp.ckpt, help="path to the checkpoints to resume training")
    parser.add_argument("--init-lr", type=float, default=hp.init_lr, help="learning rate")
    # TODO
    parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
    training.add_argument('--amp-run', action='store_true', help='Enable AMP')
    training.add_argument('--cudnn-enabled', default=True, help='Enable cudnn')
    training.add_argument('--cudnn-benchmark', default=True, help='Run cudnn benchmark')
    training.add_argument('--disable-uniform-initialize-bn-weight', action='store_true',help='disable uniform initialization of batchnorm layer weight')

    opt = parser.parse_args()
    main(opt)
