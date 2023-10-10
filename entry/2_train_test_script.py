
import os

import torch
from torch import nn, optim

from traffic_log.setLog import logger
from utils.setConfig import setup_config

from models.cnn_lstm import cnn_rnn
from train_valid.trainProcess import train_process
from train_valid.validateProcess import validate_process
from data_loader.dataLoader import data_loader
from data_loader.tensordata import get_tensor_data
from utils.helper import adjust_learning_rate, save_checkpoint

from utils.evaluate_tools import display_model_performance_metrics

# from torch.utils.tensorboard import SummaryWriter


def train_pipeline():
    cfg = setup_config()  # 获取 config 文件
    logger.info(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info('是否使用 GPU 进行训练, {}'.format(device))

    os.makedirs(cfg.train.model_dir, exist_ok=True)
    model_path = os.path.join(cfg.train.model_dir, cfg.train.model_name)  # 模型的路径
    num_classes = len(cfg.test.label2index)

    model = cnn_rnn(model_path, pretrained=cfg.test.pretrained, input_size=cfg.train.input_size,
                    hidden_size=cfg.train.hidden_size, num_layers=cfg.train.num_layers, bidirectional=cfg.train.Bi,
                    num_classes=num_classes).to(device)  # 定义模型)

    criterion = nn.CrossEntropyLoss()  # 定义损失函数
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)  # 定义优化器
    logger.info('成功初始化模型.')

    train_loader = data_loader(pcap_file=cfg.train.train_pay, seq_file=cfg.train.train_seq,
                               label_file=cfg.train.train_label,
                               batch_size=cfg.train.BATCH_SIZE)  # 获得 train dataloader
    test_loader = data_loader(pcap_file=cfg.train.test_pay, seq_file=cfg.train.test_seq,
                              label_file=cfg.train.test_label,
                              batch_size=cfg.train.BATCH_SIZE)  # 获得 train dataloader
    logger.info('成功加载数据集.')

    if cfg.test.evaluate:  # 是否只进行测试
        logger.info('进入测试模式.')
        validate_process(test_loader, model, criterion, device, 20)  # 总的一个准确率
        torch.cuda.empty_cache()  # 清除显存
        # 计算每个类别详细的准确率
        index2label = {j: i for i, j in cfg.test.label2index.items()}  # index->label 对应关系
        print(index2label)
        label_list = [index2label.get(i) for i in range(len(index2label))]  # 17 个 label 的标签
        pcap_data, label_data = get_tensor_data(pcap_file=cfg.train.test_ip_lengths, label_file=cfg.train.test_label,
                                                trimed_file_len=cfg.train.TRIMED_FILE_LEN)
        start_index = 0
        y_pred = None
        for i in list(range(100, 3800, 100)):
            data = pcap_data[start_index:i].reshape(-1, 256, 1)
            y_pred_batch = model(data.to(device))

            start_index = i
            if y_pred == None:
                y_pred = y_pred_batch.cpu().detach()
            else:
                y_pred = torch.cat((y_pred, y_pred_batch.cpu().detach()), dim=0)
                print(y_pred.shape)

        _, pred = y_pred.topk(1, 1, largest=True, sorted=True)

        Y_data_label = [index2label.get(i.tolist()) for i in label_data]  # 转换为具体名称
        pred_label = [index2label.get(i.tolist()) for i in pred.view(-1).cpu().detach()]

        Y_data_label = Y_data_label[:3700]
        display_model_performance_metrics(true_labels=Y_data_label, predicted_labels=pred_label, classes=label_list)
        return

    best_prec1 = 0
    # loss_writer = SummaryWriter(log_dir=os.path.join("./tensorboard", "loss"))
    # acc_writer = SummaryWriter(log_dir=os.path.join("./tensorboard", "acc"))

    for epoch in range(cfg.train.epochs):
        adjust_learning_rate(optimizer, epoch, cfg.train.lr)  # 动态调整学习率

        train_loss, train_acc = train_process(train_loader, model, criterion, optimizer, epoch, device,
                                              80)  # train for one epoch
        prec1, val_loss, val_acc = validate_process(test_loader, model, criterion, device,
                                                    20)  # evaluate on validation set

        # loss_writer.add_scalars("loss", {'train': train_loss, 'val': val_loss}, epoch)
        # acc_writer.add_scalars("train_acc", {'train': train_acc, 'val': val_acc}, epoch)

        # remember the best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        # 保存最优的模型
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict()
        }, is_best, model_path)

    # loss_writer.close()
    # acc_writer.close()
    logger.info('Finished! (*￣︶￣)')


if __name__ == "__main__":
    train_pipeline()  # 用于测试