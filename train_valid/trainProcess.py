

from utils.helper import AverageMeter, accuracy
from traffic_log.setLog import logger


def train_process(train_loader, model, criterion, optimizer, epoch, device, print_freq):
    """训练一个 epoch 的流程

    Args:
        train_loader (dataloader): [description]
        model ([type]): [description]
        criterion ([type]): [description]
        optimizer ([type]): [description]
        epoch (int): 当前所在的 epoch
        device (torch.device): 是否使用 gpu
        print_freq ([type]): [description]
    """
    losses = AverageMeter()  # 在一个 train loader 中的 loss 变化
    top1 = AverageMeter()  # 记录在一个 train loader 中的 accuracy 变化

    model.train()  # 切换为训练模型

    for i, (pay, seq, target) in enumerate(train_loader):
        # pay = pay.reshape(-1,256,1)
        pay = pay.to(device)
        seq = seq.to(device)
        target = target.to(device)

        output = model(pay,seq)  # 得到模型预测结果
        loss = criterion(output, target)  # 计算 loss

        # 计算准确率, 记录 loss 和 accuracy
        # print(pay.size(0))
        prec1 = accuracy(output.data, target)
        losses.update(loss.item(), pay.size(0))
        top1.update(prec1[0].item(), pay.size(0))

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}], Loss {loss.val:.4f} ({loss.avg:.4f}), Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), loss=losses, top1=top1))
    return losses.val, top1.val
