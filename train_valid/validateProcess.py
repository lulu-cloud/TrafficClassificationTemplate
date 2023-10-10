
import torch

from utils.helper import AverageMeter, accuracy
from traffic_log.setLog import logger


def validate_process(val_loader, model, criterion, device, print_freq):
    losses = AverageMeter()
    top1 = AverageMeter()
    
    model.eval()  # switch to evaluate mode

    for i, (pay, seq, target) in enumerate(val_loader):
        # pay = pay.reshape(-1,256,1)
        pay = pay.to(device)
        seq = seq.to(device)
        target = target.to(device)

        with torch.no_grad():

            output = model(pay,seq)  # compute output
            loss = criterion(output, target)  # 计算验证集的 loss

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)
            losses.update(loss.item(), pay.size(0))
            top1.update(prec1[0].item(), pay.size(0))

            if (i+1) % print_freq == 0:
                logger.info('Test: [{0}/{1}], Loss {loss.val:.4f} ({loss.avg:.4f}), Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), loss=losses, top1=top1))

    logger.info(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg, losses.val, top1.val
