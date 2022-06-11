import logging
import time

import torch

from lib.core.metrics.eval_metrics import Metrics
from lib.utils.utils import AverageMeter


def train_one_epoch(config, train_loader, model, criterion, optimizer, epoch):
    start = time.time()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    epoch_accuracy = AverageMeter()
    epoch_recall = AverageMeter()
    epoch_precision = AverageMeter()
    epoch_f1_score = AverageMeter()

    metrics = Metrics(config.MODEL.NUM_CLASSES)

    logging.info('=> switch to train mode')
    model.train()

    end = time.time()

    for i, (images, lables) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        images = images.cuda(non_blocking=True)
        lables = lables.cuda(non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, lables)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()

        # measure metrics and record loss
        losses.update(loss.item(), images.size(0))
        metrics.update(outputs, lables)
        accuracy, recall, precision, f1_score, _ = metrics.compute_metrics()
        epoch_accuracy.update(accuracy, images.size(0))
        epoch_recall.update(recall, images.size(0))
        epoch_precision.update(precision, images.size(0))
        epoch_f1_score.update(f1_score, images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # log info
        if i % config.PRINT_FREQ == 0:
            msg = (
                '=> Epoch[{0}][{1}/{2}]: '
                'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s) '
                'Speed {speed:.1f} samples/s '
                'Data {data_time.val:.3f}s ({data_time.avg:.3f}s) '
                'Loss {loss.val:.5f} ({loss.avg:.5f}) '
                'Accuracy {epoch_accuracy.val:.3f} ({epoch_accuracy.avg:.3f}) '
                'Precision {epoch_precision.val:.3f} ({epoch_precision.avg:.3f}) '
                'Recall {epoch_recall.val:.3f} ({epoch_recall.avg:.3f}) '
                'F1_Score {epoch_f1_score.val:.3f} ({epoch_f1_score.avg:.3f})'.format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    speed=images.size(0) / batch_time.val,
                    data_time=data_time,
                    loss=losses,
                    epoch_accuracy=epoch_accuracy,
                    epoch_precision=epoch_precision,
                    epoch_recall=epoch_recall,
                    epoch_f1_score=epoch_f1_score,
                )
            )
            logging.info(msg)
        metrics.reset()
        torch.cuda.synchronize()
    logging.info(
        '=> Epoch[{}] train end, duration: {:.2f}s'.format(epoch, time.time() - start)
    )
