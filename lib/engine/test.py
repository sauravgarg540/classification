import logging
import time

import torch

from lib.core.metrics.eval_metrics import Metrics
from utils.utils import AverageMeter


def test(config, val_loader, model, criterion, epoch):

    val_start = time.time()

    logging.info('=> Epoch[{}] validate start'.format(epoch))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    metrics = Metrics(config.MODEL.NUM_CLASSES)

    # switch to evaluate mode
    logging.info('=> switch to evaluate mode')
    model.eval()

    with torch.no_grad():
        end = time.time()
        for images, lables in val_loader:

            # measure data loading time
            data_time.update(time.time() - end)

            # compute output
            images = images.cuda(non_blocking=True)
            lables = lables.cuda(non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, lables)

            # measure metrics and record loss
            losses.update(loss.item(), images.size(0))
            metrics.update(outputs, lables)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        (
            accuracy,
            recall,
            precision,
            f1_score,
            confusion_matrix,
        ) = metrics.compute_metrics()

        msg = (
            '=> Validation: '
            'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s) '
            'Speed {speed:.1f} samples/s '
            'Data {data_time.val:.3f}s ({data_time.avg:.3f}s) '
            'Loss {loss.avg:.5f} '
            'Accuracy {epoch_accuracy.avg:.4f} '
            'Precision {epoch_precision.avg:.4f} '
            'Recall {epoch_recall.avg:.4f} '
            'F1_Score {epoch_f1_score.avg:.4f}'.format(
                batch_time=batch_time,
                speed=images.size(0) / batch_time.val,
                data_time=data_time,
                loss=losses,
                epoch_accuracy=accuracy,
                epoch_precision=precision,
                epoch_recall=recall,
                epoch_f1_score=f1_score,
            )
        )
        logging.info(msg)
    logging.info(
        '=> Epoch[{}] validate end, duration: {:.2f}s'.format(
            epoch, time.time() - val_start
        )
    )
    return accuracy, precision, recall, f1_score, confusion_matrix
