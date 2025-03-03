import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
import gc


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
  
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('closs', misc.SmoothedValue(window_size=1))
    metric_logger.add_meter('mae', misc.SmoothedValue(window_size=1))
    metric_logger.add_meter('corr', misc.SmoothedValue(window_size=1))
    lr = optimizer.param_groups[0]["lr"]
    metric_logger.update(lr=lr)
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))



    prefix_audio = torch.tensor(data_loader.dataset.tokenizer.encode("Audio: ", bos=False, eos=False), dtype=torch.int64)
    prefix_video = torch.tensor(data_loader.dataset.tokenizer.encode("Video: ", bos=False, eos=False), dtype=torch.int64)

    for data_iter_step, (examples, labels, values, example_mask, videos, audios) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        prefix_audio=prefix_audio.to(examples.device)
        prefix_video=prefix_video.to(examples.device)
        values = values.float()
        l1_loss, corr_loss = model(examples, labels, values, videos = videos, prefix_video = prefix_video, audios = audios, prefix_audio =  prefix_audio)
        c_loss = 0.9*l1_loss + 0.1*corr_loss
        loss = c_loss
        loss_value = loss.item()
        c_loss_value = c_loss.item()


        if torch.isnan(loss):
            print("NaN loss encountered. Skipping this batch.")
            del examples, labels, videos, audios, values, l1_loss, corr_loss, loss, c_loss
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            continue

        loss = loss/accum_iter

        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0,clip_grad=args.clip_grad)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(mae=l1_loss.item())
        metric_logger.update(corr = 1 - corr_loss.item())
        metric_logger.update(closs=c_loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        c_loss_value_reduce = misc.all_reduce_mean(c_loss_value)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('c_train_loss', c_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

        gc.collect()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def val_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, 
                    device: torch.device, epoch: int,
                    log_writer=None,
                    args=None):
  
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1
    metric_logger.add_meter('closs', misc.SmoothedValue(window_size=1))
    metric_logger.add_meter('mae', misc.SmoothedValue(window_size=1))
    metric_logger.add_meter('corr', misc.SmoothedValue(window_size=1))

    accum_iter = args.accum_iter

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))



    prefix_audio = torch.tensor(data_loader.dataset.tokenizer.encode("Audio: ", bos=False, eos=False), dtype=torch.int64)
    prefix_video = torch.tensor(data_loader.dataset.tokenizer.encode("Video: ", bos=False, eos=False), dtype=torch.int64)

    for data_iter_step, (examples, labels, values, example_mask, videos, audios) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        prefix_audio=prefix_audio.to(examples.device)
        prefix_video=prefix_video.to(examples.device)
        values = values.float()
        l1_loss, corr_loss = model(examples, labels, values, videos = videos, prefix_video = prefix_video, audios = audios, prefix_audio =  prefix_audio)
        c_loss = 0.9*l1_loss + 0.1*corr_loss
        loss = c_loss
        loss_value = loss.item()
        c_loss_value = c_loss.item()


        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        metric_logger.update(mae=l1_loss.item())
        metric_logger.update(corr = 1 - corr_loss.item())
        metric_logger.update(closs=c_loss_value)


        c_loss_value_reduce = misc.all_reduce_mean(c_loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('c_train_loss', c_loss_value_reduce, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}