# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os
import h5py
import numpy as np

import torch

from core.config import get_model_name
from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images

logger = logging.getLogger(__name__)


def routing(raw_features, aggre_features, is_aggre, meta):
    if not is_aggre:
        return raw_features

    output = []
    for r, a, m in zip(raw_features, aggre_features, meta):
        view = torch.zeros_like(a)
        batch_size = a.size(0)
        for i in range(batch_size):
            s = m['source'][i]
            view[i] = a[i] if s == 'h36m' else r[i]
        output.append(view)
    return output


def train(config, data, model, criterion, optim, epoch, output_dir,
          writer_dict):
    is_aggre = config.NETWORK.AGGRE
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, weight, meta) in enumerate(data):
        data_time.update(time.time() - end)

        raw_features, aggre_features = model(input)
        output = routing(raw_features, aggre_features, is_aggre, meta)

        loss = 0
        target_cuda = []
        for t, w, o in zip(target, weight, output):
            t = t.cuda(non_blocking=True)
            w = w.cuda(non_blocking=True)
            target_cuda.append(t)
            loss += criterion(o, t, w)
        target = target_cuda

        if is_aggre:
            for t, w, r in zip(target, weight, raw_features):
                t = t.cuda(non_blocking=True)
                w = w.cuda(non_blocking=True)
                loss += criterion(r, t, w)

        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.update(loss.item(), len(input) * input[0].size(0))

        nviews = len(output)
        acc = [None] * nviews
        cnt = [None] * nviews
        pre = [None] * nviews
        for j in range(nviews):
            _, acc[j], cnt[j], pre[j] = accuracy(
                output[j].detach().cpu().numpy(),
                target[j].detach().cpu().numpy())
        acc = np.mean(acc)
        cnt = np.mean(cnt)
        avg_acc.update(acc, cnt)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            gpu_memory_usage = torch.cuda.memory_allocated(0)
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t' \
                  'Memory {memory:.1f}'.format(
                      epoch, i, len(data), batch_time=batch_time,
                      speed=len(input) * input[0].size(0) / batch_time.val,
                      data_time=data_time, loss=losses, acc=avg_acc, memory=gpu_memory_usage)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', avg_acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            for k in range(len(input)):
                view_name = 'view_{}'.format(k + 1)
                prefix = '{}_{}_{:08}'.format(
                    os.path.join(output_dir, 'train'), view_name, i)
                save_debug_images(config, input[k], meta[k], target[k],
                                  pre[k] * 4, output[k], prefix)


def validate(config,
             loader,
             dataset,
             model,
             criterion,
             output_dir,
             writer_dict=None):

    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_acc = AverageMeter()

    nsamples = len(dataset) * 4
    is_aggre = config.NETWORK.AGGRE
    njoints = config.NETWORK.NUM_JOINTS
    height = int(config.NETWORK.HEATMAP_SIZE[0])
    width = int(config.NETWORK.HEATMAP_SIZE[1])
    all_preds = np.zeros((nsamples, njoints, 3), dtype=np.float32)
    all_heatmaps = np.zeros(
        (nsamples, njoints, height, width), dtype=np.float32)

    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, weight, meta) in enumerate(loader):
            raw_features, aggre_features = model(input)
            output = routing(raw_features, aggre_features, is_aggre, meta)

            loss = 0
            target_cuda = []
            for t, w, o in zip(target, weight, output):
                t = t.cuda(non_blocking=True)
                w = w.cuda(non_blocking=True)
                target_cuda.append(t)
                loss += criterion(o, t, w)

            if is_aggre:
                for t, w, r in zip(target, weight, raw_features):
                    t = t.cuda(non_blocking=True)
                    w = w.cuda(non_blocking=True)
                    loss += criterion(r, t, w)
            target = target_cuda

            nimgs = len(input) * input[0].size(0)
            losses.update(loss.item(), nimgs)

            nviews = len(output)
            acc = [None] * nviews
            cnt = [None] * nviews
            pre = [None] * nviews
            for j in range(nviews):
                _, acc[j], cnt[j], pre[j] = accuracy(
                    output[j].detach().cpu().numpy(),
                    target[j].detach().cpu().numpy())
            acc = np.mean(acc)
            cnt = np.mean(cnt)
            avg_acc.update(acc, cnt)

            batch_time.update(time.time() - end)
            end = time.time()

            preds = np.zeros((nimgs, njoints, 3), dtype=np.float32)
            heatmaps = np.zeros(
                (nimgs, njoints, height, width), dtype=np.float32)
            for k, o, m in zip(range(nviews), output, meta):
                pred, maxval = get_final_preds(config,
                                               o.clone().cpu().numpy(),
                                               m['center'].numpy(),
                                               m['scale'].numpy())
                pred = pred[:, :, 0:2]
                pred = np.concatenate((pred, maxval), axis=2)
                preds[k::nviews] = pred
                heatmaps[k::nviews] = o.clone().cpu().numpy()

            all_preds[idx:idx + nimgs] = preds
            all_heatmaps[idx:idx + nimgs] = heatmaps
            idx += nimgs

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(loader), batch_time=batch_time,
                          loss=losses, acc=avg_acc)
                logger.info(msg)

                for k in range(len(input)):
                    view_name = 'view_{}'.format(k + 1)
                    prefix = '{}_{}_{:08}'.format(
                        os.path.join(output_dir, 'validation'), view_name, i)
                    save_debug_images(config, input[k], meta[k], target[k],
                                      pre[k] * 4, output[k], prefix)

        # save heatmaps and joint locations
        u2a = dataset.u2a_mapping
        a2u = {v: k for k, v in u2a.items() if v != '*'}
        a = list(a2u.keys())
        u = np.array(list(a2u.values()))

        save_file = config.TEST.HEATMAP_LOCATION_FILE
        file_name = os.path.join(output_dir, save_file)
        file = h5py.File(file_name, 'w')
        file['heatmaps'] = all_heatmaps[:, u, :, :]
        file['locations'] = all_preds[:, u, :]
        file['joint_names_order'] = a
        file.close()

        name_value, perf_indicator = dataset.evaluate(all_preds)
        names = name_value.keys()
        values = name_value.values()
        num_values = len(name_value)
        _, full_arch_name = get_model_name(config)
        logger.info('| Arch ' +
                    ' '.join(['| {}'.format(name) for name in names]) + ' |')
        logger.info('|---' * (num_values + 1) + '|')
        logger.info('| ' + full_arch_name + ' ' +
                    ' '.join(['| {:.3f}'.format(value) for value in values]) +
                    ' |')

    return perf_indicator


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
