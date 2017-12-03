#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import argparse
import os
import shutil
import time
import sys
import glob
import copy

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

from colorama import Fore
from importlib import import_module

import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import config
from dataloader import getDataloaders
from utils import (save_checkpoint, AverageMeter, adjust_learning_rate,
                   get_optimizer)

try:
    from tensorboard_logger import configure, log_value
except BaseException:
    configure = None

model_names = list(map(lambda n: os.path.basename(n)[:-3],
                       glob.glob('models/[A-Za-z]*.py')))

parser = argparse.ArgumentParser(
                description='Image classification PK main script')

exp_group = parser.add_argument_group('exp', 'experiment setting')
exp_group.add_argument('--save', default='save/default-{}'.format(time.time()),
                       type=str, metavar='SAVE',
                       help='path to the experiment logging directory'
                       '(default: save/default-CLOCKTIME)')
exp_group.add_argument('--resume', default='', type=str, metavar='PATH',
                       help='path to latest checkpoint (default: none)')
exp_group.add_argument('--evaluate', dest='evaluate', default='',
                       choices=['', 'val', 'test'],
                       help='eval mode: evaluate model on val/test set'
                       ' (default: training mode)')
exp_group.add_argument('-f', '--force', dest='force', action='store_true',
                       help='force to overwrite existing save path')
exp_group.add_argument('--print-freq', '-p', default=100, type=int,
                       metavar='N', help='print frequency (default: 100)')
exp_group.add_argument('--no_tensorboard', dest='tensorboard',
                       action='store_false',
                       help='do not use tensorboard_logger for logging')

# dataset related
data_group = parser.add_argument_group('data', 'dataset setting')
data_group.add_argument('--data', metavar='D', default='coco-debug',
                        choices=config.datasets.keys(),
                        help='datasets: ' +
                        ' | '.join(config.datasets.keys()) +
                        ' (default: coco-train-minival)')
data_group.add_argument('--data-root', metavar='DIR', default='data/COCO',
                        help='path to dataset (default: data)')
data_group.add_argument('-j', '--num-workers', dest='num_workers', default=4,
                        type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
data_group.add_argument('--normalized', action='store_true',
                        help='normalize the data into zero mean and unit std')

# model arch related
arch_group = parser.add_argument_group('arch', 'model architecture setting')
arch_group.add_argument('--arch', '-a', metavar='ARCH', default='faster_rcnn',
                        type=str, choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: faster_rcnn)')
arch_group.add_argument('--backbone', metavar='BACKBONE', default='resnet-50-c4',
                        type=str, help='backbone of RPN')
arch_group.add_argument('-d', '--depth', default=56, type=int, metavar='D',
                        help='depth (default=56)')
arch_group.add_argument('--drop-rate', default=0.0, type=float,
                        metavar='DROPRATE', help='dropout rate (default: 0.2)')
arch_group.add_argument('--death-mode', default='none',
                        choices=['none', 'linear', 'uniform'],
                        help='death mode (default: none)')
arch_group.add_argument('--death-rate', default=0.5, type=float,
                        help='death rate rate (default: 0.5)')
arch_group.add_argument('--bn-size', default=4, type=int,
                        metavar='B', help='bottle neck ratio for DenseNet'
                        ' (0 means dot\'t use bottle necks) (default: 4)')
arch_group.add_argument('--compression', default=0.5, type=float,
                        metavar='C', help='compression ratio for DenseNet'
                        ' (1 means dot\'t use compression) (default: 0.5)')
used to set the argument when to resume automatically
arch_resume_names = ['arch', 'depth', 'death_mode', 'death_rate', 'death_rate',
                     'bn_size', 'compression']

# training related
optim_group = parser.add_argument_group('optimization', 'optimization setting')
optim_group.add_argument('--niters', default=160000, type=int, metavar='N',
                         help='number of total iterations to run (default: 160000)')
optim_group.add_argument('--start-iter', default=1, type=int, metavar='N',
                         help='manual iter number (useful on restarts, default: 1)')
optim_group.add_argument('--eval-freq', default=1000, type=int, metavar='N',
                         help='number of iterations to run before evaluation (default: 1000)')
optim_group.add_argument('--patience', default=0, type=int, metavar='N',
                         help='patience for early stopping'
                         '(0 means no early stopping)')
optim_group.add_argument('-b', '--batch-size', default=16, type=int,
                         metavar='N', help='mini-batch size (default: 64)')
optim_group.add_argument('--optimizer', default='sgd',
                         choices=['sgd', 'rmsprop', 'adam'], metavar='N',
                         help='optimizer (default=sgd)')
optim_group.add_argument('--lr', '--learning-rate', default=0.02, type=float,
                         metavar='LR',
                         help='initial learning rate (default: 0.02)')
optim_group.add_argument('--decay_rate', default=0.1, type=float, metavar='N',
                         help='decay rate of learning rate (default: 0.1)')
optim_group.add_argument('--momentum', default=0.9, type=float, metavar='M',
                         help='momentum (default=0.9)')
optim_group.add_argument('--no_nesterov', dest='nesterov',
                         action='store_false',
                         help='do not use Nesterov momentum')
optim_group.add_argument('--alpha', default=0.001, type=float, metavar='M',
                         help='alpha for Adam (default: 0.001)')
optim_group.add_argument('--beta1', default=0.9, type=float, metavar='M',
                         help='beta1 for Adam (default: 0.9)')
optim_group.add_argument('--beta2', default=0.999, type=float, metavar='M',
                         help='beta2 for Adam (default: 0.999)')
optim_group.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                         metavar='W', help='weight decay (default: 1e-4)')


def get_model(arch, **kargs):
    m = import_module('models.' + arch)
    model = m.create_model(**kargs)
    # TODO: uncomment these and modify them
    # if arch.startswith('alexnet') or arch.startswith('vgg'):
    #     model.features = torch.nn.DataParallel(model.features)
    #     model.cuda()
    # else:
    #     model = torch.nn.DataParallel(model).cuda()
    model.cuda()
    return model


def main():
    # parse arg and start experiment
    global args
    best_ap = -1.
    best_iter = 0

    args = parser.parse_args()
    args.config_of_data = config.datasets[args.data]
    # args.num_classes = config.datasets[args.data]['num_classes']
    if configure is None:
        args.tensorboard = False
        print(Fore.RED +
              'WARNING: you don\'t have tesnorboard_logger installed' +
              Fore.RESET)

    # optionally resume from a checkpoint
    if args.resume:
        if args.resume and os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            old_args = checkpoint['args']
            print('Old args:')
            print(old_args)
            # set args based on checkpoint
            if args.start_iter <= 0:
                args.start_iter = checkpoint['iter'] + 1
            best_iter = args.start_iter - 1
            best_ap = checkpoint['best_ap']
            for name in arch_resume_names:
                if name in vars(args) and name in vars(old_args):
                    setattr(args, name, getattr(old_args, name))
            model = get_model(**vars(args))
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (iter {})"
                  .format(args.resume, checkpoint['iter']))
        else:
            print(
                "=> no checkpoint found at '{}'".format(
                    Fore.RED +
                    args.resume +
                    Fore.RESET),
                file=sys.stderr)
            return
    else:
        # create model
        print("=> creating model '{}'".format(args.arch))
        model = get_model(**vars(args))

    # cudnn.benchmark = True
    cudnn.enabled = False

    # create dataloader
    if args.evaluate == 'val':
        train_loader, val_loader, test_loader = getDataloaders(
            splits=('val'), **vars(args))
        validate(val_loader, model, best_iter)
        return
    elif args.evaluate == 'test':
        train_loader, val_loader, test_loader = getDataloaders(
            splits=('test'), **vars(args))
        validate(test_loader, model, best_iter)
        return
    else:
        train_loader, val_loader, test_loader = getDataloaders(
            splits=('train', 'val'), **vars(args))

    # define optimizer
    optimizer = get_optimizer(model, args)

    # check if the folder exists
    if os.path.exists(args.save):
        print(Fore.RED + args.save + Fore.RESET
              + ' already exists!', file=sys.stderr)
        if not args.force:
            ans = input('Do you want to overwrite it? [y/N]:')
            if ans not in ('y', 'Y', 'yes', 'Yes'):
                os.exit(1)
        print('remove existing ' + args.save)
        shutil.rmtree(args.save)
    os.makedirs(args.save)
    print('create folder: ' + Fore.GREEN + args.save + Fore.RESET)

    # copy code to save folder
    if args.save.find('debug') < 0:
        shutil.copytree(
            '.',
            os.path.join(
                args.save,
                'src'),
            symlinks=True,
            ignore=shutil.ignore_patterns(
                '*.pyc',
                '__pycache__',
                '*.path.tar',
                '*.pth',
                '*.ipynb',
                '.*',
                'data',
                'save',
                'save_backup'))

    # set up logging
    global log_print, f_log
    f_log = open(os.path.join(args.save, 'log.txt'), 'w')

    def log_print(*args):
        print(*args)
        print(*args, file=f_log)
    log_print('args:')
    log_print(args)
    print('model:', file=f_log)
    print(model, file=f_log, flush=True)
    # log_print('model:')
    # log_print(model)
    # log_print('optimizer:')
    # log_print(vars(optimizer))
    log_print('# of params:',
              str(sum([p.numel() for p in model.parameters()])))
    torch.save(args, os.path.join(args.save, 'args.pth'))
    scores = ['iter\tlr\ttrain_loss\tval_ap']
    if args.tensorboard:
        configure(args.save, flush_secs=5)

    for i in range(args.start_iter, args.niters + 1, args.eval_freq):
        # print('iter {:3d} lr = {:.6e}'.format(i, lr))
        # if args.tensorboard:
        #     log_value('lr', lr, i)

        # train for args.eval_freq iterations
        train_loss = train(train_loader, model, optimizer,
                           i, args.eval_freq)
        i += args.eval_freq - 1

        # evaluate on validation set
        val_ap = validate(val_loader, model, i)

        # save scores to a tsv file, rewrite the whole file to prevent
        # accidental deletion
        scores.append(('{}\t{}' + '\t{:.4f}' * 2)
                      .format(i, lr, train_loss, val_ap))
        with open(os.path.join(args.save, 'scores.tsv'), 'w') as f:
            print('\n'.join(scores), file=f)

        # remember best err@1 and save checkpoint
        # TODO: change this
        is_best = val_ap > best_ap
        if is_best:
            best_ap = val_ap
            best_iter = i
            print(Fore.GREEN + 'Best var_err1 {}'.format(best_ap) +
                  Fore.RESET)
        save_checkpoint({
            'args': args,
            'iter': i,
            'best_iter': best_iter,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_ap': best_ap,
        }, is_best, args.save)
        if not is_best and i - best_iter >= args.patience > 0:
            break
    print('Best val_ap: {:.4f} at iter {}'.format(best_ap, best_iter))


def train(train_loader, model, optimizer, start_iter, num_iters):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_losses = AverageMeter()
    rpn_losses = AverageMeter()
    odn_losses = AverageMeter()
    rpn_ce_losses = AverageMeter()
    rpn_box_losses = AverageMeter()
    odn_ce_losses = AverageMeter()
    odn_box_losses = AverageMeter()

    # switch to train mode
    end_iter = start_iter + num_iters - 1
    model.train()

    end = time.time()
    # for i in range(start_iter, start_iter + num_iters):
    for i, (inputs, anns) in enumerate(train_loader):
        i += start_iter
        # get minibatch
        # inputs, anns = next(train_loader)
        lr = adjust_learning_rate(optimizer, args.lr, args.decay_rate,
                                  i, args.niters)  # TODO: add custom
        # measure data loading time
        data_time.update(time.time() - end)

        optimizer.zero_grad()
        # forward images one by one (TODO: support batch mode later, or
        # multiprocess)
        for j, input in enumerate(inputs):
            input_anns = anns[j]  # anns of this input
            if len(input_anns) == 0:
                continue
            gt_bbox = np.vstack([ann['bbox'] + [ann['ordered_id']] for ann in input_anns])
            im_info= [[input.size(1), input.size(2),
                        input_anns[0]['scale_ratio']]]
            input_var= torch.autograd.Variable(input.unsqueeze(0).cuda(),
                                 requires_grad=False)

            cls_prob, bbox_pred, rois= model(input_var, im_info, gt_bbox)
            loss= model.loss
            loss.backward()
            # record loss
            total_losses.update(loss.data[0], input_var.size(0))
            rpn_losses.update(model.rpn.loss.data[0], input_var.size(0))
            rpn_ce_losses.update(
                model.rpn.cross_entropy.data[0], input_var.size(0))
            rpn_box_losses.update(
                model.rpn.loss_box.data[0], input_var.size(0))
            odn_losses.update(model.odn.loss.data[0], input_var.size(0))
            odn_ce_losses.update(
                model.odn.cross_entropy.data[0], input_var.size(0))
            odn_box_losses.update(
                model.odn.loss_box.data[0], input_var.size(0))

        # do SGD step
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.print_freq > 0 and (i + 1) % args.print_freq == 0:
            print('iter: [{0}] '
                  'Time {batch_time.val:.3f} '
                  'Data {data_time.val:.3f} '
                  'Loss {total_losses.val:.4f} '
                  'RPN {rpn_losses.val:.4f} '
                  '{rpn_ce_losses.val:.4f} '
                  '{rpn_box_losses.val:.4f} '
                  'ODN {odn_losses.val:.4f} '
                  '{odn_ce_losses.val:.4f} '
                  '{odn_box_losses.val:.4f} '
                  .format(i, batch_time=batch_time,
                          data_time=data_time,
                          total_losses=total_losses,
                          rpn_losses=rpn_losses,
                          rpn_ce_losses=rpn_ce_losses,
                          rpn_box_losses=rpn_box_losses,
                          odn_losses=odn_losses,
                          odn_ce_losses=odn_ce_losses,
                          odn_box_losses=odn_box_losses))

        del inputs
        del anns
        if i == end_iter:
            break

    print('iter: [{0}-{1}] '
          'Time {batch_time.avg:.3f} '
          'Data {data_time.avg:.3f} '
          'Loss {total_losses.avg:.4f} '
          'RPN {rpn_losses.avg:.4f} '
          '{rpn_ce_losses.avg:.4f} '
          '{rpn_box_losses.avg:.4f} '
          'ODN {odn_losses.avg:.4f} '
          '{odn_ce_losses.avg:.4f} '
          '{odn_box_losses.avg:.4f} '
          .format(start_iter, end_iter,
                  batch_time=batch_time,
                  data_time=data_time,
                  total_losses=total_losses,
                  rpn_losses=rpn_losses,
                  rpn_ce_losses=rpn_ce_losses,
                  rpn_box_losses=rpn_box_losses,
                  odn_losses=odn_losses,
                  odn_ce_losses=odn_ce_losses,
                  odn_box_losses=odn_box_losses))

    if args.tensorboard:
        log_value('train_total_loss', total_losses.avg, end_iter)
        log_value('train_rpn_loss', rpn_losses.avg, end_iter)
        log_value('train_rpn_ce_loss', rpn_ce_losses.avg, end_iter)
        log_value('train_rpn_box_loss', rpn_box_losses.avg, end_iter)
        log_value('train_odn_loss', odn_losses.avg, end_iter)
        log_value('train_odn_ce_loss', odn_ce_losses.avg, end_iter)
        log_value('train_odn_box_loss', odn_box_losses.avg, end_iter)
    return total_losses.avg


def validate(val_loader, model, i, silence=False):
    batch_time = AverageMeter()
    coco_gt = val_loader.dataset.coco
    coco_pred = COCO()
    coco_pred.dataset['images'] = [img for img in coco_gt.datasets['images']]
    coco_pred.dataset['categories'] = copy.deepcopy(coco_gt.dataset['categories'])
    id = 0

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (inputs, anns) in enumerate(val_loader):

        # forward images one by one (TODO: support batch mode later, or
        # multiprocess)
        for j, input in enumerate(inputs):
            input_anns= anns[j] # anns of this input
            gt_bbox= np.vstack([ann['bbox'] + [ann['ordered_id']] for ann in input_anns])
            im_info= [[input.size(1), input.size(2),
                        input_anns[0]['scale_ratio']]]
            input_var= Variable(input.unsqueeze(0),
                                 requires_grad=False).cuda()

            cls_prob, bbox_pred, rois = model(input_var, im_info)
            scores, pred_boxes = model.interpret_outputs(cls_prob, bbox_pred, rois, im_info)
            print(scores, pred_boxes)
            # for i in range(scores.shape[0]):


        # measure elapsed time
        batch_time.update(time.time() - end)
        end= time.time()

    coco_pred.createIndex()
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds= sorted(coco_gt.getImgIds())
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    print('iter: [{0}] '
          'Time {batch_time.avg:.3f} '
          'Val Stats: {1}'
          .format(i, coco_eval.stats,
                  batch_time=batch_time))

    return coco_eval.stats[0]


if __name__ == '__main__':
    main()
