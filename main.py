from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import time
import cv2
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torchvision.models as models
from torch.autograd import Variable
from torch.utils.data import DataLoader

from libs.data_factory.data_factory import KeyPointDataset, KeyPointDataset_test
from libs.net.build_network import Network

from utils import *

# try:
# 	from tensorboard_logger import configure, log_value
# except BaseException:
# 	configure = None

img_error_id = ['97fb2cb75320cb0094681715dbb5aa7a13b27fb7',
				'bc4f5b8d1daef85d542a4dee4843c045d9511e9f',
				'9b4694e434c41e7aab3b921b7397293d68311a61',
				'ff585c6fbdb0672e0173afa08e689a98af22aa82',
				'c298cd4a867943e79e03ba92b52204e19c0cbe16',
				'46d05fc2173ae77970ba1eb33107092021753654',
				'9b1d18cae697da8160beed40834ee3b1bfb7386e',
				'd39b35193ef4d7f587ffe0cb4de9d2203b59fa63',
				'aa8e4ce1f69018eaeebac4fa8714a3c00ee85cba',
				'fd0a492dd5aa8d9bd73758f323bbbd1d88e2b03b',
				'b125b9cda788d1a02a11131f7aa1b0f835e13cbf',
				'e622c27c0760ed757e7f60b0fac37595ec538506',
				'f94c9f8d14f1c432e38b282012a570e9504c1239',
				'ba8ca016b0000b99f44176cb5c2636a951796621',
				'3f8957d7948790c29886f29e27e3809d8acd3ccc',
				'6805fee7416a6bc5003d291dc06956d5a5b06dc3',
				'daae63a17f06df617d7681d78968dc856685c1d4',
				'aee27632b9990f08cc39da6c8ce595544de96d16',
				'f3f1402e49251ddbc079ae208fa80ae6036eda94',
				'3aef0e2d3a64f2d45b2f0b2b4d38d202aff098cf',
				'89269460a718cd1902c525084d0ba2424ad1a348',
				'977e1f4381be79089a943a594d02b1f8a875682b']

model_names = sorted(name for name in models.__dict__
					 if name.islower() and not name.startswith("__")
					 and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch KeyPoint Training')
dir_group = parser.add_argument_group('dir', 'dataset path')
dir_group.add_argument('--train_dir',metavar='TRAIN_DIR',
					   default='/data/KeyPoints/keypointdata/ai_challenger/ai_challenger_keypoint_train_20170902',
					   type=str,help='path to the train data')
dir_group.add_argument('--train_img_dir',metavar='TRAIN_IMG_DIR',
					   default='/data/KeyPoints/keypointdata/ai_challenger/ai_challenger_keypoint_train_20170902/keypoint_train_images_20170902',
					   type=str,help='path to the train image')
dir_group.add_argument('--train_anno_dir',metavar='TRAIN_ANNO_DIR',
					   default='/data/KeyPoints/keypointdata/ai_challenger/ai_challenger_keypoint_train_20170902/keypoint_train_annotations_20170909.json',
					   type=str,help='path to the train annotations')
dir_group.add_argument('--val_img_dir',metavar='VAL_IMG_DIR',
					   default='/data/KeyPoints/keypointdata/ai_challenger/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911',
					   type=str,help='path to the train annotations')
dir_group.add_argument('--val_anno_dir',metavar='VAL_ANNO_DIR',
					   default='/data/KeyPoints/keypointdata/ai_challenger/ai_challenger_keypoint_validation_20170911/keypoint_validation_annotations_20170911.json',
					   type=str,help='path to the train annotations')
dir_group.add_argument('--test_img_dir',metavar='TEST_IMG_DIR',
					   default='/data/KeyPoints/keypointdata/ai_challenger/ai_challenger_keypoint_test_a_20170923/keypoint_test_a_images_20170923',
					   type=str,help='path to the train annotations')
dir_group.add_argument('--resume',metavar='CHECKPOINT',default='/home/mtm/output/save/mask_4092000_6/checkpoint.pth.tar',
					   type=str,help='path to latest checkpoint (default: none)')
# dir_group.add_argument('--resume',metavar='CHECKPOINT',default='',
# 					   type=str,help='path to latest checkpoint (default: none)')
dir_group.add_argument('--save', default='/home/mtm/output/save/',
                       type=str, metavar='SAVE',
                       help='path to the experiment logging directory'
                       '(default: save/default-CLOCKTIME)')


exp_group = parser.add_argument_group('exp', 'experiment setting')
exp_group.add_argument('--print-freq', '-p', default=50, type=int,
					   metavar='N', help='print frequency (default: 100)')
exp_group.add_argument('--no_tensorboard', dest='tensorboard',
					   action='store_false',
					   help='do not use tensorboard_logger for logging')

optim_group = parser.add_argument_group('optimization', 'optimization setting')
optim_group.add_argument('--epochs', default=5000000, type=int, metavar='N',
						 help='number of total epochs to run (default: 160000)')
optim_group.add_argument('--start-epoch', default=0, type=int, metavar='N',
						 help='manual epoch number (useful on restarts, default: 0)')
optim_group.add_argument('--save-freq', default=2000, type=int, metavar='N',
						 help='number of iterations to run before evaluation (default: 1000)')
optim_group.add_argument('--patience', default=0, type=int, metavar='N',
						 help='patience for early stopping (0 means no early stopping)')
optim_group.add_argument('-b', '--batch-size', default=1, type=int,
						 metavar='N', help='mini-batch size (default: 1)')
optim_group.add_argument('--optimizer', default='adam',
						 choices=['sgd', 'rmsprop', 'adam'], metavar='N',
						 help='optimizer (default=adam)')
optim_group.add_argument('--lr', '--learning-rate', default=0.000008, type=float,
						 metavar='LR',
						 help='initial learning rate (default: 0.02)')
optim_group.add_argument('--decay_rate', default=0.5, type=float, metavar='N',
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

parser.add_argument('--pretrained', dest='pretrained', action='store_true',
					default=True,
                    help='use pre-trained model')

parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
					choices=model_names,
					help='model architecture: ' +
						 ' | '.join(model_names) +
						 ' (default: resnet50)')

def main():
	global args, best_prec1
	args = parser.parse_args()
	best_iter = 0
	best_ap = -1.

	# if configure is None:
	# 	args.tensorboard = False
	# 	print(Fore.RED +
	# 		  'WARNING: you don\'t have tesnorboard_logger installed' +
	# 		  Fore.RESET)

	# create model
	if args.pretrained:
		print("=> using pre-trained model '{}'".format(args.arch))
		pretrain_model = models.__dict__[args.arch](pretrained=True)
	else:
		print("=> creating model '{}'".format(args.arch))
		pretrain_model = models.__dict__[args.arch]()

	model = Network(pretrain_model)
	#model = torch.nn.DataParallel(model).cuda()
	model = model.cuda()


	optimizer = get_optimizer(model, args)
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			args.start_epoch = checkpoint['epoch']
			args.arch = checkpoint['arch']
			#best_prec1 = checkpoint['best_prec1']

			pretrained_dict = checkpoint['state_dict']
			model_dict = model.state_dict()

			# 1. filter out unnecessary keys
			pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
			# 2. overwrite entries in the existing state dict
			model_dict.update(pretrained_dict)
			# 3. load the new state dict
			model.load_state_dict(model_dict)




			optimizer.load_state_dict(checkpoint['optimizer'])
			print("=> loaded checkpoint '{}' (epoch {})"
				  .format(args.resume, checkpoint['epoch']))
		else:
			print("=> no checkpint found at '{}'".format(args.resume))

	train_dataset = KeyPointDataset(
		json_file='/data/KeyPoints/keypointdata/ai_challenger/ai_challenger_keypoint_train_20170902/keypoint_train_annotations_20170909.json',
		image_dir='/data/KeyPoints/keypointdata/ai_challenger/ai_challenger_keypoint_train_20170902/keypoint_train_images_20170902'
	)
	train_loader = DataLoader(train_dataset,
							  batch_size=1,
							  shuffle=True,
							  num_workers=4)

	for epoch in range(args.start_epoch, args.epochs+1, args.save_freq):
		adjust_learning_rate(optimizer, args.lr, args.decay_rate, epoch, args.epochs)

		train_loss, output = train(train_loader, model, optimizer, epoch, args.save_freq)
		epoch += args.save_freq - 1

		save_({
			'epoch': epoch + 1,
			'args': args,
			'arch': args.arch,
			'state_dict': model.state_dict(),
			'optimizer': optimizer.state_dict(),
		},args.save, output)

	# val_dataset = KeyPointDataset(
	# 	json_file= args.val_anno_dir,
	# 	image_dir= args.val_img_dir
	# )
	#
	# val_loader = DataLoader(val_dataset,
	# 						  batch_size=1,
	# 						  shuffle=True,
	# 						  num_workers=4)
	#
	#
	#
	# test(val_loader, model)


	# test_dataset = KeyPointDataset_test(
	# 	image_dir = args.test_img_dir
	# )
	#
	# test_loader = DataLoader(test_dataset,
	# 						 batch_size=1,
	# 						 shuffle=True,
	# 						 num_workers=4)
	#
	# test(test_loader, model)











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
	mask_losses = AverageMeter()
	output = {}
	end_iter = start_iter + num_iters - 1
	# switch to train mode
	model.train()

	end = time.time()

	# get minibatch
	for i, sample in enumerate(train_loader):
		i += start_iter
		# measure data loading time
		data_time.update(time.time() - end)

		optimizer.zero_grad()

		image_id = sample['image_id']
		if image_id in img_error_id:
			print('drop: ', image_id)
			continue
		image = Variable(sample['image_t']).cuda()
		gt_boxes = Variable(sample['gt_boxes']).cuda()
		gt_masks = Variable(sample['gt_masks']).cuda()
		image_height = image.size()[2]
		image_width = image.size()[3]
		outputs = model(image, image_height, image_width, True, gt_boxes, gt_masks)
		loss = model.loss
		loss.backward()

		# record loss
		total_losses.update(loss.data[0])
		rpn_losses.update(model.RPN.loss.data[0])
		rpn_ce_losses.update(model.RPN.cross_entropy.data[0])
		rpn_box_losses.update(model.RPN.loss_box.data[0])
		odn_losses.update(model.odn.loss.data[0])
		odn_ce_losses.update(model.odn.cross_entropy.data[0])
		odn_box_losses.update(model.odn.loss_box.data[0])
		mask_losses.update(model.mask.mask_loss.data[0])


		optimizer.step()

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
				  'MASK {mask_losses.val:.4f}'
				  .format(i, batch_time=batch_time,
						  data_time=data_time,
						  total_losses=total_losses,
						  rpn_losses=rpn_losses,
						  rpn_ce_losses=rpn_ce_losses,
						  rpn_box_losses=rpn_box_losses,
						  odn_losses=odn_losses,
						  odn_ce_losses=odn_ce_losses,
						  odn_box_losses=odn_box_losses,
						  mask_losses=mask_losses))

		del sample

		if i == end_iter:
			output['image_id'] = image_id[0]
			output['rois'] = outputs[0].tolist()
			output['cls_prob'] = outputs[1].tolist()
			output['keypoints'] = outputs[2].tolist()
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
		  'MASK {mask_losses.avg:.4f}'
		  .format(start_iter, end_iter,
				  batch_time=batch_time,
				  data_time=data_time,
				  total_losses=total_losses,
				  rpn_losses=rpn_losses,
				  rpn_ce_losses=rpn_ce_losses,
				  rpn_box_losses=rpn_box_losses,
				  odn_losses=odn_losses,
				  odn_ce_losses=odn_ce_losses,
				  odn_box_losses=odn_box_losses,
				  mask_losses=mask_losses))

	# if args.tensorboard:
	# 	log_value('train_total_loss', total_losses.avg, end_iter)
	# 	log_value('train_rpn_loss', rpn_losses.avg, end_iter)
	# 	log_value('train_rpn_ce_loss', rpn_ce_losses.avg, end_iter)
	# 	log_value('train_rpn_box_loss', rpn_box_losses.avg, end_iter)
	# 	log_value('train_odn_loss', odn_losses.avg, end_iter)
	# 	log_value('train_odn_ce_loss', odn_ce_losses.avg, end_iter)
	# 	log_value('train_odn_box_loss', odn_box_losses.avg, end_iter)
	return total_losses.avg, output

def validate(val_loader, model):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	total_losses = AverageMeter()
	rpn_losses = AverageMeter()
	odn_losses = AverageMeter()
	rpn_ce_losses = AverageMeter()
	rpn_box_losses = AverageMeter()
	odn_ce_losses = AverageMeter()
	odn_box_losses = AverageMeter()
	mask_losses = AverageMeter()
	output = {}
	# switch to train mode
	model.train()

	end = time.time()

	# get minibatch
	for i, sample in enumerate(val_loader):
		# measure data loading time
		data_time.update(time.time() - end)

		if i == 10:
			break
		image_id = sample['image_id']
		if image_id in img_error_id:
			print('drop: ', image_id)
			continue
		image = Variable(sample['image_t']).cuda()
		gt_boxes = Variable(sample['gt_boxes']).cuda()
		gt_masks = Variable(sample['gt_masks']).cuda()
		image_height = image.size()[2]
		image_width = image.size()[3]
		outputs = model(image, image_height, image_width, True,gt_boxes, gt_masks)
		loss = model.loss

		# record loss
		total_losses.update(loss.data[0])
		rpn_losses.update(model.RPN.loss.data[0])
		rpn_ce_losses.update(model.RPN.cross_entropy.data[0])
		rpn_box_losses.update(model.RPN.loss_box.data[0])
		odn_losses.update(model.odn.loss.data[0])
		odn_ce_losses.update(model.odn.cross_entropy.data[0])
		odn_box_losses.update(model.odn.loss_box.data[0])
		mask_losses.update(model.mask.mask_loss.data[0])


		batch_time.update(time.time() - end)
		end = time.time()


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
			  'MASK {mask_losses.val:.4f}'
			  .format(i, batch_time=batch_time,
					  data_time=data_time,
					  total_losses=total_losses,
					  rpn_losses=rpn_losses,
					  rpn_ce_losses=rpn_ce_losses,
					  rpn_box_losses=rpn_box_losses,
					  odn_losses=odn_losses,
					  odn_ce_losses=odn_ce_losses,
					  odn_box_losses=odn_box_losses,
					  mask_losses=mask_losses))

		output['image_id'] = image_id[0]
		output['rois'] = outputs[0].tolist()
		output['cls_prob'] = outputs[1].tolist()
		output['keypoints'] = outputs[2].tolist()

		val_save_(args.save, output)

		del sample

def test(test_loader, model):
	results = []
	count = 0
	start = time.time()
	for i, sample in enumerate(test_loader):

		# measure data loading time
		result = {}

		image_id = sample['image_id']
		end = time.time()
		print('image_', count, ' --- image_id: ',image_id, '--- time: ', int(end-start))
		count += 1
		image = Variable(sample['image_t']).cuda()
		image_height = image.size()[2]
		image_width = image.size()[3]
		outputs = model(image, image_height, image_width, False)
		cls_prob = outputs[1]
		rois = outputs[0]
		keypoints = outputs[2]
		if len(rois) != 0:
			masks_pre = keypoints.transpose(0, 2, 3, 1)

			keypoint_pre = []
			for j in range(masks_pre.shape[0]):
				mask_key = masks_pre[j]
				temp = cv2.resize(mask_key,
								  (int(rois[j][2] - rois[j][0] + 1), int(rois[j][3] - rois[j][1] + 1)),
								  interpolation=cv2.INTER_LINEAR)

				temp = temp.transpose(2, 0, 1)
				a = np.argmax(temp.reshape(14, -1), axis=1)

				a_mod = a % (448 * 448)
				a1 = np.array(
					np.array(a_mod / 448, dtype=int) / 448.0 * (rois[j][3] - rois[j][1] + 1) + rois[j][
						1], dtype=int)
				a2 = np.array(a_mod % 448 / 448.0 * (rois[j][2] - rois[j][0] + 1) + rois[j][0],
							  dtype=int)
				a3 = np.array(
					a/(448*448), dtype=int) + 1
				keypoint_pre.append(np.hstack((a2.reshape(-1, 1), a1.reshape(-1, 1), a3.reshape(-1, 1))))

			keypoint_pre = np.array(keypoint_pre)
			result['image_id'] = image_id[0]

			result['keypoint_annotations'] = {}
			for i in range(keypoint_pre.shape[0]):
				result['keypoint_annotations']['human%d'%(i+1)] = keypoint_pre[i].reshape(-1).tolist()

			results.append(result)

	test_save_(args.save, results)


if __name__ == '__main__':
	main()