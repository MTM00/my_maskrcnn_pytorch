import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import torch.nn as nn
from .network import Conv2d
from ..layers.proposal_layer import proposal_layer as proposal_layer_py
from ..layers.anchor_target_layer import anchor_target_layer as anchor_target_layer_py
from ..layers.proposal_target_layer import proposal_target_layer as proposal_target_layer_py
from ..layers.pos_filter_layer import pos_filter_layer as pos_filter_layer_py
from ..layers.build_mask_label import build_mask_label as build_mask_label_py
from ..boxes.bbox_transform import bbox_transform_inv, clip_boxes
from ..roi_align.modules.roi_align import RoIAlign

base_anchors = 9
def np_to_variable(x, is_cuda=True, dtype=torch.FloatTensor, requires_grad=False):
	if is_cuda:
		return Variable(torch.from_numpy(x).type(dtype).cuda(), requires_grad=requires_grad)
	else:
		return Variable(torch.from_numpy(x).type(dtype), requires_grad=requires_grad)


class Pyramid_Network(nn.Module):
	def __init__(self, pretrain_model):
		super(Pyramid_Network, self).__init__()

		self.pyramid1 = nn.Sequential(*list(pretrain_model.children())[:5])
		self.pyramid2 = nn.Sequential(*list(pretrain_model.children())[5:6])
		self.pyramid3 = nn.Sequential(*list(pretrain_model.children())[6:7])
		self.pyramid4 = nn.Sequential(*list(pretrain_model.children())[7:8])
		self.conv = nn.ModuleList([
			Conv2d(256, 256, 1),
			Conv2d(512, 256, 1),
			Conv2d(1024, 256, 1),
			Conv2d(2048, 256, 1)
		])

		self.conv2 = nn.ModuleList([Conv2d(256,256,3) for i in range(3)])

		for p in self.pyramid1.parameters():
			p.requires_grad = False
		for p in self.pyramid2.parameters():
			p.requires_grad = False
		for p in self.pyramid3.parameters():
			p.requires_grad = False
		for p in self.pyramid4.parameters():
			p.requires_grad = False


	def forward(self, input):
		pyramid = []
		pyramid.append(self.pyramid1(input))
		pyramid.append(self.pyramid2(pyramid[0]))
		pyramid.append(self.pyramid3(pyramid[1]))
		pyramid.append(self.pyramid4(pyramid[2]))

		pyramid[3] = self.conv[3](pyramid[3])
		for c in range(3, 0, -1):
			s, s_ = pyramid[c], pyramid[c-1]

			up_shape = s_.size()

			map = torch.nn.UpsamplingBilinear2d((up_shape[2],up_shape[3]))
			s = map(s)
			s_ = self.conv[c-1](s_)

			s.add_(s_)
			s = self.conv2[c-1](s)

			pyramid[c-1] = s

		return pyramid


class RPN(nn.Module):
	def __init__(self):
		super(RPN, self).__init__()
		self.rpn_conv = Conv2d(256, 256, 3)
		self.box_conv = Conv2d(256, base_anchors*4, 1, relu=False, same_padding=False, bn=False)
		self.cls_conv = Conv2d(256, base_anchors*2, 1, relu=False, same_padding=False, bn=False)
		self.training = True
		self.softmax2d = nn.Softmax2d()
		# loss
		self.cross_entropy = None
		self.loss_box = None

	@property
	def loss(self):
		return self.cross_entropy + self.loss_box * 10


	def forward(self, pyramid_network, image_height, image_width, gt_boxes=None, gt_masks=None):
		features = pyramid_network[3]
		stride = 32
		anchor_scales = [16, 32, 64]
		rpn_conv1 = self.rpn_conv(features)

		# rpn score
		rpn_cls_score = self.cls_conv(rpn_conv1)
		#rpn_cls_score_reshape = rpn_cls_score.permute(0,2,3,1).contiguous()
		rpn_cls_prob = self.softmax2d(rpn_cls_score)
		#rpn_cls_prob_reshape = rpn_cls_prob.permute(0,3,1,2,).contiguous()

		# rpn boxes
		rpn_bbox_pred = self.box_conv(rpn_conv1)
		#rpn_bbox_pred_reshape = rpn_bbox_pred.permute(0,2,3,1).contiguous()

		# proposal layer
		rois = self.proposal_layer(rpn_cls_prob, rpn_bbox_pred, image_height, image_width, stride, anchor_scales, is_train=True)

		# generating training labels and build the rpn loss
		if self.training:
			assert gt_boxes is not None
			rpn_data = self.anchor_target_layer(rpn_cls_score, gt_boxes, image_height, image_width,
												stride, anchor_scales)
			self.cross_entropy, self.loss_box = self.build_loss(rpn_cls_score, rpn_bbox_pred, rpn_data)

		return rois

	def build_loss(self, rpn_cls_score, rpn_bbox_pred, rpn_data):
		rpn_cls_score = rpn_cls_score.view(-1, 2)
		rpn_label = rpn_data[0].view(-1)

		# TODO: very what -1 means, I assume it's the don't care label
		rpn_keep = Variable(rpn_label.data.ne(-1).nonzero().squeeze())
		if rpn_cls_score.is_cuda:
			rpn_keep = rpn_keep.cuda()
		rpn_cls_score = torch.index_select(rpn_cls_score, 0, rpn_keep)
		rpn_label = torch.index_select(rpn_label, 0, rpn_keep)

		rpn_cross_entropy = F.cross_entropy(rpn_cls_score, rpn_label)

		# box loss
		rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]
		rpn_bbox_targets = torch.mul(rpn_bbox_targets, rpn_bbox_inside_weights)
		rpn_bbox_pred = torch.mul(rpn_bbox_pred, rpn_bbox_inside_weights)

		fg_cnt = torch.sum(rpn_label.data.ne(0))
		rpn_loss_box = F.smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, size_average=False) / (fg_cnt + 1e-4)

		print('rpn_cross_entropy : ', rpn_cross_entropy)
		print('rpn_loss_box : ', rpn_loss_box)

		return rpn_cross_entropy, rpn_loss_box

	@staticmethod
	def proposal_layer(rpn_cls_prob, rpn_bbox_pred, image_height, image_width, stride, anchor_scales, is_train):
		is_cuda = rpn_cls_prob.is_cuda
		rpn_cls_prob = rpn_cls_prob.data.cpu().numpy()
		rpn_bbox_pred = rpn_bbox_pred.data.cpu().numpy()
		x = proposal_layer_py(rpn_cls_prob, rpn_bbox_pred, image_height, image_width, stride, anchor_scales, is_train)
		x = np_to_variable(x, is_cuda=is_cuda)
		return x

	@staticmethod
	def anchor_target_layer(rpn_cls_score, gt_boxes, image_height, image_width, stride, anchor_scales):
		is_cuda = rpn_cls_score.is_cuda
		rpn_cls_score = rpn_cls_score.data.cpu().numpy()
		gt_boxes = gt_boxes.data.cpu().numpy().reshape((-1,4))
		rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
			anchor_target_layer_py(rpn_cls_score, gt_boxes, image_height, image_width, stride, anchor_scales)

		rpn_labels = np_to_variable(rpn_labels, is_cuda=is_cuda, dtype=torch.LongTensor)
		rpn_bbox_targets = np_to_variable(rpn_bbox_targets, is_cuda=is_cuda)
		rpn_bbox_inside_weights = np_to_variable(rpn_bbox_inside_weights, is_cuda=is_cuda)
		rpn_bbox_outside_weights = np_to_variable(rpn_bbox_outside_weights, is_cuda=is_cuda)

		return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights

class ObjectDetectionNetwork(nn.Module):

	def __init__(self, num_classes, debug=False):
		super(ObjectDetectionNetwork, self).__init__()

		self.num_classes = num_classes
		self.roi_align = RoIAlign(14,14,1.0/32)
		self.fc1 = nn.Linear(256*14*14,1024)
		self.relu1 = nn.ReLU()
		self.dropout = nn.Dropout()
		self.fc2 = nn.Linear(1024,1024)
		self.relu2 = nn.ReLU()

		self.score_fc = nn.Linear(1024, self.num_classes)
		self.bbox_fc = nn.Linear(1024, self.num_classes*4)


		# loss
		self.cross_entropy = None
		self.loss_box = None

		# for log
		self.debug = debug

	@property
	def loss(self):
		return self.cross_entropy + self.loss_box * 10

	def forward(self, features, rois, roi_data=None):
		aligned_features = self.roi_align(features[3], rois)
		aligned_features_flatten = aligned_features.view((-1,256*14*14))
		aligned_features_flatten = self.fc1(aligned_features_flatten)
		aligned_features_flatten = self.relu1(aligned_features_flatten)
		aligned_features_flatten = self.dropout(aligned_features_flatten)
		aligned_features_flatten = self.fc2(aligned_features_flatten)
		aligned_features_flatten = self.relu2(aligned_features_flatten)

		cls_score = self.score_fc(aligned_features_flatten)
		cls_prob = F.softmax(cls_score)
		bbox_pred = self.bbox_fc(aligned_features_flatten)

		if self.training:
			self.cross_entropy, self.loss_box = self.build_loss(cls_score, bbox_pred, roi_data)

		return cls_prob, bbox_pred, aligned_features

	def build_loss(self, cls_score, bbox_pred, roi_data):
		# classification loss
		label = roi_data[1].squeeze()
		cross_entropy = F.cross_entropy(cls_score, label)

		# bounding box regression L1 loss
		bbox_targets, bbox_inside_weights, bbox_outside_weights = roi_data[2:5]
		bbox_targets = torch.mul(bbox_targets, bbox_inside_weights)
		bbox_pred = torch.mul(bbox_pred, bbox_inside_weights)

		loss_box = F.smooth_l1_loss(bbox_pred, bbox_targets)

		print('cross_entropy : ', cross_entropy)
		print('loss_box : ', loss_box)

		return cross_entropy, loss_box

class Mask(nn.Module):
	def __init__(self):
		super(Mask, self).__init__()
		self.mask_conv_1 = Conv2d(256, 256, 3, bn=False)
		self.mask_conv_2 = Conv2d(256, 256, 3, bn=False)
		self.mask_conv_3 = Conv2d(256, 256, 3, bn=False)
		self.mask_conv_4 = Conv2d(256, 256, 3, bn=False)
		self.mask_conv_5 = Conv2d(256, 256, 3, bn=False)
		self.mask_conv_6 = Conv2d(256, 256, 3, bn=False)

		# to 28 * 28
		self.mask_deconv_1 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
		# to 56 * 56
		self.mask_deconv_2 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
		# to 112 * 112
		self.mask_deconv_3 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
		# to 224 * 224
		self.mask_deconv_4 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
		# to 448 * 448
		self.mask_deconv_5 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)

		self.keypoint_1 = nn.Conv2d(256, 2, 1)
		self.keypoint_2 = nn.Conv2d(256, 2, 1)
		self.keypoint_3 = nn.Conv2d(256, 2, 1)
		self.keypoint_4 = nn.Conv2d(256, 2, 1)
		self.keypoint_5 = nn.Conv2d(256, 2, 1)
		self.keypoint_6 = nn.Conv2d(256, 2, 1)
		self.keypoint_7 = nn.Conv2d(256, 2, 1)
		self.keypoint_8 = nn.Conv2d(256, 2, 1)
		self.keypoint_9 = nn.Conv2d(256, 2, 1)
		self.keypoint_10 = nn.Conv2d(256, 2, 1)
		self.keypoint_11 = nn.Conv2d(256, 2, 1)
		self.keypoint_12 = nn.Conv2d(256, 2, 1)
		self.keypoint_13 = nn.Conv2d(256, 2, 1)
		self.keypoint_14 = nn.Conv2d(256, 2, 1)

		self.softmax2d = nn.Softmax2d()

		# loss
		self.mask_loss = None



	def forward(self, cls_prob_pos, rois_pos, aligned_features_pos, index_keep, image_height, image_width, gt_boxes=None, gt_masks=None, roi_data=None):
		features = self.mask_conv_1(aligned_features_pos)
		features = self.mask_conv_2(features)
		features = self.mask_conv_3(features)
		features = self.mask_conv_4(features)
		features = self.mask_conv_5(features)
		features = self.mask_conv_6(features)

		features = self.mask_deconv_1(features)
		features = self.mask_deconv_2(features)
		features = self.mask_deconv_3(features)
		features = self.mask_deconv_4(features)
		features = self.mask_deconv_5(features)

		keypoint_1 = self.keypoint_1(features)
		keypoint_2 = self.keypoint_2(features)
		keypoint_3 = self.keypoint_3(features)
		keypoint_4 = self.keypoint_4(features)
		keypoint_5 = self.keypoint_5(features)
		keypoint_6 = self.keypoint_6(features)
		keypoint_7 = self.keypoint_7(features)
		keypoint_8 = self.keypoint_8(features)
		keypoint_9 = self.keypoint_9(features)
		keypoint_10 = self.keypoint_10(features)
		keypoint_11 = self.keypoint_11(features)
		keypoint_12 = self.keypoint_12(features)
		keypoint_13 = self.keypoint_13(features)
		keypoint_14 = self.keypoint_14(features)

		keypoints = []
		keypoints.append(keypoint_1)
		keypoints.append(keypoint_2)
		keypoints.append(keypoint_3)
		keypoints.append(keypoint_4)
		keypoints.append(keypoint_5)
		keypoints.append(keypoint_6)
		keypoints.append(keypoint_7)
		keypoints.append(keypoint_8)
		keypoints.append(keypoint_9)
		keypoints.append(keypoint_10)
		keypoints.append(keypoint_11)
		keypoints.append(keypoint_12)
		keypoints.append(keypoint_13)
		keypoints.append(keypoint_14)
		keypoints = torch.cat(keypoints,0)

		keypoints = keypoints.view(14,-1,2,448,448)


		if self.training:
			self.mask_loss = self.build_loss(keypoints, rois_pos, index_keep, gt_boxes, gt_masks, image_height, image_width, roi_data)

		return keypoints

	@property
	def loss(self):
		return self.mask_loss

	def build_loss(self, keypoints, rois_pos, index_keep, gt_boxes, gt_masks, image_height, image_width, roi_data):

		mask_label = self.build_mask_label(rois_pos, gt_boxes, gt_masks, index_keep, roi_data, image_height, image_width)
		#keypoints = keypoints.view(-1,2*448*448)
		mask_label = mask_label.view(-1,2*448*448)
		keypoints = keypoints.view(-1, 2, 448, 448)
		keypoints_pro = self.softmax2d(keypoints)
		keypoints_pro = keypoints_pro.view(-1, 2*448*448)
		mask_loss = -torch.sum(mask_label * torch.log(keypoints_pro))/mask_label.size()[0]

		print('mask_loss : ', mask_loss)

		return mask_loss



	@staticmethod
	def build_mask_label(rois_pos, gt_boxes, gt_masks, index_keep, roi_data, image_height, image_width):
		is_cuda = rois_pos.is_cuda
		rois_pos = rois_pos.data.cpu().numpy()
		gt_boxes = gt_boxes.data.cpu().numpy()
		gt_masks = gt_masks.data.cpu().numpy()
		index_keep = index_keep.data.cpu().numpy()
		gt_assignment = roi_data[5].data.cpu().numpy()

		mask_label = build_mask_label_py(rois_pos, gt_boxes, gt_masks, index_keep, gt_assignment, image_height, image_width)

		mask_label = np_to_variable(np.array(mask_label), is_cuda=is_cuda)

		return mask_label


class Network(nn.Module):
	def __init__(self, pretrain_model, num_classes=2):
		super(Network, self).__init__()
		self.num_classes = num_classes
		self.pyramid_network = Pyramid_Network(pretrain_model)
		self.RPN = RPN()
		self.odn = ObjectDetectionNetwork(self.num_classes)
		self.mask = Mask()

		self.mask_loss = None
		self.rpn_loss = None
		self.odn_loss = None

	@property
	def loss(self):
		return self.rpn_loss + self.odn_loss + self.mask_loss


	def forward(self, input, image_height, image_width, gt_boxes=None, gt_masks=None):
		pyramid = self.pyramid_network(input)
		rois = self.RPN(pyramid, image_height, image_width, gt_boxes, gt_masks)
		if self.training:
			roi_data = self.proposal_target_layer(rois, gt_boxes, image_height, image_width, self.num_classes)
			rois = roi_data[0]

		else:
			roi_data = None

		cls_prob, bbox_pred, aligned_features = self.odn(pyramid, rois, roi_data)
		rois = self.interpret_outputs(rois, bbox_pred,cls_prob, image_height, image_width)
		cls_prob_pos, rois_pos, aligned_features_pos, index_keep, is_empty = self.pos_filter_layer(cls_prob, rois,
																						 aligned_features)
		if not is_empty:
			keypoints = self.mask(cls_prob_pos, rois_pos, aligned_features_pos, index_keep, image_height, image_width, gt_boxes, gt_masks, roi_data)
			self.mask_loss = self.mask.loss
		else:
			keypoints = Variable(torch.zeros(1))
			self.mask_loss = 0.0
		self.rpn_loss = self.RPN.loss
		self.odn_loss = self.odn.loss

		return keypoints

	@staticmethod
	def proposal_target_layer(rpn_rois, gt_boxes, image_height, image_width, num_classes):
		"""
		----------
		rpn_rois:  (1 x H x W x A, 5) [0, x1, y1, x2, y2]
		gt_boxes: (G, 4) [x1 ,y1 ,x2, y2] int
		num_classes
		----------
		Returns
		----------
		rois: (1 x H x W x A, 5) [0, x1, y1, x2, y2]
		labels: (1 x H x W x A, 1) {0,1,...,_num_classes-1}
		bbox_targets: (1 x H x W x A, K x4) [dx1, dy1, dx2, dy2]
		bbox_inside_weights: (1 x H x W x A, Kx4) 0, 1 masks for the computing loss
		bbox_outside_weights: (1 x H x W x A, Kx4) 0, 1 masks for the computing loss
		"""
		is_cuda = rpn_rois.is_cuda
		rpn_rois = rpn_rois.data.cpu().numpy()
		gt_boxes = gt_boxes.data.cpu().numpy().reshape((-1, 4))
		rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, gt_assignment_keep = \
			proposal_target_layer_py(rpn_rois, gt_boxes, image_height, image_width, num_classes)
		# print labels.shape, bbox_targets.shape, bbox_inside_weights.shape
		rois = np_to_variable(rois, is_cuda=is_cuda)
		labels = np_to_variable(labels, is_cuda=is_cuda, dtype=torch.LongTensor)
		bbox_targets = np_to_variable(bbox_targets, is_cuda=is_cuda)
		bbox_inside_weights = np_to_variable(bbox_inside_weights, is_cuda=is_cuda)
		bbox_outside_weights = np_to_variable(bbox_outside_weights, is_cuda=is_cuda)
		gt_assignment_keep = np_to_variable(gt_assignment_keep, is_cuda=is_cuda)

		return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, gt_assignment_keep

	@staticmethod
	def interpret_outputs(rois, bbox_pred, cls_prob, image_height, image_width, clip=True):
		is_cuda = rois.is_cuda
		box_deltas = bbox_pred.data.cpu().numpy()
		boxes = rois.data.cpu().numpy()[:, 1:5]
		cls_prob = cls_prob.data.cpu().numpy()

		idx = cls_prob.argmax(axis=1)
		box_deltas = box_deltas.reshape(-1,2,4)
		box_deltas = box_deltas[np.arange(box_deltas.shape[0]),idx]

		pred_boxes = bbox_transform_inv(boxes, box_deltas)
		if clip:
			pred_boxes = clip_boxes(pred_boxes, [image_height, image_width])

		pred_boxes = np_to_variable(pred_boxes, is_cuda=is_cuda)
		return pred_boxes

	@staticmethod
	def pos_filter_layer(cls_prob, rois, aligned_features):
		is_cuda = cls_prob.is_cuda
		cls_prob = cls_prob.data.cpu().numpy()
		bbox_pred = rois.data.cpu().numpy()
		aligned_features = aligned_features.data.cpu().numpy()
		cls_prob_pos, bbox_pred_pos, aligned_features_pos, index_keep = pos_filter_layer_py(cls_prob, bbox_pred,
																							aligned_features)

		is_empty = True
		if bbox_pred_pos.size != 0:
			cls_prob_pos = np_to_variable(cls_prob_pos, is_cuda=is_cuda)
			bbox_pred_pos = np_to_variable(bbox_pred_pos, is_cuda=is_cuda)
			aligned_features_pos = np_to_variable(aligned_features_pos, is_cuda=is_cuda)
			index_keep = np_to_variable(index_keep, is_cuda=is_cuda)
			is_empty = False

		return cls_prob_pos, bbox_pred_pos, aligned_features_pos, index_keep, is_empty