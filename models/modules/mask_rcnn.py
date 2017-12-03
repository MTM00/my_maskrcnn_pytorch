#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

# rewrite these functions in pytorch

from .rpn_msr.proposal_layer import proposal_layer as proposal_layer_py
from .rpn_msr.anchor_target_layer import anchor_target_layer as anchor_target_layer_py
from .rpn_msr.proposal_target_layer import proposal_target_layer as proposal_target_layer_py
# from models.modules.rpn_msr.proposal_layer import proposal_layer as proposal_layer_py
# from models.modules.rpn_msr.anchor_target_layer import anchor_target_layer as anchor_target_layer_py
# from models.modules.rpn_msr.proposal_target_layer import proposal_target_layer as proposal_target_layer_py

# for demo
# import cv2
# from utils.blob import im_list_to_blob
from .fast_rcnn.nms_wrapper import nms
from .fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes

# combine two roi layers into one module
from .roi_pooling.modules.roi_pool import RoIPool
from .roi_align.modules.roi_align import RoIAlignAvg


# TODO: for demo
# def nms_detections(pred_boxes, scores, nms_thresh, inds=None):
#     dets = np.hstack((pred_boxes,
#                       scores[:, np.newaxis])).astype(np.float32)
#     keep = nms(dets, nms_thresh)
#     if inds is None:
#         return pred_boxes[keep], scores[keep]
#     return pred_boxes[keep], scores[keep], inds[keep]

def np_to_variable(x, is_cuda=True, dtype=torch.FloatTensor, requires_grad=False):
    if is_cuda:
        return Variable(torch.from_numpy(x).type(dtype).cuda(), requires_grad=requires_grad)
    else:
        return Variable(torch.from_numpy(x).type(dtype), requires_grad=requires_grad)

class RegionProposalNetwork(nn.Module):
    _feat_stride = [16, ]
    anchor_scales = [2, 4, 8, 16, 32]

    def __init__(self, backbone='resnet-50-c4', debug=False):
        super(RegionProposalNetwork, self).__init__()
        if backbone == 'resnet-50-c4':
            resnet = models.resnet50(pretrained=True)
        elif backbone == 'resnet-101-c4':
            resnet = models.resnet101(pretrained=True)
        else:
            raise NotImplementedError
        self.backbone = backbone
        self.res4 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.maxpool,
                                  resnet.layer1, resnet.layer2, resnet.layer3)
        self.conv1 = nn.Sequential(nn.Conv2d(1024, 512, 3, 1, 1),
                                   nn.BatchNorm2d(512), nn.ReLU(True))
        self.score_conv = nn.Conv2d(512, len(self.anchor_scales) * 3 * 2, 1)
        self.bbox_conv = nn.Conv2d(512, len(self.anchor_scales) * 3 * 4, 1)

        # loss
        self.cross_entropy = None
        self.loss_box = None
        self.debug = debug

    @property
    def loss(self):
        return self.cross_entropy + self.loss_box * 10

    def forward(self, im_data, im_info, gt_boxes=None, gt_ishard=None,
                dontcare_areas=None):
        # check im_data
        # im_data = np_to_variable(im_data, is_cuda=True)
        # im_data = im_data.permute(0, 3, 1, 2)
        features = self.res4(im_data)

        rpn_conv1 = self.conv1(features)

        # rpn score
        rpn_cls_score = self.score_conv(rpn_conv1)
        rpn_cls_score_reshape = self.reshape_layer(rpn_cls_score, 2)
        rpn_cls_prob = F.softmax(rpn_cls_score_reshape)
        rpn_cls_prob_reshape = self.reshape_layer(rpn_cls_prob, len(self.anchor_scales)*3*2)
        if self.debug:
            print('rpn_cls_score:', rpn_cls_score.size())
            print('rpn_cls_score_reshape:', rpn_cls_score_reshape.size())
            print('rpn_cls_prob:', rpn_cls_prob.size())
            print('rpn_cls_prob_reshape:', rpn_cls_prob_reshape.size())

        # rpn boxes
        rpn_bbox_pred = self.bbox_conv(rpn_conv1)

        # proposal layer
        # TODO: stop using cfg_key
        cfg_key = 'TRAIN' if self.training else 'TEST'
        rois = self.proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info,
                                   cfg_key, self._feat_stride, self.anchor_scales)

        # generating training labels and build the rpn loss
        if self.training:
            assert gt_boxes is not None
            rpn_data = self.anchor_target_layer(rpn_cls_score, gt_boxes, gt_ishard, dontcare_areas,
                                                im_info, self._feat_stride, self.anchor_scales)
            self.cross_entropy, self.loss_box = self.build_loss(rpn_cls_score_reshape, rpn_bbox_pred, rpn_data)

        return features, rois

    def build_loss(self, rpn_cls_score_reshape, rpn_bbox_pred, rpn_data):
        # classification loss
        # TODO: double-check permutation
        rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(-1, 2)
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

        return rpn_cross_entropy, rpn_loss_box

    @staticmethod
    def reshape_layer(x, d):
        input_shape = x.size()
        # x = x.permute(0, 3, 1, 2)
        # b c w h
        x = x.view(input_shape[0], int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3])
        # x = x.permute(0, 2, 3, 1)
        return x

    @staticmethod
    def proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, cfg_key,
                       _feat_stride, anchor_scales):
        is_cuda = rpn_cls_prob_reshape.is_cuda
        rpn_cls_prob_reshape = rpn_cls_prob_reshape.data.cpu().numpy()
        rpn_bbox_pred = rpn_bbox_pred.data.cpu().numpy()
        x = proposal_layer_py(rpn_cls_prob_reshape, rpn_bbox_pred, im_info,
                              cfg_key, _feat_stride, anchor_scales)
        x = np_to_variable(x, is_cuda=is_cuda)
        return x.view(-1, 5)

    @staticmethod
    def anchor_target_layer(rpn_cls_score, gt_boxes, gt_ishard, dontcare_areas, im_info, _feat_stride, anchor_scales):
        """
        rpn_cls_score: for pytorch (1, Ax2, H, W) bg/fg scores of previous conv layer
        gt_boxes: (G, 5) vstack of [x1, y1, x2, y2, class]
        gt_ishard: (G, 1), 1 or 0 indicates difficult or not
        dontcare_areas: (D, 4), some areas may contains small objs but no labelling. D may be 0
        im_info: a list of [image_height, image_width, scale_ratios]
        _feat_stride: the downsampling ratio of feature map to the original input image
        anchor_scales: the scales to the basic_anchor (basic anchor is [16, 16])
        ----------
        Returns
        ----------
        rpn_labels : (1, 1, HxA, W), for each anchor, 0 denotes bg, 1 fg, -1 dontcare
        rpn_bbox_targets: (1, 4xA, H, W), distances of the anchors to the gt_boxes(may contains some transform)
                        that are the regression objectives
        rpn_bbox_inside_weights: (1, 4xA, H, W) weights of each boxes, mainly accepts hyper param in cfg
        rpn_bbox_outside_weights: (1, 4xA, H, W) used to balance the fg/bg,
        beacuse the numbers of bgs and fgs mays significiantly different
        """
        is_cuda = rpn_cls_score.is_cuda
        rpn_cls_score = rpn_cls_score.data.cpu().numpy()
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
            anchor_target_layer_py(rpn_cls_score, gt_boxes, gt_ishard, dontcare_areas, im_info, _feat_stride, anchor_scales)

        rpn_labels = np_to_variable(rpn_labels, is_cuda=is_cuda, dtype=torch.LongTensor)
        rpn_bbox_targets = np_to_variable(rpn_bbox_targets, is_cuda=is_cuda)
        rpn_bbox_inside_weights = np_to_variable(rpn_bbox_inside_weights, is_cuda=is_cuda)
        rpn_bbox_outside_weights = np_to_variable(rpn_bbox_outside_weights, is_cuda=is_cuda)

        return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


class ObjectDetectionNetwork(nn.Module):

    def __init__(self, backbone='resnet-50-c4', num_classes=80, debug=False):
        super(ObjectDetectionNetwork, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        if backbone == 'resnet-50-c4':
            resnet = models.resnet50()
        elif backbone == 'resnet-101-c4':
            resnet = models.resnet101()
        else:
            raise NotImplementedError

        self.roi_align = RoIAlignAvg(7, 7, 1./16)
        # self.roi_align = RoIPool(7, 7, 1./16)

        # change the create new res5 with stride=1
        resnet.inplanes = 1024
        self.res5 = resnet._make_layer(models.resnet.Bottleneck, 512, 3, stride=1)

        self.avgpool = resnet.avgpool
        self.score_fc = nn.Linear(2048, self.num_classes) 
        self.bbox_fc = nn.Linear(2048, self.num_classes * 4) 

        # loss
        self.cross_entropy = None
        self.loss_box = None

        # for log
        self.debug = debug

    @property
    def loss(self):
        return self.cross_entropy + self.loss_box * 10

    def forward(self, features, rois, roi_data=None):
        # roi align
        aligned_features = self.roi_align(features, rois)
        x = self.res5(aligned_features)
        x = self.avgpool(x).view(-1, 2048)

        if self.debug:
            print('aligned features size:', aligned_features.size())

        cls_score = self.score_fc(x)
        cls_prob = F.softmax(cls_score)
        bbox_pred = self.bbox_fc(x)

        if self.training:
            self.cross_entropy, self.loss_box = self.build_loss(cls_score, bbox_pred, roi_data)

        return cls_prob, bbox_pred

    def build_loss(self, cls_score, bbox_pred, roi_data):
        # classification loss
        label = roi_data[1].squeeze()

        cross_entropy = F.cross_entropy(cls_score, label)

        # bounding box regression L1 loss
        bbox_targets, bbox_inside_weights, bbox_outside_weights = roi_data[2:]
        bbox_targets = torch.mul(bbox_targets, bbox_inside_weights)
        bbox_pred = torch.mul(bbox_pred, bbox_inside_weights)

        loss_box = F.smooth_l1_loss(bbox_pred, bbox_targets)

        return cross_entropy, loss_box


class FasterRCNN(nn.Module):

    def __init__(self, backbone='resnet-50-c4', num_classes=80, debug=False):
        super(FasterRCNN, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes

        self.rpn = RegionProposalNetwork(backbone=backbone)
        self.odn = ObjectDetectionNetwork(backbone=backbone, num_classes=num_classes)

        # for log
        self.debug = debug

    @property
    def loss(self):
        return self.rpn.loss + self.odn.loss

    def forward(self, im_data, im_info, gt_boxes=None, gt_ishard=None,
                dontcare_areas=None):
        features, rois = self.rpn(im_data, im_info, gt_boxes, gt_ishard,
                                  dontcare_areas)

        if self.training:
            roi_data = self.proposal_target_layer(rois, gt_boxes, gt_ishard,
                                                  dontcare_areas, self.num_classes)
            rois = roi_data[0]
        else:
            roi_data = None

        cls_prob, bbox_pred = self.odn(features, rois, roi_data)

        return cls_prob, bbox_pred, rois

    @staticmethod
    def proposal_target_layer(rpn_rois, gt_boxes, gt_ishard, dontcare_areas, num_classes):
        """
        ----------
        rpn_rois:  (1 x H x W x A, 5) [0, x1, y1, x2, y2]
        gt_boxes: (G, 5) [x1 ,y1 ,x2, y2, class] int
        # gt_ishard: (G, 1) {0 | 1} 1 indicates hard
        dontcare_areas: (D, 4) [ x1, y1, x2, y2]
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
        rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = \
            proposal_target_layer_py(
                rpn_rois, gt_boxes, gt_ishard, dontcare_areas, num_classes)
        # print labels.shape, bbox_targets.shape, bbox_inside_weights.shape
        rois = np_to_variable(rois, is_cuda=is_cuda)
        labels = np_to_variable(labels, is_cuda=is_cuda, dtype=torch.LongTensor)
        bbox_targets = np_to_variable(bbox_targets, is_cuda=is_cuda)
        bbox_inside_weights = np_to_variable(bbox_inside_weights, is_cuda=is_cuda)
        bbox_outside_weights = np_to_variable(bbox_outside_weights, is_cuda=is_cuda)

        return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights


    # TODO: modify this part
    @staticmethod
    def interpret_outputs(cls_prob, bbox_pred, rois, im_info, nms=True, clip=True, min_score=0.0):
        # find class
        scores, inds = cls_prob.data.max(1)
        scores, inds = scores.cpu().numpy(), inds.cpu().numpy()

        keep = np.where((inds > 0) & (scores >= min_score))
        scores, inds = scores[keep], inds[keep]

        # Apply bounding-box regression deltas
        keep = keep[0]
        box_deltas = bbox_pred.data.cpu().numpy()[keep]
        box_deltas = np.asarray([
            box_deltas[i, (inds[i] * 4): (inds[i] * 4 + 4)] for i in range(len(inds))
        ], dtype=np.float)
        boxes = rois.data.cpu().numpy()[keep, 1:5] / im_info[0][2]
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        if clip:
            im_shape = (round(im_info[0][0] / im_info[0][2]),
                        round(im_info[0][1] / im_info[0][2]))
            pred_boxes = clip_boxes(pred_boxes, im_shape)

        # nms
        if nms and pred_boxes.shape[0] > 0:
            pred_boxes, scores, inds = nms_detections(pred_boxes, scores, 0.3, inds=inds)

        return scores, pred_boxes

    # TODO: modify this part
    # def detect(self, image, thr=0.3):
    #     im_data, im_scales = self.get_image_blob(image)
    #     im_info = np.array(
    #         [[im_data.shape[1], im_data.shape[2], im_scales[0]]],
    #         dtype=np.float32)

    #     cls_prob, bbox_pred, rois = self(im_data, im_info)
    #     pred_boxes, scores, classes = \
    #         self.interpret_faster_rcnn(cls_prob, bbox_pred, rois, im_info, image.shape, min_score=thr)
    #     return pred_boxes, scores, classes

    # TODO: modify this part
    # def get_image_blob_noscale(self, im):
    #     im_orig = im.astype(np.float32, copy=True)
    #     im_orig -= self.PIXEL_MEANS

    #     processed_ims = [im]
    #     im_scale_factors = [1.0]

    #     blob = im_list_to_blob(processed_ims)

    #     return blob, np.array(im_scale_factors)

    # TODO: double check
    # def get_image_blob(self, im):
    #     """Converts an image into a network input.
    #     Arguments:
    #         im (ndarray): a color image in BGR order
    #     Returns:
    #         blob (ndarray): a data blob holding an image pyramid
    #         im_scale_factors (list): list of image scales (relative to im) used
    #             in the image pyramid
    #     """
    #     im_orig = im.astype(np.float32, copy=True)
    #     im_orig -= self.PIXEL_MEANS

    #     im_shape = im_orig.shape
    #     im_size_min = np.min(im_shape[0:2])
    #     im_size_max = np.max(im_shape[0:2])

    #     processed_ims = []
    #     im_scale_factors = []

    #     for target_size in self.SCALES:
    #         im_scale = float(target_size) / float(im_size_min)
    #         # Prevent the biggest axis from being more than MAX_SIZE
    #         if np.round(im_scale * im_size_max) > self.MAX_SIZE:
    #             im_scale = float(self.MAX_SIZE) / float(im_size_max)
    #         im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
    #                         interpolation=cv2.INTER_LINEAR)
    #         im_scale_factors.append(im_scale)
    #         processed_ims.append(im)

    #     # Create a blob to hold the input images
    #     blob = im_list_to_blob(processed_ims)

    #     return blob, np.array(im_scale_factors)


class MaskRCNN(FasterRCNN):

    def __init__(self, backbone='resnet50', classes=None, debug=False):
        super(MaskRCNN, self).__init__(backbone, classes, debug)

