import numpy as np
from ..configs.config_v1 import cfg
from .nms_wrapper import nms


def pos_filter_layer(cls_prob, bbox_pred, aligned_features):

	pos_index = np.where(cls_prob[:,1]>0.5)[0]


	if len(pos_index) > cfg['MASK']['MAX_MASK_PER_IMG']:
		index_keep = np.random.choice(pos_index, cfg['MASK']['MAX_MASK_PER_IMG'])

	else:
		index_keep = pos_index
	cls_prob_pos = cls_prob[index_keep]
	bbox_pred_pos = bbox_pred[index_keep]
	aligned_features_pos = aligned_features[index_keep]


	keep = nms(np.hstack((bbox_pred_pos, cls_prob_pos[:, 1].reshape(-1,1))), 0.7)

	cls_prob_pos = cls_prob_pos[keep]
	bbox_pred_pos = bbox_pred_pos[keep]
	aligned_features_pos = aligned_features_pos[keep]
	index_keep = index_keep[keep]

	return cls_prob_pos, bbox_pred_pos, aligned_features_pos, index_keep