import numpy as np
#from ..boxes.cython_bbox import bbox_overlaps
from ..layers.get_gt_masks import get_gt_masks
import cv2
_DEBUG = False
def build_mask_label(rois_pos, gt_boxes, gt_masks, index_keep, gt_assignment, image_height, image_width):
	gt_assignment = np.array(gt_assignment[np.array(index_keep,dtype=int)],dtype=int)

	gt_boxes = gt_boxes.reshape(-1, 4)
	gt_masks = gt_masks.reshape(-1, 14, 3)
	if _DEBUG:
		print('rois_pos : ', rois_pos)
		print('gt_boxes : ', gt_boxes.shape)
		print('gt_masks : ', gt_masks)
		print('gt_assignment : ', gt_assignment)
		print('image_height : ', image_height)
		print('image_width : ', image_width)

	# N * 14 * 3
	ass_masks = gt_masks[gt_assignment]
	ass_s_temp = []
	ass_e_temp = []
	for i in range(ass_masks.shape[0]):
		ass_s_temp.append(ass_masks[i, :, :2] - rois_pos[i, :2])
		ass_e_temp.append(-ass_masks[i, :, :2] + rois_pos[i, 2:])

	ass_s_temp = np.array(ass_s_temp)
	ass_e_temp = np.array(ass_e_temp)

	ass_s_label = np.array(ass_s_temp>0, dtype=int)
	ass_e_label = np.array(ass_e_temp>0, dtype=int)

	ass_label = ass_s_label[:,:,0] * ass_s_label[:,:,1] * ass_e_label[:,:,0] * ass_e_label[:,:,1]
	ass_label = np.concatenate((ass_label, ass_label, ass_label),axis=-1).reshape(-1,3,14).transpose(0,2,1)

	coordinate = np.concatenate((ass_s_temp, ass_masks[:, :, -1].reshape(-1,14,1)),axis=-1)

	coordinate = coordinate * ass_label

	#coordinate = np.array(coordinate.transpose((1,0,2)),dtype=int)
	coordinate = np.array(coordinate, dtype=int)



	mask_labels = []
	for j in range(coordinate.shape[0]):
		coor = coordinate[j]
		#coor = np.delete(coor, np.where(coor[:,-1] == 0)[0], axis=0)
		rois_height = int(rois_pos[j][3]-rois_pos[j][1]) + 1
		rois_width = int(rois_pos[j][2]-rois_pos[j][0]) + 1
		# rate_height = 448.0/rois_height
		# rate_width = 448.0/rois_width

		mask_label_keys = np.zeros((448, 448, 28))
		#mask_label_keys = mask_label_keys + 28
		for n in range(coor.shape[0]):
			if coor[n,2] != 0:
				label_height = coor[n,1] / rois_height * 448.0
				label_width = coor[n,0] / rois_width * 448.0
				mask_label_keys[int(label_height), int(label_width), (coor[n,2]-1)+2*n] = 1
		#mask_label = cv2.resize(mask_label_keys,(448, 448), interpolation=cv2.INTER_LINEAR)
		#mask_labels.append(mask_label)
		mask_labels.append(mask_label_keys)


	mask_labels = np.array(mask_labels).reshape((-1,448,448,28)).transpose((0,3,1,2))


	# mask_label = get_gt_masks(
	# 	np.ascontiguousarray(rois_pos, dtype=np.float),
	# 	np.ascontiguousarray(gt_masks, dtype=np.float),
	# 	np.ascontiguousarray(gt_assignment, dtype=np.int32),
	# 	448,
	# 	2,
	# 	14)
	# mask_label = np.array(mask_label)
	# print(mask_label.shape)
	return mask_labels