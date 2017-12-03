cimport cython
import numpy as np
cimport numpy as np
import cv2

DTYPE = np.float
ctypedef np.float_t DTYPE_t

def get_gt_masks(
      np.ndarray[DTYPE_t, ndim=2] rois, 
      np.ndarray[DTYPE_t, ndim=3] gt_masks,
      np.ndarray[np.int32_t, ndim=1] gt_assignment,
      int mask_size,
      int num_class,
      int num_keypoints):
  cdef unsigned int N = rois.shape[0]
  cropped =[]
  cdef unsigned int j,i
  cdef np.ndarray[DTYPE_t, ndim=1] roi
  cdef np.ndarray[DTYPE_t, ndim=2] gt_masks_n
  for j in np.arange(num_keypoints):
    for i in np.arange(N):
      roi = rois[i]
      gt_mask_n = gt_masks[gt_assignment[i]]
      gt_masks_temp = np.zeros(
        (int(roi[3] - roi[1]) + 1, int(roi[2] - roi[0]) + 1, num_class))
      if gt_mask_n[j, 2] != 3 and roi[1]<=int(gt_mask_n[j][1])<int(roi[3]) and roi[0]<=int(gt_mask_n[j][0])<int(roi[2]):
        gt_masks_temp[int(gt_mask_n[j][1]-roi[1])+1, int(gt_mask_n[j][0]-roi[0])+1, int(gt_mask_n[j][2]-1)] = 1

      # if gt_masks_temp.sum() == 0:
      #     gt_masks_temp = np.zeros((56,56,cfg.FLAGS.keypoint_classes))
      # else:
      gt_masks_temp = cv2.resize(gt_masks_temp, (mask_size, mask_size), interpolation=cv2.INTER_AREA)

      # if np.sum(gt_masks_temp) == 0:
      #     gt_masks_temp[0][0][0] = 1e-10
      cropped.append(gt_masks_temp)
  return cropped