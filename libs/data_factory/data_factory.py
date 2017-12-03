from __future__ import print_function, division
import os
import glob
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json
import warnings

warnings.filterwarnings("ignore")

plt.ion()


def show_masks(image, masks):
	"""Show image with masks"""
	plt.imshow(image)
	plt.scatter(masks[:, 0], masks[:, 1], s=10, marker='.', c='r')
	plt.pause(0.001)

class KeyPointDataset(Dataset):
	"""KeyPoint dataset."""

	def __init__(self, json_file, image_dir, transform=None):
		"""
		:param json_file: Path to the annotations file
		:param data_dir:  Directory with all the images
		:param transform:  Optional transform to be applied on a sample
		"""
		self.anno = json.load(open(json_file, 'r'))
		self.image_dir = image_dir
		self.transform = transform

	def __len__(self):
		return len(self.anno)

	def __getitem__(self, idx):
		img_path = os.path.join(self.image_dir, self.anno[idx]['image_id']+'.jpg')
		image = io.imread(img_path)
		image = np.array(image,dtype=np.float32)
		image_t = image.transpose((2,0,1))
		gt_boxes = np.array(list(self.anno[idx]['human_annotations'].values()))
		gt_boxes = gt_boxes.reshape((-1,4))
		gt_masks = np.array(list(self.anno[idx]['keypoint_annotations'].values()))
		gt_masks = gt_masks.reshape((-1,14,3))
		sample = {'image_t': image_t,
				  'image_id': self.anno[idx]['image_id'],
				  'gt_boxes': gt_boxes,
				  'gt_masks': gt_masks
				  }

		if self.transform:
			sample = self.transform(sample)

		return sample

class KeyPointDataset_test(Dataset):

	def __init__(self, image_dir):

		self.img_path = glob.glob(image_dir + '/*')

	def __len__(self):
		return len(self.img_path)

	def __getitem__(self, idx):
		image = io.imread(self.img_path[idx])
		image = np.array(image, dtype=np.float32)
		image_t = image.transpose((2,0,1))
		sample = {
			'image_t' : image_t,
			'image_id' : self.img_path[idx].split('/')[-1].split('.')[0]
		}

		return sample

# keypoint_dataset = KeyPointDataset(
# 	json_file='/data/KeyPoints/keypointdata/ai_challenger/ai_challenger_keypoint_train_20170902/keypoint_train_annotations_20170909.json',
# 	image_dir='/data/KeyPoints/keypointdata/ai_challenger/ai_challenger_keypoint_train_20170902/keypoint_train_images_20170902'
# 	)
#
# # fig = plt.figure()
# #
# # for i in range(4):
# # 	sample = keypoint_dataset[i]
# #
# # 	print(i, sample['image'].shape, sample['gt_boxes'].shape)
# #
# # 	ax = plt.subplot(1, 4, i+1)
# # 	plt.tight_layout()
# # 	ax.set_title('Sample #{}'.format(i))
# # 	ax.axis('off')
# # 	show_masks(sample['image'],sample['gt_masks'])
# #
# # 	if i == 3:
# # 		plt.show()
# # 		break
#
# dataloader = DataLoader(keypoint_dataset,
# 						batch_size=1,
# 						shuffle=True,
# 						num_workers=4)
#
# for _,sample in enumerate(dataloader):
# 	print(sample['image_t'])
# 	break

