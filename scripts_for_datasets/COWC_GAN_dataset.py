from __future__ import print_function, division

import glob
# Ignore warnings
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")


class COWCGANDataset(Dataset):
  def __init__(self, data_dir_gt, data_dir_lq, image_height=256, image_width=256, transform = None, ext='png'):
    self.data_dir_gt = data_dir_gt
    self.data_dir_lq = data_dir_lq
    #take all under same folder for train and test split.
    self.transform = transform
    self.image_height = image_height
    self.image_width = image_width
    #sort all images for indexing, filter out check.jpgs
    images = list([path.as_posix() for path in Path(self.data_dir_lq).rglob(f"*.{ext}")])
    self.imgs_gt = images
    self.imgs_lq = images
    # self.annotation = list(sorted(glob.glob(self.data_dir_lq+"*.txt")))

  def __getitem__(self, idx):
    #get the paths
    img_path_gt = self.imgs_gt[idx]  #os.path.join(self.data_dir_gt, self.imgs_gt[idx])
    img_path_lq = self.imgs_lq[idx] # os.path.join(self.data_dir_lq, self.imgs_lq[idx])
    # annotation_path = self.annotation[idx]#os.path.join(self.data_dir_lq, self.annotation[idx])
    img_gt = cv2.imread(img_path_gt,1) #read color image height*width*channel=3
    img_lq = cv2.imread(img_path_lq,1) #read color image height*width*channel=3
    img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
    img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2RGB)
    #get the bounding box
    boxes = list()
    label_car_type = list()
    obj_class = 0
    if obj_class == 0:
        boxes.append([0, 0, 1, 1])
        labels = np.ones(len(boxes))  # all are cars
        label_car_type.append(obj_class)
        # create dictionary to access the values
        target = {}
        target['object'] = 0
        target['image_lq'] = img_lq
        target['image'] = img_gt
        target['bboxes'] = boxes
        target['labels'] = labels
        target['label_car_type'] = label_car_type
        target['idx'] = idx
        target['LQ_path'] = img_path_lq

    else:
        labels = np.ones(len(boxes)) # all are cars
        #create dictionary to access the values
        target = {}
        target['object'] = 1
        target['image_lq'] = img_lq
        target['image'] = img_gt
        target['bboxes'] = boxes
        target['labels'] = labels
        target['label_car_type'] = label_car_type
        target['idx'] = idx
        target['LQ_path'] = img_path_lq

    if self.transform is None:
        #convert to tensor
        target = self.convert_to_tensor(**target)
        return target
        #transform
    else:
        transformed = self.transform(**target)
        #print(transformed['image'], transformed['bboxes'], transformed['labels'], transformed['idx'])
        target = self.convert_to_tensor(**transformed)
        return target

  def __len__(self):
    return len(self.imgs_lq)

  def convert_to_tensor(self, **target):
      #convert to tensor
      target['object'] = torch.tensor(target['object'], dtype=torch.int64)
      target['image_lq'] = torch.from_numpy(target['image_lq'].transpose((2, 0, 1)))
      target['image'] = torch.from_numpy(target['image'].transpose((2, 0, 1)))
      target['bboxes'] = torch.as_tensor(target['bboxes'], dtype=torch.int64)
      target['labels'] = torch.ones(len(target['bboxes']), dtype=torch.int64)
      target['label_car_type'] = torch.as_tensor(target['label_car_type'], dtype=torch.int64)
      target['image_id'] = torch.tensor([target['idx']])

      return target

