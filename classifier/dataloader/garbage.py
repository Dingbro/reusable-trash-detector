import os
import random
import time
import warnings
import json
warnings.simplefilter("ignore", UserWarning)

import torch
import numpy as np
import cv2
from torchvision import transforms
import glob
from PIL import Image

int_to_label = {0:"종이", 1:"종이팩", 2:"알류미늄캔", 3:"유리", 4:"페트", 5:"플라스틱", 6:"비닐"}
label_to_int = {int_to_label[ele]: ele for ele in int_to_label}

class GarbageDatasetwithBbox(torch.utils.data.Dataset):
    def __init__(self, root, data_file, transform, soft_label = False, mode='train'):
        self.root = root
        self.transform = transform
        self.mode = mode

        with open(data_file, 'r') as json_file:
            self.label_dic = json.load(json_file)

        imnames = []
        imclasses = []
        bboxes = []
        
        for ele in self.label_dic:
            
            if mode=='train':
                if soft_label:
                    temp = ele['class']
                else:
                    temp = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]

                    temp_class = ele['class']

                    for i in range(len(temp_class)):
                        if temp_class[i]+0.001>0.5:
                            temp[i]=1.0
                        
            else:
                temp = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
                if 'labels' in ele.keys():
                    if len(ele['labels'])==0:
                        continue

                    for t in ele['labels']:
                        temp[t]=1.0
                else:
                    if len(ele['class'])==0:
                        continue
                    for t in ele['class']:
                        temp[t]=1.0
                

            imnames.append(ele['filename'])
            imclasses.append(temp)
            bboxes.append(ele['bbox'])

        self.imnames = imnames
        self.imclasses = imclasses
        self.bboxed = bboxes


    def __getitem__(self, idx):

        img_path = os.path.join(self.root, self.imnames[idx])
        label = self.imclasses[idx]
        #image = Image.open(img_path).convert('RGB')
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bbox = self.bboxed[idx]
        if self.mode == 'train':
            randfloat = random.uniform(1/32, 1)
        else:
            randfloat = 1/8

        origin_bbox, new_bboxes = self.postprocessing(image.shape, bbox, randfloat)
        image = image[new_bboxes[1]:new_bboxes[1]+new_bboxes[3],new_bboxes[0]:new_bboxes[0]+new_bboxes[2],:]
        
        if self.transform is not None:
            try:
                image = self.transform(image=image)['image']
            except:
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = self.transform(image=image)['image']
            
        
        return img_path, image, torch.tensor(label)

    def __len__(self):
        return len(self.imnames)

    def postprocessing(self, shape, bboxes, margin=1/2):
        
        height = shape[0]
        width = shape[1]

        bboxes = np.array([[bboxes[0]*456, bboxes[1]*456, bboxes[2]*456, bboxes[3]*456]]).astype(np.float32).tolist()
        multiply = max(shape)/456
        bboxes = (np.array(bboxes)*multiply).astype(np.int64).tolist()
        bboxes = bboxes[0][:]

        if width>height:
            pad = (width-height)/2
            bboxes[1]=bboxes[1]-pad
        else:
            pad = (height-width)/2
            bboxes[0]=bboxes[0]-pad
        
        x_min, y_min, bbox_width, bbox_height = bboxes
        origin_bbox = np.array([x_min, y_min, bbox_width, bbox_height]).astype(np.int64).tolist()

        #return np.array([x_min, y_min, bbox_width, bbox_height]).astype(np.int64).tolist()
        
        if bbox_width>bbox_height:
            bbox_pad = (bbox_width-bbox_height)/2
            y_min = max((y_min-bbox_pad),0)
            bbox_height = min((bbox_height+bbox_pad*2),height-y_min)
        else:
            bbox_pad = (bbox_height-bbox_width)/2
            x_min = max((x_min-bbox_pad),0)
            bbox_width = min((bbox_width+bbox_pad*2),width-x_min)
        
        margin = max(bbox_width, bbox_height)*margin
        #margin = 0
        
        x_min = max(x_min-margin/2, 0)
        y_min = max(y_min-margin/2, 0)
        bbox_width = min(bbox_width+margin, width-x_min)
        bbox_height = min(bbox_height+margin, height-y_min)
        
        final_bbox = np.array([x_min, y_min, bbox_width, bbox_height]).astype(np.int64).tolist()
        
        return origin_bbox, final_bbox

class GarbageDataset(torch.utils.data.Dataset):
    def __init__(self, root, data_file, transform, soft_label = False, mode='train'):
        self.root = root
        self.transform = transform

        with open(data_file, 'r') as json_file:
            self.label_dic = json.load(json_file)

        imnames = []
        imclasses = []
        
        for ele in self.label_dic:
            
            if mode=='train':
                if soft_label:
                    temp = ele['class']
                else:
                    temp = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]

                    temp_class = ele['class']

                    for i in range(len(temp_class)):
                        if temp_class[i]+0.001>0.5:
                            temp[i]=1.0
                        
            else:
                temp = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
                if 'labels' in ele.keys():
                    if len(ele['labels'])==0:
                        continue

                    for t in ele['labels']:
                        temp[t]=1.0
                else:
                    if len(ele['class'])==0:
                        continue
                    for t in ele['class']:
                        temp[t]=1.0
                

            imnames.append(ele['filename'])
            imclasses.append(temp)

        self.imnames = imnames
        self.imclasses = imclasses

    def __getitem__(self, idx):

        img_path = os.path.join(self.root, self.imnames[idx])
        label = self.imclasses[idx]
        #image = Image.open(img_path).convert('RGB')
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform is not None:
            #image = self.transform(image)
            image = self.transform(image=image)['image']
        
        return img_path, image, torch.tensor(label)

    def __len__(self):
        return len(self.imnames)

class GarbageBboxDataset(torch.utils.data.Dataset):
    def __init__(self, root, data_file, transform, mode='train'):
        self.root = root
        self.transform = transform
        self.mode = mode

        with open(data_file) as json_file:
            self.label_dic = json.load(json_file)

        imnames = []
        imclasses = []
        
        for ele in self.label_dic:

            if mode == 'test':
                imnames.append(ele['filename'])
            else:
                imnames.append(ele['filename'])
                imclasses.append(ele['bbox'])

        self.imnames = imnames
        self.imclasses = imclasses

    def __getitem__(self, idx):

        img_path = os.path.join(self.root, self.imnames[idx])
        if self.mode != 'test':
            label = self.imclasses[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            if self.mode == 'test':
                t = self.transform(image=image)
                image = t['image']
                image = np.transpose(image, (2, 0, 1)).astype(np.float32)
                return img_path, image, 0

            else:
                t = self.transform(image=image, bboxes = [label], category_ids = [0])
                image = t['image']
                bbox = t['bboxes']

                height = image.shape[0]
                width = image.shape[1]

                if len(bbox)>0:
                    bbox = bbox[0]
                    bbox = np.array([bbox[0]/width, bbox[1]/height, bbox[2]/width, bbox[3]/height])
                else:
                    bbox = np.array([0.0,0.0,0.0,0.0])

                image = np.transpose(image, (2, 0, 1)).astype(np.float32)

                return img_path, image, bbox

    def __len__(self):
        return len(self.imnames)