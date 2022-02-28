import sys
import torch
import cv2 
import numpy as np
import albumentations as A
import argparse
from albumentations.pytorch import ToTensorV2

from utils.inference_util import bboxtestprocess, clsalbval, postprocessing

int_to_label = {0:"종이", 1:"종이팩", 2:"알류미늄캔", 3:"유리", 4:"페트", 5:"플라스틱", 6:"비닐"}
bbox_image_size = 456
bbox_margin = 1/8
bbox_device = 'cpu'
cls_device = 'cpu'
cls_image_size = 456
cls_advprop = True

def inference(config):
    bbox_preprocess = bboxtestprocess(bbox_image_size)
    cls_preprocess = clsalbval(cls_image_size, cls_advprop, devide = False)

    bbox_model = torch.jit.load(config.bbox_model_path, map_location = bbox_device)
    cls_model = torch.jit.load(config.cls_model_path, map_location = cls_device)

    #origin_image = cv2.imread('../dataset/garbage_v2/GOPR5444.jpg')
    origin_image = cv2.imread(config.filename)
    origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
    image = bbox_preprocess(image = origin_image)['image']

    with torch.no_grad():
        bboxes = bbox_model(torch.unsqueeze(image, 0).to(device = bbox_device))[0]
        bboxes = np.array([[bboxes[0]*bbox_image_size, bboxes[1]*bbox_image_size, bboxes[2]*bbox_image_size, bboxes[3]*bbox_image_size]]).astype(np.float32).tolist()
        multiply = max(origin_image.shape)/bbox_image_size
        origin_bboxes = (np.array(bboxes)*multiply).astype(np.int64).tolist()
        origin_bbox, new_bboxes = postprocessing(origin_image.shape, origin_bboxes[0][:], bbox_margin)
        cuted_image = origin_image[new_bboxes[1]:new_bboxes[1]+new_bboxes[3],new_bboxes[0]:new_bboxes[0]+new_bboxes[2],:]

    cls_input = torch.unsqueeze(cls_preprocess(image = cuted_image)['image'], 0).to(device = cls_device)

    output = torch.sigmoid(cls_model(cls_input))
    rounded_output = torch.round(output)[0].detach().cpu()

    """for i,val in enumerate(rounded_output):
        if int(val)==1: 
            print(int_to_label[i])"""

    print(', '.join([int_to_label[i] for i,val in enumerate(rounded_output) if int(val)==1]))

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    #exp
    args.add_argument("--filename", type=str, default = 'dataset/garbage_v2/GOPR5444.jpg')
    args.add_argument("--bbox_model_path", type=str, default = 'modelweights/garbage_bbox_jit.pth')
    args.add_argument("--cls_model_path", type=str, default = 'modelweights/garbage_cls_jit.pth')

    config = args.parse_args()
    inference(config)

