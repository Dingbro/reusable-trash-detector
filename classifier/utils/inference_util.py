import sys
import torch
import cv2 
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

def postprocessing(shape, bboxes, margin=1/2):
    
    height = shape[0]
    width = shape[1]
    
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
    
    #margin = max(bbox_width, bbox_height)*margin
    #margin = min(max(bbox_width, bbox_height)*margin, 10)
    margin = 10
    
    x_min = max(x_min-margin/2, 0)
    y_min = max(y_min-margin/2, 0)
    bbox_width = min(bbox_width+margin, width-x_min)
    bbox_height = min(bbox_height+margin, height-y_min)
    
    final_bbox = np.array([x_min, y_min, bbox_width, bbox_height]).astype(np.int64).tolist()
    
    return origin_bbox, final_bbox

def bboxtestprocess(size):
    nomalize = A.Normalize()

    albumentations_transform = A.Compose([
        A.LongestMaxSize(size, interpolation = cv2.INTER_AREA),
        A.PadIfNeeded(size, size, value=(255,255,255), border_mode=0),
        A.Resize(size,size, interpolation = cv2.INTER_AREA),
        nomalize,
        ToTensorV2()
    ]
    )
    return albumentations_transform

class advprop_normal_divide(A.ImageOnlyTransform):
    def __init__(self, a=2.0, b=1.0, always_apply=False, p=1):
        super().__init__(always_apply=always_apply, p=p)
        self.a = a
        self.b = b

    def apply(self, img, **params):
        #return (img.astype(np.float32)/255) * 2.0 - 1.0
        return (img.astype(np.float32)/255) * 2.0 - 1.0

    def get_transform_init_args_names(self):
        return ('a', 'b')


class advprop_normal(A.ImageOnlyTransform):
    def __init__(self, a=2.0, b=1.0, always_apply=False, p=1):
        super().__init__(always_apply=always_apply, p=p)
        self.a = a
        self.b = b

    def apply(self, img, **params):
        #return (img.astype(np.float32)/255) * 2.0 - 1.0
        return (img.astype(np.float32)) * 2.0 - 1.0

    def get_transform_init_args_names(self):
        return ('a', 'b')

def clsalbval(size, advprop, devide):
    if advprop:
        if devide:
            nomalize = advprop_normal_divide()
        else:
            nomalize = advprop_normal()
    else:
        nomalize = A.Normalize()

    albumentations_transform = A.Compose([
        A.SmallestMaxSize(size, interpolation = cv2.INTER_AREA),
        A.CenterCrop(size,size, p=1),
        nomalize,
        ToTensorV2()
    ])
    return albumentations_transform