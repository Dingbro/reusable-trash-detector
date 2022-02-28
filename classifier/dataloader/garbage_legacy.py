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

int_to_label = {0:"general", 1:"paper", 2:"can", 3:"glass", 4:"plastic", 5:"vinyl", 6:"styrofoam", 7:"food", 8:"food_none"}
label_to_int = {int_to_label[ele]: ele for ele in int_to_label}
cls_label_to_int = {"A":0, "B":1}

class GarbageDataset(torch.utils.data.Dataset):
    def __init__(self, root, data_file, transform):
        self.root = root
        self.transform = transform

        with open(data_file) as json_file:
            self.label_dic = json.load(json_file)

        imnames = []
        imclasses = []
        
        for img_name in self.label_dic:
            imnames.append(img_name)
            temp = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
            for t in self.label_dic[img_name]:
                temp[label_to_int[t]]=1.0
            imclasses.append(temp)

        self.imnames = imnames
        self.imclasses = imclasses

    def __getitem__(self, idx):

        img_path = os.path.join(self.root, self.imnames[idx])
        label = self.imclasses[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
        
        return img_path, image, torch.tensor(label)

    def __len__(self):
        return len(self.imnames)