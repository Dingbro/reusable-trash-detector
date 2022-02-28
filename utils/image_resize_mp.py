from multiprocessing import Pool
import requests
import time
import json
from tqdm import tqdm
import glob
import numpy as np
import cv2
import albumentations as A

NUM_PROCESS = 16

def split_list(l, n):
    r = []
    t = 0
    while t < len(l)-n:
        r.append(l[t:t+n])
        t+=n
    r.append(l[t:])
    return r


split_filenames = glob.glob('/home/ubuntu/Garbage-V2/yolov5/data/garbage_final/v5/images/*/*.jpg')
split_outputs = split_list(split_filenames, int((len(split_filenames)/NUM_PROCESS) + 1))
print(len(split_outputs), len(split_outputs[0]))

def worker(filenames):
    transform = A.LongestMaxSize(1280, interpolation = cv2.INTER_AREA)
    for i, ele in tqdm(enumerate(filenames)):
        try:
            image = cv2.imread(ele)
            image = transform(image=image)['image']
            cv2.imwrite(ele,image)
        except:
            continue
            

pool = Pool(processes=NUM_PROCESS) 
pool.map(worker, split_outputs) 
