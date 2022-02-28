import os
import argparse
from pprint import pprint
import json
from dataloader.preprocess import valpreprocess, randaugprocess, randaugafterprocess, bboxaugprocess, bboxvalprocess, bboxtestprocess
from dataloader.garbage import GarbageDataset, GarbageBboxDataset
from GarbageNet.efficientnet_fpn.model import FPNEfficientNet
from GarbageNet.CLSmodeling import GarbageNet, MultiheadClassifier
from sklearn.metrics import mean_absolute_error

import torch
import glob
import shutil
import albumentations as A
import matplotlib.pyplot as plt
import random
import cv2
import numpy as np
from tqdm import tqdm
from apex import amp
import time


from utils.util import multi_label_acc, f1_score_multi

with open(os.path.join('result/models/bbox_1103_fpnefficientnetb5_456_l1_real_val_lr_1e-3_adam_03','exp_config.json'), 'r') as f:
    exp_dict = json.load(f)
    parser = argparse.ArgumentParser()
    args = parser.parse_args("")
    args.__dict__.update(exp_dict)
    pprint(args)
#args.model_name = 'rexnet2.0'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

val_preprocess = bboxtestprocess(args.image_size, args.advprop)
valset = GarbageBboxDataset(root = args.root, data_file = args.evalfile, transform = val_preprocess, mode = 'test')
valloader = torch.utils.data.DataLoader(valset, batch_size=args.eval_batch_size, shuffle=False, num_workers=8)

if args.model_name[:3] == 'fpn':
    model = FPNEfficientNet.from_name(args.model_name[3:])
    print('fpn load')
else:
    model = GarbageNet(args)
    print('no fpn load')

load_epoch = 320

model.load_state_dict(torch.load('result/models/{}/{:03d}.pth'.format(args.exp_name, load_epoch))['model'])
model.eval()
model = model.to(torch.device("cuda"))
#mode = amp.initialize(model, opt_level='O1')
time.sleep(10)

torch.cuda.empty_cache()
print('no cuda')

time.sleep(20)