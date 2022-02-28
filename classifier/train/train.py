import os
import math
import datetime
import numpy as np
import time
import json
import shutil
import argparse
from pprint import pprint

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import neptune
import warnings


from GarbageNet.CLSmodeling import GarbageNet, MultiheadClassifier
from GarbageNet.efficientnet_fpn.model import FPNEfficientNet
from GarbageNet.optimization import RangerLars, AdamP
from GarbageNet.scheduler import GradualWarmupScheduler
from dataloader.preprocess import valpreprocess, randaugprocess, randaugafterprocess, bboxaugprocess, bboxvalprocess, clsalbtrain, clsalbval, albinversepreprocess
from dataloader.garbage import GarbageDataset, GarbageBboxDataset, GarbageDatasetwithBbox
from utils.util import get_model_params, rand_bbox, accuracy, set_seed, cutmix_criterion, cutmix_data, mixup_data, mixup_criterion, f1_score_multi, multi_label_acc, f1_loss, FocalLoss

INT_TO_LABEL = {0:"paper", 1:"pack", 2:"can", 3:"glass", 4:"pet", 5:"plastic", 6:"vinyl"}

warnings.filterwarnings("ignore")

def save_model(checkpoint_dir, model, optimizer, scheduler):
    state = {
        'model': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(state, os.path.join(checkpoint_dir + '.pth'))
    print('model saved')


def load_model(model_name, model, optimizer=None, scheduler=None):
    state = torch.load(os.path.join(model_name))
    model.load_state_dict(state['model'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(state['scheduler'])
    print('model loaded')

def make_parser():
    args = argparse.ArgumentParser()
    #exp
    args.add_argument("--pwd", type=str, default=os.getcwd())
    args.add_argument("--seed", type=int, default=777)
    args.add_argument("--no_save_model", action='store_true', default=False)
    args.add_argument("--save_only_best", action='store_true', default=False)
    args.add_argument("--output_dir", type=str, default='result')
    args.add_argument("--exp_name", type=str, default = 'baseline')    
    args.add_argument("--train_batch_size", type=int, default=512)
    args.add_argument("--eval_batch_size", type=int, default=512)
    args.add_argument("--cuda", type=bool, default=True)
    args.add_argument("--amp", type=bool, default=False)
    args.add_argument("--gpus", type=str, default='0')
    args.add_argument("--num_workers", type = int, default = 8)
    args.add_argument("--local_rank", type=int, default=-1)
    args.add_argument("--no_val", action='store_true', default=False)

    #dataset
    args.add_argument("--root", type=str, default="/home/ubuntu/Garbage-V2/dataset/garbage_v1")
    args.add_argument("--trainfile", type=str, default="dataset/garbage_v1/meta/all_combine_label_train.json")
    args.add_argument("--evalfile", type=str, default="dataset/garbage_v1/meta/all_combine_label_val.json")
    args.add_argument("--image_size", type=int, default=224)
    args.add_argument("--advprop", action='store_true')
    args.add_argument("--soft_label", action='store_true')
    args.add_argument("--no_cutout", action='store_true', default=False)
    args.add_argument("--resize_op", action='store_true', default=False)

    #model
    args.add_argument("--model_name", type=str, default="efficientnet-b0")
    args.add_argument("--num_classes", type=int, default=9)
    args.add_argument("--pretrained", action='store_true')
    args.add_argument("--transfer", type=str, default=None)
    
    #hparams
    args.add_argument("--lr", type=float, default=1e-4)
    args.add_argument("--num_epochs", type=int, default=100)
    args.add_argument("--val_same_epoch", type=int, default=20)
    args.add_argument("--weight_decay", type=float, default=1e-5)
    args.add_argument("--criterion", type=str, default="f1_loss")
    args.add_argument("--optim", type=str, default="rangerlars")
    args.add_argument("--scheduler", type=str, default="cosine")
    args.add_argument("--warmup", type=int, default=5)
    args.add_argument("--randaugment_n", type=int, default=2)
    args.add_argument("--randaugment_m", type=int, default=10)
    args.add_argument("--cutmix_alpha", type=float, default=0)
    args.add_argument("--cutmix_prob", type=float, default=0)
    args.add_argument("--mixup_alpha", type=float, default=0)
    args.add_argument("--mixup_prob", type=float, default=0)
    args.add_argument("--gradient_accumulation_steps", type=float, default=1)
    args.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    args.add_argument("--mode", type=str, default="multi")

    config = args.parse_args()
    return config

def main(args):
    # make dir
    #args_dic['pwd'] = os.getcwd()
    
    log_dir = os.path.join(args.output_dir, 'logs')
    log_dir = os.path.join(log_dir, args.exp_name)
    checkpoint_dir = os.path.join(args.output_dir, 'models')
    checkpoint_dir = os.path.join(checkpoint_dir, args.exp_name)
    print(checkpoint_dir)
    if os.path.exists(checkpoint_dir) or os.path.exists(log_dir) :
        flag_continue = input(f"Model name [{args.exp_name}] already exists. Do you want to overwrite? (y/n): ")
        if flag_continue.lower() == 'y' or flag_continue.lower() == 'yes':
            shutil.rmtree(log_dir)
            os.mkdir(log_dir)
            shutil.rmtree(checkpoint_dir)
            os.mkdir(checkpoint_dir)
        else:
            print("Exit pre-training")
            exit()
    else:
        os.mkdir(log_dir)
        os.mkdir(checkpoint_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    
    with open(os.path.join(checkpoint_dir,'exp_config.json'), 'w') as fp:
        json.dump(vars(args), fp, indent=4)
        
    # Start Experiment loading exp config
    with open(os.path.join(checkpoint_dir,'exp_config.json'), 'r') as f:
        exp_dict = json.load(f)
        parser = argparse.ArgumentParser()
        args = parser.parse_args("")
        args.__dict__.update(exp_dict)
        pprint(args)
    
    neptune.init('kboseong/Garbage')
    neptune.create_experiment(name = args.exp_name, params = vars(args), tags = [args.mode, args.root.split('/')[-1], os.environ["NODE_NUM"]], upload_source_files=['GarbageNet/modeling.py', 'dataloader/garbage.py', 'dataloader/preprocess.py', 'train.py'])

    #fix seed
    set_seed(args.seed)

    #make dataset
    if args.mode == 'multi' or args.mode=='multihead':
        '''train_preprocess = randaugprocess(args.image_size, args.advprop, args.no_cutout, args.resize_op, args.randaugment_n, args.randaugment_m)
        train_after_preprocess = randaugafterprocess(args.image_size, args.advprop, args.no_cutout, args.resize_op, args.randaugment_n, args.randaugment_m)
        val_preprocess = valpreprocess(args.image_size, args.advprop, args.resize_op)'''
        train_preprocess = clsalbtrain(args.image_size, args.advprop)
        train_after_preprocess = clsalbtrain(args.image_size, args.advprop, True)
        val_preprocess = clsalbval(args.image_size, args.advprop)
    elif args.mode == 'bbox':
        train_preprocess = bboxaugprocess(args.image_size, args.advprop)
        train_after_preprocess = bboxvalprocess(args.image_size, args.advprop)
        val_preprocess = bboxvalprocess(args.image_size, args.advprop)
    else: 
        raise ValueError('no supported exp mode')

    if args.mode == 'multi' or args.mode =='multihead':
        trainset = GarbageDatasetwithBbox(root = args.root, data_file = args.trainfile, transform = train_preprocess, soft_label = args.soft_label, mode='train')
        trainset_after = GarbageDatasetwithBbox(root = args.root, data_file = args.trainfile, transform = train_after_preprocess, soft_label = args.soft_label, mode='train')
        valset = GarbageDatasetwithBbox(root = args.root, data_file = args.evalfile, transform = val_preprocess, soft_label = args.soft_label, mode='eval')
    elif args.mode == 'bbox':
        trainset = GarbageBboxDataset(root = args.root, data_file = args.trainfile, transform = train_preprocess)
        trainset_after = GarbageBboxDataset(root = args.root, data_file = args.trainfile, transform = train_after_preprocess)
        valset = GarbageBboxDataset(root = args.root, data_file = args.evalfile, transform = val_preprocess)
    else: 
        raise ValueError('no supported exp mode')

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
    
    if args.val_same_epoch>0:
        trainloader_after = torch.utils.data.DataLoader(trainset_after, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers)
   
    #make model
    if args.model_name[:3] == 'fpn':
        model = FPNEfficientNet.from_name(args.model_name[3:])
    elif args.mode == 'multihead':
        model = MultiheadClassifier(args)
    else:
        model = GarbageNet(args)
    
    #model.load_state_dict(torch.load('result/models/final_efficientnetb6_multiBCE_01/090.pth')['model'])
    model = model.cuda() if args.cuda else model
    model = nn.DataParallel(model)

    # set criterion 
    if args.criterion == 'crossentropy':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.criterion == 'multiBCE':
        criterion = nn.BCEWithLogitsLoss()
    elif args.criterion == 'f1_loss':
        criterion = f1_loss()
    elif args.criterion == 'mse':
        criterion = torch.nn.MSELoss()
    elif args.criterion == 'l1':
        criterion = torch.nn.L1Loss() 
    elif args.criterion == 'focal':
        criterion = FocalLoss()
    else :
        raise ValueError('no supported loss function name')
        
    criterion = criterion.cuda() if args.cuda else criterion
    
    # set optimizer
    if args.optim == 'rangerlars':
        parameters = get_model_params(model, args.weight_decay)
        optimizer = RangerLars(parameters, lr=args.lr)
    elif args.optim == 'adamP':
        parameters = get_model_params(model, args.weight_decay)
        optimizer = AdamP(parameters, lr=args.lr)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) 
    else :
        raise ValueError('no supported optimizer name')

    #set scheduler:
    if args.scheduler =='StepLR':
        scheduler = StepLR(optimizer, step_size=int(args.num_epochs/4), gamma=0.1)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs+args.val_same_epoch, eta_min=0.)
    else :
        raise ValueError('no supported scheduler name')

    if args.warmup > 0:
        scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=args.warmup, after_scheduler=scheduler)
        scheduler = scheduler_warmup

    writer = SummaryWriter(log_dir)
    
    scaler = torch.cuda.amp.GradScaler()

    num_batches = len(trainloader)
    
    #check parameter of model
    print("------------------------------------------------------------")
    total_params = sum(p.numel() for p in model.parameters())
    print("num of parameter : ",total_params)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("num of trainable_ parameter :",trainable_params)
    print("num batches :",num_batches)
    print("------------------------------------------------------------")
    
    # train
    global_iter = 0
    val_global_iter = 0
    best_metric = 1000
    
    for epoch in range(args.num_epochs+args.val_same_epoch):
        # -------------- train -------------- #
        model.train()
        epoch_loss = []
        if epoch < args.num_epochs:
            train_iter_obj = iter(trainloader) 
        else :
            if epoch == args.num_epochs and args.mode!='bbox':
                print('freeze backbone model weight excpet fc layer')
                #model.freeze_backbone()
            train_iter_obj = iter(trainloader_after)
            

        t1 = time.time()
        for iter_ in range(len(trainloader)):
            
            train_data = train_iter_obj.next()
            # fetch train data
            _, image, label = train_data
            #print(image.shape)
            if args.cuda:
                image = image.cuda()
                label = label.cuda()

            # cut mix
            if args.mode == 'bbox':
                pred = model(image)
                loss = criterion(pred, label)
            
            elif args.mode == 'multihead':
                
                pred = model(image)
                label = torch.transpose(label, 0,1)

                combine_loss = 0
                for i in range(args.num_classes):
                    loss = criterion(pred[i], label[i].type(torch.long))
                    combine_loss+= loss
                
                loss = combine_loss/args.num_classes

            else:
                with torch.cuda.amp.autocast():
                    r = np.random.rand(1)
                    if args.cutmix_alpha > 0 and r < args.cutmix_prob:
                        image, target_a, target_b, lam = cutmix_data(image, label, args.cutmix_alpha)
                        pred = model(image)
                        loss = cutmix_criterion(criterion, pred, target_a, target_b, lam)
                    elif args.mixup_alpha > 0 and r < args.mixup_prob:
                        image, target_a, target_b, lam  = mixup_data(image,label, args.mixup_alpha)
                        pred = model(image)
                        loss = mixup_criterion(criterion, pred, target_a, target_b, lam)
                    else:
                        # compute pred
                        pred = model(image)
                        loss = criterion(pred, label)

                
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                
            # loss backward
            scaler.scale(loss).backward()
            
            epoch_loss.append(float(loss))
            print('Epoch: {} | Iteration: {} | Running loss: {:1.5f}'.format(epoch, iter_, np.mean(np.array(epoch_loss))))

            if (iter_ + 1) % args.gradient_accumulation_steps == 0:
                if args.amp:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            
            t2 = time.time()

            #logging
            writer.add_scalar('Loss/train', loss, global_iter)
            writer.add_scalar('Speed/train', (t2-t1)/(args.train_batch_size*(iter_+1)) , global_iter)
            neptune.log_metric('train_loss',x = global_iter, y=loss.item())
            neptune.log_metric('train_speed',x = global_iter, y=(t2-t1)/(args.train_batch_size*(iter_+1)))

            global_iter+=1
            #break
        # scheduler update
        if epoch < args.num_epochs:
            scheduler.step()
        
        if args.no_val :
            model_save_dir = os.path.join(checkpoint_dir, f'{(epoch + 1):03}')
            save_model(model_save_dir, model, optimizer, scheduler)
            continue
        # -------------- validation -------------- #
        model.eval()
        val_loss_list, pred_list, label_list = [],[],[]

        t1 = time.time()
        
        with torch.no_grad():
            for iter_, val_data in enumerate(valloader):
                
                _, image, label = val_data
                if args.cuda:
                    image = image.cuda()
                    label = label.cuda()

                pred = model(image)
                
                if args.mode == 'bbox':
                    pred_list.append(pred.detach().cpu().numpy())
                    label_list.append(label.detach().cpu().numpy())
                    loss = criterion(pred, label)
                elif args.mode == 'multihead':
                    pred = model(image)

                    pred_list += list(torch.transpose(torch.argmax(torch.stack(pred), dim=2),0,1).detach().cpu().numpy())
                    label_list += list(label.detach().cpu().numpy()) 

                    label = torch.transpose(label, 0,1)

                    combine_loss = 0
                    for i in range(args.num_classes):
                        loss = criterion(pred[i], label[i].type(torch.long))
                        combine_loss+= loss
                    
                    loss = combine_loss/args.num_classes

                else:
                    pred_list += list(torch.round(torch.sigmoid(pred)).detach().cpu().numpy())
                    label_list += list(label.detach().cpu().numpy()) 
                    loss = criterion(pred, label)
            
                val_loss_list.append(float(loss))

                print('Val_Epoch: {} | iter : {} | val_loss : {:1.5f}' .format(epoch, iter_, float(loss)))  
                t2=time.time()
                writer.add_scalar('Speed/val', (t2-t1)/(args.eval_batch_size*(iter_+1)) , val_global_iter)
                neptune.log_metric('val_speed',x = val_global_iter, y=(t2-t1)/(args.train_batch_size*(iter_+1)))
                val_global_iter+=1
                #break

        #print & logging
        val_loss_mean = np.mean(np.array(val_loss_list))
        flag = False

        if args.mode == 'bbox':
            p = np.concatenate(pred_list)
            y = np.concatenate(label_list)
            mae = mean_absolute_error(y,p)*args.image_size

            if best_metric>=mae:
                flag=True
                best_metric = mae
            else:
                flag=False

            print('Epoch: {} | loss : {:1.5f} | mae : {:1.5f}'.format(epoch, val_loss_mean, mae))

            writer.add_scalar('Loss/val', val_loss_mean, epoch)
            writer.add_scalar('Metric/mae', mae, epoch)
            writer.add_scalar('Satus/lr', optimizer.param_groups[0]['lr'] , epoch)

            neptune.log_metric('val_loss',x = epoch, y=val_loss_mean)
            neptune.log_metric('mae',x = epoch, y=mae)
            neptune.log_metric('lr',x = epoch, y=optimizer.param_groups[0]['lr'])

        else:
            f1_scores, weight_f1, mean_f1 = f1_score_multi(label_list, pred_list, average='weighted')
            val_acc = multi_label_acc(label_list, pred_list)

            #print(hamming_metric)
            print('Val_Epoch: {} | iter : {} | val_loss : {:1.3f} | f1 : {:1.3f} | acc : {:1.3f} '.format(epoch, iter_, float(val_loss_mean), weight_f1, val_acc))
            
            writer.add_scalar('Metric/f1_avg', weight_f1 , epoch)
            writer.add_scalar('Metric/f1_mean', mean_f1, epoch)
            writer.add_scalar('Loss/val', val_loss_mean, epoch)
            writer.add_scalar('Acc/val', val_acc, epoch)
            writer.add_scalar('Satus/lr', optimizer.param_groups[0]['lr'] , epoch)

            neptune.log_metric('f1_avg',x = epoch, y=weight_f1)
            neptune.log_metric('f1_mean',x = epoch, y=mean_f1)
            neptune.log_metric('val_loss',x = epoch, y=val_loss_mean)
            neptune.log_metric('acc',x = epoch, y=val_acc)
            neptune.log_metric('lr',x = epoch, y=optimizer.param_groups[0]['lr'])

            for i, score in enumerate(f1_scores):
                label_name = INT_TO_LABEL[i]
                writer.add_scalar('Details/{}_f1'.format(label_name), score, epoch)
                neptune.log_metric('{}_f1'.format(label_name),x = epoch, y=score)
            
        # save model
        if epoch == 0:
            model_save_dir = os.path.join(checkpoint_dir, f'{(epoch + 1):03}')
            save_model(model_save_dir, model, optimizer, scheduler)
        if not args.no_save_model:
            if args.save_only_best:
                if flag:
                    try:
                        os.remove(model_save_dir+'.pth')
                    except Exception as e:
                        print(e)
                        continue
                    model_save_dir = os.path.join(checkpoint_dir, f'{(epoch + 1):03}')
                    save_model(model_save_dir, model, optimizer, scheduler)
                    
            else:
                model_save_dir = os.path.join(checkpoint_dir, f'{(epoch + 1):03}')
                save_model(model_save_dir, model, optimizer, scheduler)

            

    print('finish')

if __name__ == '__main__':
    # mode argument
    args = make_parser()
    main(args)