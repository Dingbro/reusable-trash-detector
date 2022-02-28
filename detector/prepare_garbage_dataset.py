import boto3
import os
import shutil
import glob
import json
import random
import argparse
import subprocess


AWS_ACCESS_KEY_ID = "AKIA4URNBZMTTKGQNZKK"
AWS_SECRET_ACCESS_KEY = "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"
AWS_DEFAULT_REGION = "ap-northeast-2"
os.environ['AWS_ACCESS_KEY_ID'] = AWS_ACCESS_KEY_ID
os.environ['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_ACCESS_KEY
os.environ['AWS_DEFAULT_REGION'] = AWS_DEFAULT_REGION

OUTPUT_FOLDERS = [
    "s3://drcha-ai-datalake/labelstudio/reusable-trash-detector/v1/outputs/",
    "s3://drcha-ai-datalake/labelstudio/reusable-trash-detector/v2/outputs/",
    "s3://drcha-ai-datalake/labelstudio/reusable-trash-detector/v2/outputs/"
]
LS_DOWNLOAD_LOCAL_PATHS = [
    "data/garbage_04/ls/v1/outputs/",
    "data/garbage_04/ls/v2/outputs/",
    "data/garbage_04/ls/v3/outputs/",
]
TRAIN_SPLIT_RATIO = 0.8

ROOT = "data/garbage_04"
SOURCE_IMAGE_PATH = "/home/ubuntu/Garbage-V2/GarbageNet/dataset/images"
TARGET_IMAGE_PATH = os.path.join(ROOT, 'images')
TARGET_LABEL_PATH = os.path.join(ROOT, 'labels')

TRAIN_IMAGE_META_FILE = os.path.join(ROOT, "train_03.txt")
VAL_IMAGE_META_FILE = os.path.join(ROOT, "val_03.txt")

LS_LABELS_TO_INT = {'종이':0, '종이팩':1, '철, 캔류':2, '유리':3, '페트':4, '플라스틱 컵':5, '플라스틱':5, '비닐':6}
LS_INT_TO_LABEL = {LS_LABELS_TO_INT[ele]:ele for ele in LS_LABELS_TO_INT}

def main(config):
    random.seed(config.seed)

    if config.download:
        for i in range(len(OUTPUT_FOLDERS)): 
            print(OUTPUT_FOLDERS[i],LS_DOWNLOAD_LOCAL_PATHS[i] )
            subprocess.run(["aws", "s3", "cp", OUTPUT_FOLDERS[i], LS_DOWNLOAD_LOCAL_PATHS[i], '--recursive'])

    all_output_files = []
    for LS_DOWNLOAD_LOCAL_PATH in LS_DOWNLOAD_LOCAL_PATHS:
        all_output_files += glob.glob("{}/*".format(LS_DOWNLOAD_LOCAL_PATH))

    all_output_annotations = []
    for ele in all_output_files:
        try:
            with open(ele, 'r') as json_file:
                all_output_annotations.append(json.load(json_file))
        except:
            continue

    if config.reset:
        try:
            shutil.rmtree(TARGET_IMAGE_PATH)
        except:
            pass
        try:
            shutil.rmtree(TARGET_LABEL_PATH)
        except:
            pass
        try:
            shutil.rm(TRAIN_IMAGE_META_FILE)
        except:
            pass
        try:
            shutil.rm(VAL_IMAGE_META_FILE)
        except:
            pass

        os.mkdir(TARGET_IMAGE_PATH)
        os.mkdir('{}/{}'.format(TARGET_IMAGE_PATH,'train'))
        os.mkdir('{}/{}'.format(TARGET_IMAGE_PATH,'val'))
        os.mkdir(TARGET_LABEL_PATH)
        os.mkdir('{}/{}'.format(TARGET_LABEL_PATH,'train'))
        os.mkdir('{}/{}'.format(TARGET_LABEL_PATH,'val'))

    all_file_annots = {}

    label_info = {'train':{LS_INT_TO_LABEL[ele]:0 for ele in LS_INT_TO_LABEL}, 'val':{LS_INT_TO_LABEL[ele]:0 for ele in LS_INT_TO_LABEL}}

    for ele in all_output_annotations:
        all_results = ele['result']
        if ele['was_cancelled']:
            continue
        filename = ele['task']['data']['image'].split('/')[-1]
        rectangle_data=[]
        flag=True
        for r in all_results:
            if r['type'] in ['rectanglelabels','labels']:
                #print(ele)
                if r['type']== 'rectanglelabels':
                    class_name = r['value']['rectanglelabels'][0]
                elif r['type']== 'labels':
                    class_name = r['value']['labels'][0]
                x,y,width,height = r['value']['x']/100, r['value']['y']/100, r['value']['width']/100, r['value']['height']/100

                if x<0 or y<0 or x+width>1 or y+height>1:
                    flag=False

                bbox = [x+width/2, y+height/2, width,height]

                rectangle_data.append([
                    LS_LABELS_TO_INT[class_name],
                    bbox[0],bbox[1],bbox[2],bbox[3]
                ])


        if len(rectangle_data)>0 and flag:
            all_file_annots[filename] = rectangle_data


    all_file_annots = [[ele, all_file_annots[ele]]for ele in all_file_annots]
    

    random.shuffle(all_file_annots)
    train_list = all_file_annots[:int(len(all_file_annots)*TRAIN_SPLIT_RATIO)]
    val_list = all_file_annots[int(len(all_file_annots)*TRAIN_SPLIT_RATIO):]

    print('number of images:{}, train:{}, val:{}'.format(len(all_file_annots), len(train_list), len(val_list)))

    for ele in train_list:
        for annot in ele[1]:
            label_info['train'][LS_INT_TO_LABEL[annot[0]]]+=1
    for ele in val_list:
        for annot in ele[1]:
            label_info['val'][LS_INT_TO_LABEL[annot[0]]]+=1

    train_file_list = []
    val_file_list = []

    for ele in train_list:
        filename = ele[0]
        annots = [[str(t2) for t2 in t] for t in ele[1]]
        annots_strs = '\n'.join(['\t'.join(annot) for annot in annots])

        image_saved_dir = '{}/{}/{}'.format(TARGET_IMAGE_PATH,'train',filename)
        train_file_list.append(image_saved_dir)
        
        shutil.copy2('{}/{}'.format(SOURCE_IMAGE_PATH,filename), image_saved_dir)
        with open('{}/{}/{}'.format(TARGET_LABEL_PATH, 'train', filename.replace('.jpg','.txt')),'w') as text_file:
            text_file.write(annots_strs)

    for ele in val_list:
        filename = ele[0]
        annots = [[str(t2) for t2 in t] for t in ele[1]]
        annots_strs = '\n'.join(['\t'.join(annot) for annot in annots])

        image_saved_dir = '{}/{}/{}'.format(TARGET_IMAGE_PATH,'val',filename)
        val_file_list.append(image_saved_dir)
        
        shutil.copy2('{}/{}'.format(SOURCE_IMAGE_PATH,filename), image_saved_dir)
        with open('{}/{}/{}'.format(TARGET_LABEL_PATH, 'val', filename.replace('.jpg','.txt')),'w') as text_file:
            text_file.write(annots_strs)

    train_file_list_str = '\n'.join(train_file_list)
    val_file_list_str = '\n'.join(val_file_list)

    with open(TRAIN_IMAGE_META_FILE, 'w') as text_file:
        text_file.write(train_file_list_str)

    with open(VAL_IMAGE_META_FILE, 'w') as text_file:
        text_file.write(val_file_list_str)

    print(label_info)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--seed", type=int, default=777)
    args.add_argument("--download", action="store_true")
    args.add_argument("--reset", action="store_true")
    config = args.parse_args()

    main(config)