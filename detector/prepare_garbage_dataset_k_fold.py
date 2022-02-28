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
    "s3://drcha-ai-datalake/labelstudio/reusable-trash-detector/v3/outputs/"
]
LS_DOWNLOAD_LOCAL_PATHS = [
    "/home/ubuntu/drcha-ai-datalake/labelstudio/reusable-trash-detector/v1/outputs/",
    "/home/ubuntu/drcha-ai-datalake/labelstudio/reusable-trash-detector/v2/outputs/",
    "/home/ubuntu/drcha-ai-datalake/labelstudio/reusable-trash-detector/v3/outputs/",
]

K_FOLD = 5

ROOT = "data/garbage_04"
SOURCE_IMAGE_PATH = "/home/ubuntu/Garbage-V2/GarbageNet/dataset/images"
FOLD_PREFIX = "fold"

#TARGET_IMAGE_PATH = os.path.join(ROOT, 'images')
#TARGET_LABEL_PATH = os.path.join(ROOT, 'labels')
#TRAIN_IMAGE_META_FILE = os.path.join(ROOT, "train_03.txt")
#VAL_IMAGE_META_FILE = os.path.join(ROOT, "val_03.txt")

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
        for i in range(K_FOLD):
            try:
                shutil.rmtree(os.path.join(ROOT, '{}_{}'.format(FOLD_PREFIX, i+1)))
            except:
                pass


    all_file_annots = {}

    for ele in all_output_annotations:
        label_info = {'train':{LS_INT_TO_LABEL[ele]:0 for ele in LS_INT_TO_LABEL}, 'val':{LS_INT_TO_LABEL[ele]:0 for ele in LS_INT_TO_LABEL}}
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

    print('all annots files :{}'.format(len(all_file_annots.keys())))

    all_file_annots = [[ele, all_file_annots[ele]]for ele in all_file_annots]
    
    random.shuffle(all_file_annots)

    partition_num = int(len(all_file_annots)/K_FOLD)
    k_fold_list = [all_file_annots[i*partition_num:(i+1)*partition_num] for i in range(K_FOLD)]
    
    for i in range(K_FOLD):
        TRAIN_IMAGE_META_FILE = os.path.join(ROOT, "train_{:02d}.txt".format(i+1))
        VAL_IMAGE_META_FILE = os.path.join(ROOT, "val_{:02d}.txt".format(i+1))
        TARGET_IMAGE_PATH = os.path.join(os.path.join(ROOT, '{}_{}'.format(FOLD_PREFIX, i+1)), 'images')
        TARGET_LABEL_PATH = os.path.join(os.path.join(ROOT, '{}_{}'.format(FOLD_PREFIX, i+1)), 'labels')
        try:
            os.makedirs(os.path.join(TARGET_IMAGE_PATH,'train'))
            os.makedirs(os.path.join(TARGET_IMAGE_PATH,'val'))
            os.makedirs(os.path.join(TARGET_LABEL_PATH,'train'))
            os.makedirs(os.path.join(TARGET_LABEL_PATH,'val'))
        except:
            pass
        
        val_list = k_fold_list[i]
        temp = [0,1,2,3,4]
        temp.remove(i)
        train_list = []
        for j in range(K_FOLD):
            if i==j:
                continue
            train_list+=k_fold_list[j]

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