import numpy as np

def f1_competition(preds,labels,average='macro', num_classes = 7):
    """
        preds : sample_num*num_classes
        labels : sample_num*num_classes
    """
    
    label_info = {i:{} for i in range(num_classes)}
    for label in labels:
        for i in range(num_classes):
            if label[i]==0:
                continue
            label_info[i][label[i]]=[0,0,0]# tp, fp, fn
    #print(label_info)
            
    for pred,label in zip(preds,labels):
        for i in range(num_classes):
            a_label = label[i] 
            a_pred = pred[i]

            if a_label==0 and a_pred>0:
                if a_pred in label_info[i].keys():
                    label_info[i][a_pred][1] += 1 # fp +1
            elif a_label>0 and a_label==a_pred:
                label_info[i][a_label][0] += 1 # tp +1
            elif a_label>0 and a_pred>0 and a_label!=a_pred:
                label_info[i][a_label][2] += 1 # fn +1
                if a_pred in label_info[i].keys():
                    label_info[i][a_pred][1] +=1 #fp +1
            elif a_label>0 and a_pred==0:
                label_info[i][a_label][2] += 1 # fn +1

    class_f1 = [0]*num_classes

    for i in range(num_classes):
        info = label_info[i]
        all_num=0
        one_f1 = 0
        for num in info:
            tp, fp, fn = info[num]
            try:
                recall = tp/(tp+fn)
                precision = tp/(tp+fp)
                f1 = 2*(recall*precision)/(recall+precision)
            except:
                precision = 0
                recall = 0
                f1=0
            #print(num, precision,recall)
            weighted_f1 = f1*(tp+fn)
            all_num+=(tp+fn)
            one_f1+= weighted_f1
        
        class_f1[i]=one_f1/all_num
    
    if average=='macro':
        return class_f1, np.array(class_f1).sum()/len(class_f1)

def f1_number_multi(preds,labels,average='macro', num_classes = 7):
    
    class_dict = {i:[0,0,0,0] for i in range(7)}#tp, tn, fp, fn = 0,0,0,0
    for pred,label in zip(preds,labels):
        for i, (pred_i, label_i) in enumerate(zip(pred,label)):
            if label_i>0 and label_i==pred_i:
                class_dict[i][0]+=1
            elif label_i>0 and label_i!=pred_i:
                class_dict[i][3]+=1
            elif label_i==0 and label_i==pred_i:
                class_dict[i][1]+=1
            else:
                class_dict[i][2]+=1
    class_recall = {ele:class_dict[ele][0]/(class_dict[ele][0]+class_dict[ele][3]) for ele in class_dict}
    class_precision = {ele:class_dict[ele][0]/(class_dict[ele][0]+class_dict[ele][2]) for ele in class_dict}
    class_f1 = {ele:2*(class_recall[ele]*class_precision[ele])/(class_recall[ele]+class_precision[ele]) for ele in class_dict}
    if average=='macro':
        f1 = sum([class_f1[ele] for ele in class_f1])/len(class_f1.keys())
    return f1