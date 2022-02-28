import numpy as np
import random
import torch
import torch.nn.functional as F

def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_data(image, target, beta):
    
    lam = np.random.beta(beta, beta)
    rand_index = torch.randperm(image.size()[0]).cuda()
    target_a = target
    target_b = target[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(image.size(), lam)
    image[:, :, bbx1:bbx2, bby1:bby2] = image[rand_index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image.size()[-1] * image.size()[-2]))

    return image, target_a, target_b, lam

def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    return criterion(pred, y_a) * lam + criterion(pred, y_b) * (1. - lam)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1. / batch_size))
    return res

def get_model_params(model, weight_decay):
    no_decay = ["bias", "BatchNorm2d.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters

def f1_score_multi(label_list, pred_list, average='weighted'):
    
    label_list = np.array(label_list)
    pred_list = np.array(pred_list)
    
    label_num_info = sum(label_list)
    
    confusion = []
    for i in range(len(label_list[0])):
        confusion.append([0,0,0,0])
        
    a = (label_list*2-pred_list)+1
    for ele in a:
        for i in range(len(ele)):
            confusion[i][int(ele[i])]+=1
    #print(confusion)
    f1_scores = []
    for i, ele in enumerate(confusion):
        fp, tn, tp, fn = ele
        
        if tp+fp==0:precision=0
        else: precision = tp/(tp+fp)
        
        if tp+fn==0:recall = 0
        else: recall = tp/(tp+fn)
        
        if recall+precision==0: f1=0
        else:f1 = 2*(recall * precision) / (recall + precision)
        f1_scores.append(f1)
    
    avg_f1=0
    for i, f1 in enumerate(f1_scores):
        avg_f1 += label_num_info[i]*f1
    weighted_f1 = avg_f1/sum(label_num_info)
    mean_f1 = np.array(f1_scores).mean()
    return f1_scores, weighted_f1, mean_f1

def multi_label_acc(label_list, pred_list):
    acc_list = []
    for i in range(len(label_list)):
        
        acc_temp = sum(label_list[i] == pred_list[i])/len(label_list[i])
        #acc_temp = sum(label_list[i] == pred_list[i])/len(label_list[i])
        acc_list.append(acc_temp)
    
    return np.array(acc_list).mean()

class f1_loss(torch.nn.Module):
    def __init__(self, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true,):
        
        
        y_pred = torch.nn.functional.sigmoid(y_pred)
        
        tp = (y_true * y_pred).sum(dim=0)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0)
        fp = ((1 - y_true) * y_pred).sum(dim=0)
        fn = (y_true * (1 - y_pred)).sum(dim=0)

        f1 = 2* ((tp / (tp + fp + self.epsilon))*(tp / (tp + fn + self.epsilon))) / ((tp / (tp + fp + self.epsilon)) + (tp / (tp + fn + self.epsilon)) + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        return 1 - f1.mean()

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
        
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