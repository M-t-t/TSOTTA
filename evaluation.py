import torch
from torch import nn
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F


def evaluation(predic,truth):
    TP,TN,FP=0,0,0
    # -------Accurancy
    acc = (predic.argmax(1)== truth).sum()
    #-----sensitivity FPRweish为
    predict=predic.argmax(1)
    for i in range(len(truth)):
        if truth[i]==1:
            if predict[i]==truth[i] :
                TP=TP+1
        else:
            if predict[i]==0 :
                TN=TN+1
            if predict[i]==1 :
                FP=FP+1

    # total_P = torch.count_nonzero(truth)
    # total_N = len(truth) - total_P
    # sensitivity = TP / total_P
    # FPR = FP / total_N
    # print('sensitivity, FPR',sensitivity, FPR)
    return acc,TP,FP ,TN


def evaluation1(predic,truth,eps):
    TP,auc,FP=0,0,0

    # -------Accurancy
    acc = (predic.argmax(1)== truth).sum()
    # -------AUC
    soft_output = nn.Softmax(dim=1)
    preprob = soft_output(predic)
    y_scores = preprob[:, 1]  # 取预测标签为1的概率
    try:
        auc = roc_auc_score(truth.cpu(), y_scores.cpu())  # 评价模型预测概率与期望概率之间的差异，类似交叉熵
    except ValueError:
        pass
    #-----sensitivity FPR
    predict=predic.argmax(1)
    for i in range(len(truth)):
        if predict[i]==truth[i] and truth[i]==1:
            TP=TP+1
        if predict[i]==1 and truth[i]==0:
            FP=FP+1
    print('TP, FP',TP, FP)

    total_P=torch.count_nonzero(truth)
    total_N=len(truth)-total_P
    sensitivity=TP/(total_P+eps)
    FPR=FP/(total_N+eps)
    # print('total_P' ,total_P)
    return acc,auc,sensitivity, FPR

import math
def accuracy(model: nn.Module,x: torch.Tensor, y: torch.Tensor, batch_size: int = 1, device: torch.device = None,mode=None):
    # setup_seed(42)
    if device is None:
        device = x.device
    TP, FP,Acc=0,0,0
    nfold_logit,nfold_label=[],[]
    n_batches = math.ceil(x.shape[0] / batch_size)
    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) *
                       batch_size].to(device)
            y_curr = y[counter * batch_size:(counter + 1) *
                       batch_size].to(device)
            if mode=='src':
                _,output = model(x_curr)
            else:
                output = model(x_curr)

            nfold_logit.append(output)
            nfold_label.append(y_curr)
            #---evaluation acc / sen and FPR
        nfold_logit=torch.cat(nfold_logit,dim=0)
        nfold_label = torch.cat(nfold_label, dim=0)

        Acc, TP, FP , TN= evaluation(nfold_logit, nfold_label)

    return Acc,TP,FP,TN,nfold_logit,nfold_label,



@torch.no_grad()
def soft_k_nearest_neighbors(features,logits):
    num_neighbors = 3
    pred_probs=[]
    probs = F.softmax(logits, dim=1)
    for feats in features:
        #two different dist
        # distance = torch.cdist(feats, feature)      #Euclidean Distance
        distance = 1 - torch.matmul(F.normalize(feats, dim=1), F.normalize(features, dim=1).T)

        dists,index= distance.sort()
        index = index[:, : num_neighbors]
        prob = probs[index, :].mean(1)
        pred_probs.append(prob)
    pre_probs = torch.cat(pred_probs)
    _, pred_labels = pre_probs.max(dim=1)

    return pred_labels, pre_probs

#------------new vote method2  distance weight
@torch.no_grad()
def soft_k_nearest_neighbors_weight(features,logits):
    num_neighbors = 3
    pred_probs=[]
    probs = F.softmax(logits, dim=1)
    for feats in features:
        #two different dist
        # distance = torch.cdist(feats, feature)
        distance = 1 - torch.matmul(F.normalize(feats, dim=1), F.normalize(features, dim=1).T)
        dists,index= distance.sort()
        index = index[:, : num_neighbors]
        neibor_dist=dists[:,: num_neighbors]
        weig=torch.div(neibor_dist,neibor_dist.sum(1))
        weight,_=torch.sort(weig,dim=1,descending=True)
        weight_prob=probs[index, :].squeeze()
        prob=torch.mm(weight,weight_prob)
        pred_probs.append(prob)
    pre_probs = torch.cat(pred_probs)
    _, pred_labels = pre_probs.max(dim=1)

    return pre_probs


#------------new vote method5  cat src probs and feature weight
@torch.no_grad()
def soft_k_nearest_neighbors_weigbank(logits,features,src_output, src_features):
    logits_bank=torch.cat((logits,src_output),dim=0)
    feature_bank=torch.cat((features,src_features),dim=0)
    num_neighbors = 3
    pred_prob=[]
    probs_bank= F.softmax(logits_bank, dim=1)
    for feats in features:
        feats = feats.view(1, -1)  # 将特征转换为N*(C*W*H)，即两维
        feature= torch.flatten(feature_bank,1)
        #two different dist
        # distance = torch.cdist(feats, feature)
        distance = 1 - torch.matmul(F.normalize(feats, dim=1), F.normalize(feature, dim=1).T)
        dists,index= distance.sort()
        index = index[:, : num_neighbors]
        neibor_dist = dists[:, : num_neighbors]
        weig = torch.div(neibor_dist, neibor_dist.sum(1))
        weight, _ = torch.sort(weig, dim=1, descending=True)
        weight_prob =probs_bank[index, :].squeeze()
        prob=torch.mm(weight, weight_prob)
        pred_prob.append(prob)
    pred_probs = torch.cat(pred_prob)
    _, pred_labels = logits.max(dim=1)

    return pred_probs

#------------new vote method1  cat src probs and feature
@torch.no_grad()
def soft_k_nearest_neighbors_bank(logits,features,src_output, src_features):
    logits_bank=torch.cat((logits,src_output),dim=0)
    feature_bank=torch.cat((features,src_features),dim=0)

    num_neighbors = 3
    pred_prob=[]
    probs_bank= F.softmax(logits_bank, dim=1)
    for feats in features:
        feats = feats.view(1, -1)  # 将特征转换为N*(C*W*H)，即两维
        feature= feature_bank.view(feature_bank.shape[0], -1)
        # feature = features.view( features.shape[0], -1)
        #two different dist
        distance = torch.cdist(feats, feature,p=1)
        # distance = 1 - torch.matmul(F.normalize(feats, dim=1), F.normalize(feature, dim=1).T)
        _,index= distance.sort()
        index = index[:, : num_neighbors]
        prob = probs_bank[index, :].mean(1)
        pred_prob.append(prob)
    pre_probs = torch.cat(pred_prob)
    _, pred_labels = pre_probs.max(dim=1)

    return pred_labels, pre_probs



#------------new vote method3  average feature
@torch.no_grad()
def soft_k_nearest_neighbors_feat(features):
    num_neighbors = 3
    new_feat=[]
    # probs = F.softmax(logits, dim=1)
    for feats in features:
        feats = feats.view(1, -1)  # 将特征转换为N*(C*W*H)，即两维
        feature = features.view( features.shape[0], -1)
        #two different dist
        # distance = torch.cdist(feats, feature)
        distance = 1 - torch.matmul(F.normalize(feats, dim=1), F.normalize(feature, dim=1).T)

        dists, index = distance.sort()
        index = index[:, : num_neighbors]
        feats = features[index, :,:,:].mean(1)
        new_feat.append(feats)
    new_feats= torch.cat(new_feat)

    return new_feats

#------------new vote method4  from both src_net and  withtrain net
@torch.no_grad()
def soft_k_nearest_neighbors_src(logits,features,src_logits, src_features):   #...bad...
    num_neighbors = 3
    pred_probs, src_pred_probs = [],[]
    w_list=[]
    probs = F.softmax(logits, dim=1)
    src_probs = F.softmax(src_logits, dim=1)

    for feats,src_feats in zip(features,src_features):
        feats = feats.view(1, -1)  # 将特征转换为N*(C*W*H)，即两维  (1,C*W*H)
        feature = features.view(features.shape[0], -1)  # (N,C*W*H)
        src_feats = src_feats.view(1, -1)  # 将特征转换为N*(C*W*H)，即两维  (1,C*W*H)
        src_feature = src_features.view(src_features.shape[0], -1)  # (N,C*W*H)

        # two different dist
        # distance = torch.cdist(feats, feature)      #Euclidean Distance
        distance = 1 - torch.matmul(F.normalize(feats, dim=1), F.normalize(feature, dim=1).T)
        src_distance = 1 - torch.matmul(F.normalize(src_feats, dim=1), F.normalize(src_feature, dim=1).T)

        dists, index = distance.sort()
        index = index[:, : num_neighbors]
        neibor_dist = dists[:, : num_neighbors]
        weight = neibor_dist.sum() / neibor_dist
        weight_prob = probs[index, :].view(probs[index, :].size(1), probs[index, :].size(2))
        re = torch.mm(weight, weight_prob)
        prob = re / (weight.size(1))
        pred_probs.append(prob)

        src_dists, src_index = src_distance.sort()
        src_index = src_index[:, : num_neighbors]
        src_neibor_dist = src_dists[:, : num_neighbors]
        src_weight = src_neibor_dist.sum() / src_neibor_dist
        src_weight_prob = src_probs[src_index, :].view(src_probs[src_index, :].size(1), src_probs[src_index, :].size(2))
        src_re = torch.mm(src_weight, src_weight_prob)
        src_prob = src_re / (src_weight.size(1))
        src_pred_probs.append(src_prob)
    pre_prob= torch.cat(pred_probs)
    src_pre_prob = torch.cat(src_pred_probs)
    sum=(neibor_dist.sum()+src_neibor_dist.sum())
    all_pre_probs=(sum/neibor_dist.sum())*pre_prob+(sum/src_neibor_dist.sum())*src_pre_prob
    _, all_pred_labels = all_pre_probs.max(dim=1)

    return all_pred_labels, all_pre_probs

#------------new vote method4  from both src_net and  withtrain net
@torch.no_grad()
def soft_k_nearest_neighbors_both(logits,features,src_logits, src_features):   #...bad...
    num_neighbors = 3
    pred_probs, src_pred_probs = [],[]
    w_list=[]
    probs = F.softmax(logits, dim=1)
    src_probs = F.softmax(src_logits, dim=1)

    for feats,src_feats in zip(features,src_features):
        feats = feats.view(1, -1)  # 将特征转换为N*(C*W*H)，即两维  (1,C*W*H)
        feature = features.view(features.shape[0], -1)  # (N,C*W*H)
        src_feats = src_feats.view(1, -1)  # 将特征转换为N*(C*W*H)，即两维  (1,C*W*H)
        src_feature = src_features.view(src_features.shape[0], -1)  # (N,C*W*H)

        # two different dist
        # distance = torch.cdist(feats, feature)      #Euclidean Distance
        distance = 1 - torch.matmul(F.normalize(feats, dim=1), F.normalize(feature, dim=1).T)
        src_distance = 1 - torch.matmul(F.normalize(src_feats, dim=1), F.normalize(src_feature, dim=1).T)

        dists, index = distance.sort()
        index = index[:, : num_neighbors]
        neibor_dist = dists[:, : num_neighbors]

        src_dists, src_index = src_distance.sort()
        src_index = src_index[:, : num_neighbors]
        src_neibor_dist = src_dists[:, : num_neighbors]
        if neibor_dist.mean()<src_neibor_dist.mean():
            prob = probs[index, :].mean(1)
            pred_probs.append(prob)
        else:
            src_prob = src_probs[src_index, :].mean(1)
            pred_probs.append(src_prob)


    pre_prob = torch.cat(pred_probs)
    _, pred_labels = pre_prob.max(dim=1)

    return pred_labels, pre_prob

