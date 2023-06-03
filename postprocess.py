import torch
from torch import nn
lenth=5#5   3 of 5 for kaggle dataset
throd=3#8

def ictal_post(y_pred):
    lastTenResult = list()
    rate, el,alarm= 0,0,0
    # soft_output = nn.Softmax(dim=1)
    # p = soft_output(y_pred)
    # print('y_pred',p)

    while el < len(y_pred):
            if (y_pred[el][1]>y_pred[el][0]):
                # print('y_pred[el][1]',y_pred[el][1])
                rate = rate + 1
                lastTenResult.append(1)
            else:
                lastTenResult.append(0)
            if (len(lastTenResult) > lenth):
                rate = rate - lastTenResult.pop(0)
            if (rate >=throd):
                alarm = alarm + 1
                lastTenResult = list()
                rate = 0
            el = el + 1
    return alarm

def interictal_post(y_pred):
    SOP = 60   #不应期 30min
    lastTenResult = list()
    rate, el, alarm = 0, 0, 0
    soft_output = nn.Softmax(dim=1)
    p = soft_output(y_pred)
    while el < len(y_pred):
        if (y_pred[el][1]>y_pred[el][0]):
            rate = rate + 1
            lastTenResult.append(1)
        else:
            lastTenResult.append(0)
        if (len(lastTenResult) > lenth):
            rate = rate - lastTenResult.pop(0)
        if (rate >=throd):
            alarm = alarm + 1
            lastTenResult = list()
            rate = 0
            el += SOP
        el = el + 1
    return  alarm

import numpy as np
def fact(x):
    if x==1  or x==0:
        return 1
    return x*fact(x-1)
# print(fact(5))
def Comb(n,m):
        ma=fact(m)*fact(n-m)
        son=fact(n)
        result=son/ma
        return result
def P_value(fpr,tp,nfold):
    fpr=fpr.cpu().numpy()
    p=0
    sop=0.5  #30min=0.5h
    P = 1 - np.exp(-fpr * sop)
    for i in range(tp, nfold + 1):
        Q1 = (1 - P) ** (nfold - i)
        Q2 = (P ** i)
        p = p + Comb(nfold, i) * Q1 * Q2
    return p

#-----paired t-test

from scipy import stats

def paired_t_test():
    # AUC values for each method
    method1_auc = [0.583 ,	0.243 ,	0.591 ,	0.470 ,	0.497 ,	0.529 ,	0.650 ,	0.550 ,	0.627 ,	0.593 ,	0.625 ,	0.550 ,
                   0.430 ,	0.720 ,	0.630 ,	0.651 ,	0.205, 	0.621 ,	0.654 ,	0.549]  # 包含20个受试者的AUC值
    method2_auc = [0.565,	0.539,	0.524,	0.556,	0.456,	0.685,	0.959,	0.573,	0.678,	0.499,	0.629,	0.652,
                   0.398,	0.952,	0.515,	0.659,	0.453,	0.56,	0.732,	0.634]
    method3_auc = [0.673,	0.434,	0.506,	0.542,	0.55,	0.366,	0.59,	0.522,	0.59,	0.438,	0.586,	0.45,
                   0.397,	0.562,	0.534,	0.573,	0.186,	0.679,	0.539,	0.471]
    # method4_auc = [0.79, 0.81, 0.75, ..., 0.85]
    # method5_auc = [0.83, 0.79, 0.81, ..., 0.87]
    # method6_auc = [0.88, 0.86, 0.82, ..., 0.83]
    # method7_auc = [0.80, 0.82, 0.78, ..., 0.79]

    # Combine all AUC values into a numpy array
    auc_values = np.array([method1_auc, method2_auc, method3_auc])#, method4_auc, method5_auc, method6_auc, method7_auc])

    # Perform paired t-test for each pair of methods
    num_methods = auc_values.shape[0]
    p_values = np.zeros((num_methods, num_methods))

    for i in range(num_methods):
        for j in range(i + 1, num_methods):
            t_statistic, p_value = stats.ttest_rel(auc_values[i], auc_values[j])
            p_values[i, j] = p_value
            p_values[j, i] = p_value

    # Print the p-values
    print("P-values:")
    for i in range(num_methods):
        for j in range(i + 1, num_methods):
            print(f"Method {i + 1} vs. Method {j + 1}: p-value = {p_values[i, j]}")
            print(p_values[i, j]== p_values[j, i])






#随机种子
import random
def setup_seed(seed):
    torch.manual_seed(seed)           #为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed_all(seed)   #为当前GPU设置随机种子，如果使用多个GPU，应该使用torch.cuda.manual_seed_all()为所有的GPU设置种子。
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = False


# 写CSV
import csv
import numpy as np
import torch
import os
save_path='/home/mtt/desktop/Online-Donain-Adaption/seizure-prediction-CNN-master/logs/'
def calcualte8(predict_labels, true_labels):
    predict_labels=predict_labels.view(1,-1)
    true_labels=true_labels.view(1,-1)
    head = ['pred_labels', 'true_labels']
    labels = torch.cat((predict_labels, true_labels), dim=0)
    # print('labels',labels)
    List= labels.cpu().numpy().tolist()
    L = np.transpose(List)
    if os.path.exists(os.path.join(save_path, 'predictions.csv')):  # https://blog.csdn.net/m0_46483236/article/details/109583685
        os.remove(os.path.join(save_path, 'predictions.csv'))
    with open(os.path.join(save_path, 'predictions.csv'), 'a', encoding='utf8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(head)
        writer.writerows(L)
    # print('test finish')
# if  __name__ == '__main__':
#     predict_labels = torch.tensor([1, 2, 3, 4])
#     true_labels = torch.tensor([1, 2, 3, 4])
#     print(calcualte8(predict_labels, true_labels))

# import pandas as pd
# def resampling(data,patient):
#     targetFrequency = 256 # re-sample to target frequency
#     numts,i = 4,0
#     X = []
#     y = []
#     window_len = int(targetFrequency * numts)
#     df_sampling = pd.read_csv( 'sampling_CHBMIT.csv' , header=0, index_col=None)
#     trg = int(patient)
#     ictal_ovl_pt = df_sampling[df_sampling.Subject == trg].ictal_ovl.values[0]
#     ictal_ovl_len = int(targetFrequency * ictal_ovl_pt * numts)
#     while (window_len + (i + 1) * ictal_ovl_len <= data.shape[0]):
#         s = data[i * ictal_ovl_len:i * ictal_ovl_len + window_len, :]
#         stft_data = s.reshape(-1, 1, 1024, 22)
#         X.append(stft_data)
#         y.append(0)
#         i=i+1
#     return X,y

#
# def postprocess(y_pred):
#     lastTenResult = list()
#     rate = 0
#     alarm = 0
#     el = 0
#     while el < len(y_pred):
#         if (y_pred[el][1]>y_pred[el][0]):     #alarm是TP+FP
#             rate = rate + 1
#             lastTenResult.append(1)
#         else:
#             lastTenResult.append(0)
#         if (len(lastTenResult) > 10):    #len(lastTenResult)~~label总共预测的次数
#             rate = rate - lastTenResult.pop(0)
#         if (rate >=8):
#             alarm = alarm + 1
#             lastTenResult = list()
#             rate = 0
#             el += 60
#             continue
#         el = el + 1
#     return alarm    # alarm是tp





















