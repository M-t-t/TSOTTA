# import json
# import os
# import os.path
# import torch.optim
# from utils.downsampling import down_sample
# from sklearn.metrics import roc_auc_score
# from data_load import train_build_dataload, build_dataload, val_build_dataload
# from utils.load_signals_preictal_corrected import PrepData
# from utils.prep_data import  prepare_targetdata
# from Source_only import source_only
# from addernet import AdderNet
# from models.model import CNN
# from models.ghost_net import ghost_net
# import numpy as np
# import torch
# import torch.nn.functional as F
# import copy
# from sklearn.utils import shuffle
# import time
# from evaluation import evaluation
#
# def makedirs(dir):  # 创建路径
#     try:
#         os.makedirs(dir)
#     except:
#         pass
#
# def main(dataset='Kaggle2014Pred', build_type='cv'):
#     with open('SETTINGS_%s.json' % dataset) as f:
#         settings = json.load(f)
#     makedirs(str(settings['cachedir']))
#     makedirs(str(settings['resultdir']))
#
#     if settings['dataset'] == 'Kaggle2014Pred':
#         source_pt = ['Dog_1', 'Dog_2']  # ,'Dog_4']#, 'Dog_3', 'Dog_4']#, 'Dog_5', 'Patient_1', 'Patient_2' ]
#         target_pt = 'Dog_1'
#         source_pt.remove(target_pt)
#     elif settings['dataset'] == 'FB':
#         targets = [
#             '1',
#             '3',
#             # '4',
#             # '5',
#             '6',
#             '13',
#             '14',
#             '15',
#             '16',
#             '17',
#             '18',
#             '19',
#             '20',
#             '21'
#         ]
#     else:
#         source_pts = ['1', '2', '3', '5', '6', '7', '8', '9', '10', '11', '13', '14', '16', '17', '18', '19', '20', '21', '22', '23']
#         target_pt = '21'
#         source_pts.remove(target_pt)
#     start_time = time.time()
#     val_ratio = 0.1
#
#     # '''prepare source data'''
#     # total_ictal_X,total_ictal_y, total_interictal_X, total_interictal_y=[],[],[],[]
#     # for source_pt in source_pts:
#     #     #-----get source data
#     #     ictal_X, ictal_y = PrepData(source_pt, type='ictal', settings=settings).apply()
#     #     nfold=len(ictal_y)
#     #     print('patient{} has {} seizures'.format(source_pt,nfold))
#     #     interictal_X, interictal_y =PrepData(source_pt, type='interictal', settings=settings).apply()
#     #    #-----list->ndarray
#     #     ictal_X = np.concatenate(ictal_X, axis=0)
#     #     ictal_y = np.concatenate(ictal_y, axis=0)
#     #     # if settings['dataset']=='CHBMIT':
#     #     interictal_X = np.concatenate(interictal_X, axis=0)
#     #     interictal_y = np.concatenate(interictal_y, axis=0)
#     #     ictal_X, ictal_y,interictal_X, interictal_y=down_sample(ictal_X, ictal_y,interictal_X, interictal_y)
#     #
#     #     #list[ndarray1,ndarray2,...]
#     #     total_ictal_X.append(ictal_X)
#     #     total_ictal_y.append(ictal_y)
#     #     total_interictal_X.append(interictal_X)
#     #     total_interictal_y.append(interictal_y)
#     # end_time1 = time.time()
#     # print('prepare source data spend time :',end_time1-start_time)
#     # # list->ndarray
#     # total_ictal_X=np.concatenate(total_ictal_X, axis=0)
#     # total_ictal_y=np.concatenate(total_ictal_y, axis=0)
#     # total_interictal_X=np.concatenate( total_interictal_X, axis=0)
#     # total_interictal_y=np.concatenate(total_interictal_y, axis=0)
#     # total_ictal_X = shuffle(total_ictal_X,random_state=0)
#     # total_interictal_X = shuffle(total_interictal_X,random_state=0)
#     #
#     # # val data
#     # X_val = np.concatenate((total_ictal_X[int(total_ictal_X .shape[0] * (1 - val_ratio)):],total_interictal_X[int(total_interictal_X .shape[0] * (1 - val_ratio)):]), axis=0)
#     # y_val = np.concatenate((total_ictal_y[int(total_ictal_X.shape[0] * (1 - val_ratio)):],total_interictal_y[int(total_interictal_X.shape[0] * (1 - val_ratio)):]), axis=0)
#     #
#     # #-----concentrate train(source) dataset
#     # X_train = np.concatenate((total_ictal_X [:int(total_ictal_X.shape[0]*(1-val_ratio))],total_interictal_X[:int(total_interictal_X.shape[0]*(1-val_ratio))]),axis=0)
#     # y_train = np.concatenate((total_ictal_y[:int(total_ictal_X.shape[0]*(1-val_ratio))],total_interictal_y[:int(total_interictal_X.shape[0]*(1-val_ratio))]),axis=0)
#     #
#     # y_train[y_train == 2] = 1    #kaggle data(1,1,16,59,100)   CHBMIT data(1,1,21,59,114)
#     # y_val[y_val == 2] = 1
#     #
#     # # if settings['dataset']=='CHBMIT':
#     # source_dataloader = train_build_dataload(X_train, y_train)   #train_build_dataload(ndarray1,ndarray2)
#     # val_dataloader =val_build_dataload(X_val, y_val)  # train_build_dataload(ndarray1,ndarray2)
#     # # else:
#     # #     source_dataloader = kaggletrain_build_dataload(X_train, y_train)  # train_build_dataload(ndarray1,ndarray2)
#     # #     val_dataloader = kaggleval_build_dataload(X_val, y_val)  # train_build_dataload(ndarray1,ndarray2)
#
#     # define DDCNet model
#     based_model = CNN(args.num_classes, 18)  # X_train.shape[1]=18->(n,18,59,114)
#     # based_model = AdderNet()
#     # based_model=ghost_net()
#     based_model=based_model.cuda()
#
#     # ------------------------get source only model
#     weight_path = '/home/hgd/mtt/Online-Donain-Adaption/seizure-prediction-CNN-master/weight_files/18ch_weight_files' \
#                   '/updated_online_DA_checkpoint' + target_pt + '.pth'  # just save parmeter of model into dic
#
#     # source_only(based_model, source_dataloader, val_dataloader, weight_path)
#
#     # ----take the first one seizure from target domain as training data
#     ictal_X, ictal_y = PrepData(target_pt, type='ictal', settings=settings).apply()
#     interictal_X, interictal_y = PrepData(target_pt, type='interictal', settings=settings).apply()
#
#     based_model.load_state_dict(torch.load(weight_path))
#     based_model = based_model.cuda()
#
#     sei_acc, sei_AUC, sei_Sen, sei_FPR = [], [], [], []
#     total_logit, total_label = [], []
#     '''prepare test data flow and then test'''
#     loo_folds = prepare_targetdata(ictal_X, ictal_y, interictal_X, interictal_y)  # preparekaggle_targetdata(ictal_X, ictal_y, interictal_X, interictal_y)
#     seizure = 0
#     for X_test, y_test, nfold in loo_folds:
#         X_test, y_test = torch.tensor(X_test).to(torch.float32).cuda(), torch.tensor(y_test).to(torch.float32).cuda()
#
#         # ==========init  model
#         # src_only_net = copy.deepcopy(based_model)
#         # ancher_net = copy.deepcopy(based_model)
#         # ema_net=copy.deepcopy(based_model)
#         # model_state = copy.deepcopy(based_model.state_dict())
#         # '''model use GPU'''
#         # src_only_net = src_only_net.cuda()
#         # ancher_net=ancher_net.cuda()
#
#         # ==========init  model
#         Acc, TP, FP = 0, 0, 0
#         seizure = seizure + 1
#         test_dataloader = build_dataload(X_test, y_test)  # batch=one
#         based_model.eval()
#         with torch.no_grad():
#             for test_data, test_label in test_dataloader:
#                 # ==========init  model
#                 # src_only_net = copy.deepcopy(based_model)
#                 # src_only_net.eval()
#                 # based_model.eval()
#                 # ==========init  model
#                 ''' data use GPU'''
#                 test_data = test_data.cuda()
#                 test_label = test_label.cuda()
#                 feat, logit = based_model(test_data)
#
#                 # -----evaluation
#                 # print('logit',logit)
#                 total_logit.append(logit)
#                 total_label.append(test_label)
#                 acc = (logit.argmax(1) == test_label).sum()
#                 Acc += acc
#                 # ----sen and FPR
#                 if seizure <= nfold:
#                     _, tp, _ = evaluation(logit, test_label)
#                     TP += tp
#                 else:
#                     _, _, fp = evaluation(logit, test_label)
#                     FP += fp
#         total_P = torch.count_nonzero(y_test)
#         total_N = len(y_test) - total_P
#         if total_N == 0:
#             Sen = TP / total_P
#             sei_Sen.append(Sen.item())
#         else:
#             FPR = FP / total_N
#             sei_FPR.append(FPR.item())
#         Accuracy = Acc / len(y_test)
#         sei_acc.append(Accuracy.item())
#         print('seizure{}-- Accuracy is--{}'.format(seizure, Accuracy))
#     total_label = torch.cat(total_label)
#     total_logit = torch.cat(total_logit)
#     preprob = F.softmax(total_logit, dim=1)
#     y_scores = preprob[:, 1]  # 取预测标签为1的概率
#     # print('preprob',preprob)
#     # y_scores = total_logit[:, 1]
#     try:
#         auc = roc_auc_score(total_label.detach().cpu(), y_scores.detach().cpu())  # 评价模型预测概率与期望概率之间的差异，类似交叉熵
#     except ValueError:
#         pass
#     print('online DA--patient{}---average acc is{}'.format(target_pt, sum(sei_acc) / len(sei_acc)))
#     print('average sen is', sum(sei_Sen) / len(sei_Sen))
#     print('average FPR is', sum(sei_FPR) / len(sei_FPR))
#     print('average AUC is', auc)
#
#     end_time2 = time.time()
#     print('total spend time  :', end_time2 - start_time)
#
#
# if __name__ == '__main__':
#     # setup_seed(42)
#     import argparse
#
#     os.environ["CUDA_VISIBLE_DEVICES"] = '1'
#     if torch.cuda.is_available():
#         print('True')
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--mode", default='cv', help="cv or test. cv is for leave-one-out cross-validation")
#     parser.add_argument("--dataset", default='CHBMIT', help="FB, CHBMIT or Kaggle2014Pred")
#     parser.add_argument("--num_classes", default=2, type=int, help="no. classes in dataset (default 2)")
#
#     args = parser.parse_args()
#     assert args.mode in ['cv', 'test']
#     main(dataset=args.dataset, build_type=args.mode)
#
# # based on one seizure test
# # def main(dataset='Kaggle2014Pred', build_type='cv'):
# #     with open('SETTINGS_%s.json' %dataset) as f:
# #         settings = json.load(f)
# #     makedirs(str(settings['cachedir']))
# #     makedirs(str(settings['resultdir']))
# #
# #
# #     if settings['dataset']=='Kaggle2014Pred':
# #         targets = [
# #             # 'Dog_1',
# #             # 'Dog_2',
# #             # 'Dog_3',
# #             # 'Dog_4',
# #             # 'Dog_5',
# #             # 'Patient_1',
# #             # 'Patient_2'
# #         ]
# #     elif settings['dataset']=='FB':
# #         targets = [
# #             '1',
# #             '3',
# #             #'4',
# #             #'5',
# #             '6',
# #             '13',
# #             '14',
# #             '15',
# #             '16',
# #             '17',
# #             '18',
# #             '19',
# #             '20',
# #             '21'
# #         ]
# #     else:
# #         # '1','2','3','4','5','6','7','8','9','10','11','13','14','15','17','18','19','20','21','22','23'
# #         # '1','3','5','8','9','10','11','13','17','18','19','20','21','22','23',
# #         #'1','2','3','5','9','10','13','14','18','19','20','21','23'
# #         source_pts=['1','2','3','5','6','8','9','10','11','17','18','19','20','21','22','23']
# #         target_pt='10'
# #         source_pts.remove(target_pt)
# #     start_time=time.time()
# #     val_ratio=0.1
# #     '''prepare source data'''
# #     # total_ictal_X,total_ictal_y, total_interictal_X, total_interictal_y=[],[],[],[]
# #     # for source_pt in source_pts:
# #     #     #-----get source data
# #     #     ictal_X, ictal_y = PrepData(source_pt, type='ictal', settings=settings).apply()
# #     #     nfold=len(ictal_y)
# #     #     print('patient{} has {} seizures'.format(source_pt,nfold))
# #     #     interictal_X, interictal_y =PrepData(source_pt, type='interictal', settings=settings).apply()
# #     #    #list->ndarray
# #     #     ictal_X = np.concatenate(ictal_X, axis=0)
# #     #     ictal_y = np.concatenate(ictal_y, axis=0)
# #     #     interictal_X = np.concatenate(interictal_X, axis=0)
# #     #     interictal_y = np.concatenate(interictal_y, axis=0)
# #     #     ictal_X, ictal_y,interictal_X, interictal_y=down_sample(ictal_X, ictal_y,interictal_X, interictal_y)
# #     #
# #     #     #list[ndarray1,ndarray2,...]
# #     #     total_ictal_X.append(ictal_X)
# #     #     total_ictal_y.append(ictal_y)
# #     #     total_interictal_X.append(interictal_X)
# #     #     total_interictal_y.append(interictal_y)
# #     # end_time1 = time.time()
# #     # print('prepare source data spend time :',end_time1-start_time)
# #     # # list->ndarray
# #     # total_ictal_X=np.concatenate(total_ictal_X, axis=0)
# #     # total_ictal_y=np.concatenate(total_ictal_y, axis=0)
# #     # total_interictal_X=np.concatenate( total_interictal_X, axis=0)
# #     # total_interictal_y=np.concatenate(total_interictal_y, axis=0)
# #     #
# #     # total_ictal_X = shuffle(total_ictal_X,random_state=0)
# #     # total_interictal_X = shuffle(total_interictal_X,random_state=0)
# #     #
# #     # # val data
# #     # X_val = np.concatenate((total_ictal_X [int(total_ictal_X .shape[0] * (1 - val_ratio)):], total_interictal_X [int(total_interictal_X .shape[0] * (1 - val_ratio)):]), axis=0)
# #     # y_val = np.concatenate((total_ictal_y[int(total_ictal_X.shape[0] * (1 - val_ratio)):], total_interictal_y[int(total_interictal_X.shape[0] * (1 - val_ratio)):]), axis=0)
# #     #
# #     # #-----concentrate train(source) dataset
# #     # X_train = np.concatenate((total_ictal_X [:int(total_ictal_X.shape[0]*(1-val_ratio))],total_interictal_X[:int(total_interictal_X.shape[0]*(1-val_ratio))]),axis=0)
# #     # y_train = np.concatenate((total_ictal_y[:int(total_ictal_X.shape[0]*(1-val_ratio))], total_interictal_y[:int(total_interictal_X.shape[0]*(1-val_ratio))]),axis=0)
# #     #
# #     # y_train[y_train == 2] = 1
# #     # y_val[y_val == 2] = 1
# #     #
# #     # source_dataloader = train_build_dataload(X_train, y_train)   #train_build_dataload(ndarray1,ndarray2)
# #     # val_dataloader =build_dataload(X_val, y_val)  # train_build_dataload(ndarray1,ndarray2)
# #
# #     # define DDCNet model
# #     # Withtrain_model = DDCNet(num_classes=args.num_classes)
# #     Withtrain_model = CNN(args.num_classes,  21)  # X_train.shape[2]
# #     '''model use GPU'''
# #     Withtrain_model = Withtrain_model.cuda()
# #
# #     # ------------------------get source only model
# #     # 'weight_files/updated_online_DA_checkpoint' + target_pt + '.pth'
# #     weight_path = '/home/mtt/desktop/Online-Donain-Adaption/seizure-prediction-CNN-master/weight_files/updated_online_DA_checkpoint' + target_pt + '.pth'  # just save parmeter of model into dic
# #     # source_only(Withtrain_model, source_dataloader, val_dataloader, weight_path)
# #
# #     #----take the first one seizure from target domain as training data
# #     ictal_X, ictal_y = PrepData(target_pt, type='ictal', settings=settings).apply()
# #     interictal_X, interictal_y = PrepData(target_pt, type='interictal', settings=settings).apply()
# #
# #     # -------based_load 模型
# #     Withtrain_model.load_state_dict(torch.load(weight_path))   #updated_online_DA_checkpoint.pth
# #     src_only_net =copy.deepcopy(Withtrain_model)
# #
# #     '''model use GPU'''
# #     src_only_net= src_only_net.cuda()
# #     Withtrain_model=Withtrain_model.cuda()
# #
# #
# #     #----------prepare set up model
# #     # Withtrain_model= configure_model(Withtrain_model)
# #     # Withtrain_model = Withtrain_model.cuda()
# #
# #     ACC,AUC,Sensitivity,FPR= [], [], [], []
# #     '''prepare test data flow and then test'''
# #     loo_folds = spilt_test_data_flow(ictal_X, ictal_y, interictal_X, interictal_y)
# #     seizure = 0
# #     for X_test, y_test, X_trtest,y_trtest,mi,ma in loo_folds:
# #         # print('mi,ma',mi,ma)
# #         seizure=seizure+1
# #         test_dataloader=build_dataload(X_test,y_test)
# #         trtest_dataloader=build_dataload(X_trtest,y_trtest)
# #
# #         # ----- 测试开始
# #         Withtrain_model.eval()
# #         with torch.no_grad():
# #             test_outputs,test_labels=[],[]
# #             for test_data, test_label in test_dataloader:
# #                 ''' data use GPU'''
# #                 test_data = test_data.cuda()
# #                 test_label = test_label.cuda()
# #                 _,test_output = Withtrain_model(test_data)
# #                 test_outputs.append(test_output)
# #                 test_labels.append(test_label)
# #             ypred = torch.cat(test_outputs)
# #             target = torch.cat(test_labels)
# #
# #             acc, auc, sensitivity, fpr = evaluation1(ypred, target,1e-5)
# #             ACC.append(acc.item() / len(target))
# #             AUC.append(auc)
# #             Sensitivity.append(sensitivity.item())
# #             FPR.append(fpr.item())
# #         print('ACC', ACC)
# #         print('AUC', AUC)
# #         print('Sensitivity', Sensitivity)
# #         print('FPR', FPR)
# #
# #         # --------------------
# #         a=0
# #         if a==0:
# #             #-------prepare updata
# #             steps=1
# #             Withtrain_model.train()
# #             #==============EMA updata
# #             # Withtrain_model=EMA_model_params(Withtrain_model,src_only_net, 0.7)
# #             #==============EMA updata
# #
# #             for _ in range(steps):
# #                 trtest_outputs, trtest_labels,feature,src_trtest_outputs,src_feature= [], [],[],[],[]
# #                 for trtest_data, trtest_label in trtest_dataloader:
# #                     ''' data use GPU'''
# #                     trtest_data = trtest_data.cuda()
# #                     src_only_net.eval()
# #                     src_feat,src_logit=src_only_net(trtest_data)
# #                     feat,logit = Withtrain_model(trtest_data)
# #                     all_logits=src_logit+logit
# #
# #                     trtest_outputs.append(all_logits)
# #                     feature.append(feat)
# #                     src_trtest_outputs.append(src_logit)
# #                     src_feature.append(src_feat)
# #
# #                 output = torch.cat(trtest_outputs)
# #                 features=torch.cat(feature)
# #                 src_output = torch.cat(src_trtest_outputs)
# #                 src_features = torch.cat(src_feature)
# #
# #                 # speudo_labels, probs = soft_k_nearest_neighbors(features,output)
# #                 # speudo_labels, probs = soft_k_nearest_neighbors_weight(features,output)
# #                 # speudo_labels, probs = soft_k_nearest_neighbors_bank(output,features,src_output, src_features)
# #                 speudo_labels, probs = soft_k_nearest_neighbors_weigbank(output,features,src_output, src_features)  #logits,features,src_output, src_features
# #                 # speudo_labels=soft_k_nearest_neighbors_feat(features,output,Withtrain_model)
# #                 # speudo_labels, probs = soft_k_nearest_neighbors_src(output,features,src_output,src_features)     #logits,features,src_logits, src_features
# #
# #                 Withtrain_model=selftrain1(Withtrain_model,X_trtest,speudo_labels,probs)
# #
# #     print('patient{}-average AUC is--{},Accuracy is--{},Sensitivity is{}-FPR is--{}'.format(target_pt,
# #         sum(AUC) / len(AUC),sum(ACC) / len(ACC),sum(Sensitivity) / len(Sensitivity),sum(FPR ) / len(FPR)))
# #
# #     end_time2 = time.time()
# #     print('total spend time :', end_time2 - start_time)
#

import os
import json
import os.path

import numpy as np
import torch.optim
from sklearn.metrics import roc_auc_score
from data_load import build_dataload
from utils.load_signals import PrepData
from utils.prep_data import prepare_targetdata,preparekaggle_targetdata
from models.model import CNN,kaggle_CNN
import torch
import copy
from postprocess import interictal_post,ictal_post,P_value
from sklearn.metrics import confusion_matrix
import time
import torch.nn.functional as F
from evaluation import evaluation
from postprocess import setup_seed

def makedirs(dir):  # 创建路径
    try:
        os.makedirs(dir)
    except:
        pass


def main(dataset='Kaggle2014Pred', build_type='cv'):
    with open('SETTINGS_%s.json' % dataset) as f:
        settings = json.load(f)
    makedirs(str(settings['cachedir']))
    makedirs(str(settings['resultdir']))

    if settings['dataset'] == 'Kaggle2014Pred':
        source_pts = ['Dog_1', 'Dog_2','Dog_3', 'Dog_4']#, 'Dog_5', 'Patient_1', 'Patient_2' ]  'Dog_1', 'Dog_2',
        target_pt = 'Dog_4'
        source_pts.remove(target_pt)
    elif settings['dataset'] == 'FB':
        targets = [
            '1',
            '3',
            # '4',
            # '5',
            '6',
            '13',
            '14',
            '15',
            '16',
            '17',
            '18',
            '19',
            '20',
            '21'
        ]
    else:   #'13','16',
        source_pts = [ '1', '2','3', '5', '6', '7', '8', '9', '10', '11','13', '14', '16', '17', '18', '19', '20','21', '22', '23']#,'13','16']
        target_pt = '21'
        source_pts.remove(target_pt)

    # val_ratio = 0.1
    # '''prepare source data'''
    # total_ictal_X,total_ictal_y, total_interictal_X, total_interictal_y=[],[],[],[]
    # for source_pt in source_pts:
    #     #-----get source data
    #     ictal_X, ictal_y = PrepData(source_pt, type='ictal', settings=settings).apply()
    #     nfold=len(ictal_y)
    #     print('patient{} has {} seizures'.format(source_pt,nfold))
    #     interictal_X, interictal_y =PrepData(source_pt, type='interictal', settings=settings).apply()
    #    #-----list->ndarray
    #     ictal_X = np.concatenate(ictal_X, axis=0)
    #     ictal_y = np.concatenate(ictal_y, axis=0)
    #     # if settings['dataset']=='CHBMIT':
    #     interictal_X = np.concatenate(interictal_X, axis=0)
    #     interictal_y = np.concatenate(interictal_y, axis=0)
    #     ictal_X, ictal_y,interictal_X, interictal_y=down_sample(ictal_X, ictal_y,interictal_X, interictal_y)
    #
    #     #list[ndarray1,ndarray2,...]
    #     total_ictal_X.append(ictal_X)
    #     total_ictal_y.append(ictal_y)
    #     total_interictal_X.append(interictal_X)
    #     total_interictal_y.append(interictal_y)
    #     # print('ictal_X.shapeszfdszgsdhfdg',ictal_X.shape)
    # end_time1 = time.time()
    # print('prepare source data spend time :',end_time1-start_time)
    # # list->ndarray
    # total_ictal_X=np.concatenate(total_ictal_X, axis=0)
    # total_ictal_y=np.concatenate(total_ictal_y, axis=0)
    # total_interictal_X=np.concatenate( total_interictal_X, axis=0)
    # total_interictal_y=np.concatenate(total_interictal_y, axis=0)
    # # total_ictal_X = shuffle(total_ictal_X,random_state=0)
    # # total_interictal_X = shuffle(total_interictal_X,random_state=0)
    # # val data
    # X_val = np.concatenate((total_ictal_X[int(total_ictal_X .shape[0] * (1 - val_ratio)):],total_interictal_X[int(total_interictal_X .shape[0] * (1 - val_ratio)):]), axis=0)
    # y_val = np.concatenate((total_ictal_y[int(total_ictal_X.shape[0] * (1 - val_ratio)):],total_interictal_y[int(total_interictal_X.shape[0] * (1 - val_ratio)):]), axis=0)
    #
    # #-----concentrate train(source) dataset
    # X_train = np.concatenate((total_ictal_X [:int(total_ictal_X.shape[0]*(1-val_ratio))],total_interictal_X[:int(total_interictal_X.shape[0]*(1-val_ratio))]),axis=0)
    # y_train = np.concatenate((total_ictal_y[:int(total_ictal_X.shape[0]*(1-val_ratio))],total_interictal_y[:int(total_interictal_X.shape[0]*(1-val_ratio))]),axis=0)
    #
    # y_train[y_train == 2] = 1    #kaggle data(1,1,16,59,100)   CHBMIT data(1,1,21,59,114)
    # y_val[y_val == 2] = 1
    # # X_train, y_train = torch.tensor(X_train).to(torch.float32).to(device), torch.tensor(y_train).to(torch.float32).to( device)
    #
    # # if settings['dataset']=='CHBMIT':
    # source_dataloader = train_build_dataload(X_train, y_train)   #train_build_dataload(ndarray1,ndarray2)
    # val_dataloader =val_build_dataload(X_val, y_val)  # train_build_dataload(ndarray1,ndarray2)
    # # else:
    # # source_dataloader = kaggletrain_build_dataload(X_train, y_train)  # train_build_dataload(ndarray1,ndarray2)
    # # val_dataloader = kaggleval_build_dataload(X_val, y_val)  # train_build_dataload(ndarray1,ndarray2)

    # define model
    model = CNN(2, 18)
    # model = kaggle_CNN(2, 16)
    # weight_path = '/home/mtt/desktop/paper2/modify/TSOTTA_modify/weight_files/kaggle3-1schler' \
    weight_path = '/home/mtt/desktop/paper2/modify/TSOTTA_modify/weight_files/chbmit_weight_files'\
                  '/updated_online_DA_checkpoint' + target_pt + '.pth'  # just save parmeter of model into dic

    model.load_state_dict(torch.load(weight_path))
    src_only_net = model.cuda()
    # source_only(based_model, source_dataloader, val_dataloader, weight_path)
    # trainend_time= time.time()
    # print('train source model total spend time  :', trainend_time - start_time)
    # ----take the first one seizure from target domain as training data
    ictal_X, ictal_y = PrepData(target_pt, type='ictal', settings=settings).apply()
    interictal_X, interictal_y = PrepData(target_pt, type='interictal', settings=settings).apply()

    sei_acc, sei_AUC, sei_Sen, sei_FPR = [], [], [], []
    total_logit, total_label = [], []

    '''prepare test data flow and then test'''
    loo_folds = prepare_targetdata(ictal_X, ictal_y, interictal_X,  interictal_y)  # preparekaggle_targetdata(ictal_X, ictal_y, interictal_X, interictal_y)
    seizure = 0
    event_tp,event_fp = [],[]
    for X_test, y_test, shift, nfold in loo_folds:
        event_total_ictallogit, event_total_inetrictallogit = [], []
        X_test, y_test = torch.tensor(X_test).to(torch.float32).to(device), torch.tensor(y_test).to(torch.float32).to(device)

        Acc, TP, FP = 0, 0, 0
        seizure = seizure + 1
        test_dataloader = build_dataload(X_test, y_test)  # batch=one
        start_time = time.time()
        with torch.no_grad():
            for test_data, test_label in test_dataloader:
                ''' data use GPU'''
                test_data = test_data.to(device)
                test_label = test_label.to(device)

                src_only_net.eval()
                _, logit = src_only_net(test_data)

                total_logit.append(logit)
                total_label.append(test_label)

                '''evaluation segment-based '''
                acc = (logit.argmax(1) == test_label).sum()
                Acc += acc
                # ----sen and FPR
                if seizure <= nfold:
                    event_total_ictallogit.append(logit)
                    _, tp, _ ,_= evaluation(logit, test_label)
                    TP += tp
                else:
                    event_total_inetrictallogit.append(logit)
                    _, _, fp,_ = evaluation(logit, test_label)
                    FP += fp

            '''diferent evaluation'''
            if seizure<=nfold:
                event_total_ictallogit = torch.cat(event_total_ictallogit)
                post_tp = ictal_post(event_total_ictallogit)
                event_tp.append(post_tp)

            else:
                event_total_inetrictallogit = torch.cat(event_total_inetrictallogit)
                post_fp = interictal_post(event_total_inetrictallogit)
                event_fp.append(post_fp)
            '''diferent evaluation'''

        total_P = torch.count_nonzero(y_test)
        total_N = len(y_test) - total_P
        if total_N == 0:
            Sen = TP / total_P
            sei_Sen.append(Sen.item())
        else:
            FPR = FP / total_N
            sei_FPR.append(FPR.item())
        Accuracy = Acc / len(y_test)
        sei_acc.append(Accuracy.item())
        print('seizure{}-- Accuracy is--{}'.format(seizure, Accuracy))

    total_label = torch.cat(total_label)
    total_logit = torch.cat(total_logit)
    preprob = F.softmax(total_logit, dim=1)
    y_scores = preprob[:, 1]  # 取预测标签为1的概率
    try:
        auc = roc_auc_score(total_label.detach().cpu(), y_scores.detach().cpu())  # 评价模型预测概率与期望概率之间的差异，类似交叉熵
    except ValueError:
        pass
    print('online DA--patient{}---average acc is{}'.format(target_pt, sum(sei_acc) / len(sei_acc)))
    print('average AUC is', auc)
    print('average sen is', sum(sei_Sen) / len(sei_Sen))
    print('average FPR is', sum(sei_FPR) / len(sei_FPR))
    print('*********************')
    '''metrics'''
    marix = confusion_matrix(total_label.cpu().numpy(), total_logit.cpu().numpy().argmax(1)).ravel()
    otn, ofp, ofn, otp=marix
    print('marix:tn, fp, fn, tp is ', marix)  # (tn, fp, fn, tp)

    #----segment-based
    right=(otp+otn)/(otn+ofp+ofn+otp)
    specifity=otn/(otn+ofp)
    sensity=otp/(otp+ofn)
    print('*********************')
    print('segment-based right 为:', right)
    print('segment-based specifity 为:', specifity)
    print('segment-based sensity 为:', sensity)

    # ----event-based
    event_TP =len(np.flatnonzero(event_tp))
    # print('event_TP', event_TP)
    inter_samples=len(total_label)-sum(total_label)
    inter_time=inter_samples*30/(60*60)

    event_sensity=event_TP/(nfold)
    event_fpr=sum(event_fp)/(inter_time)

    #----p-value
    p = P_value(event_fpr, event_TP, nfold)

    # ------F-Measure
    # F_Measure = 2 * otp / (nfold + otp + sum(ofp))
    # print('event F_Measure', F_Measure)
    print('*********************')
    print('event-based Sensitivity为:', event_sensity)
    print('event-based FPR为:', sum(event_fp),(inter_time))
    print('event-based p value值为：', p)

    end_time2 = time.time()
    print('total spend time  :', end_time2 - start_time)


if __name__ == '__main__':

    device=(torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
    # x=torch.randn(1).to(device)
    # print(x)
    # num=torch.cuda.device_count()
    # print('num',torch.version.cuda,num)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='cv', help="cv or test. cv is for leave-one-out cross-validation")
    parser.add_argument("--dataset", default='CHBMIT', help="FB, CHBMIT or Kaggle2014Pred")
    parser.add_argument("--num_classes", default=2, type=int, help="no. classes in dataset (default 2)")

    args = parser.parse_args()
    assert args.mode in ['cv', 'test']
    main(dataset=args.dataset, build_type=args.mode)

# based on one seizure test
# def main(dataset='Kaggle2014Pred', build_type='cv'):
#     with open('SETTINGS_%s.json' %dataset) as f:
#         settings = json.load(f)
#     makedirs(str(settings['cachedir']))
#     makedirs(str(settings['resultdir']))
#
#
#     if settings['dataset']=='Kaggle2014Pred':
#         targets = [
#             # 'Dog_1',
#             # 'Dog_2',
#             # 'Dog_3',
#             # 'Dog_4',
#             # 'Dog_5',
#             # 'Patient_1',
#             # 'Patient_2'
#         ]
#     elif settings['dataset']=='FB':
#         targets = [
#             '1',
#             '3',
#             #'4',
#             #'5',
#             '6',
#             '13',
#             '14',
#             '15',
#             '16',
#             '17',
#             '18',
#             '19',
#             '20',
#             '21'
#         ]
#     else:
#         # '1','2','3','4','5','6','7','8','9','10','11','13','14','15','17','18','19','20','21','22','23'
#         # '1','3','5','8','9','10','11','13','17','18','19','20','21','22','23',
#         #'1','2','3','5','9','10','13','14','18','19','20','21','23'
#         source_pts=['1','2','3','5','6','8','9','10','11','17','18','19','20','21','22','23']
#         target_pt='10'
#         source_pts.remove(target_pt)
#     start_time=time.time()
#     val_ratio=0.1
#     '''prepare source data'''
#     # total_ictal_X,total_ictal_y, total_interictal_X, total_interictal_y=[],[],[],[]
#     # for source_pt in source_pts:
#     #     #-----get source data
#     #     ictal_X, ictal_y = PrepData(source_pt, type='ictal', settings=settings).apply()
#     #     nfold=len(ictal_y)
#     #     print('patient{} has {} seizures'.format(source_pt,nfold))
#     #     interictal_X, interictal_y =PrepData(source_pt, type='interictal', settings=settings).apply()
#     #    #list->ndarray
#     #     ictal_X = np.concatenate(ictal_X, axis=0)
#     #     ictal_y = np.concatenate(ictal_y, axis=0)
#     #     interictal_X = np.concatenate(interictal_X, axis=0)
#     #     interictal_y = np.concatenate(interictal_y, axis=0)
#     #     ictal_X, ictal_y,interictal_X, interictal_y=down_sample(ictal_X, ictal_y,interictal_X, interictal_y)
#     #
#     #     #list[ndarray1,ndarray2,...]
#     #     total_ictal_X.append(ictal_X)
#     #     total_ictal_y.append(ictal_y)
#     #     total_interictal_X.append(interictal_X)
#     #     total_interictal_y.append(interictal_y)
#     # end_time1 = time.time()
#     # print('prepare source data spend time :',end_time1-start_time)
#     # # list->ndarray
#     # total_ictal_X=np.concatenate(total_ictal_X, axis=0)
#     # total_ictal_y=np.concatenate(total_ictal_y, axis=0)
#     # total_interictal_X=np.concatenate( total_interictal_X, axis=0)
#     # total_interictal_y=np.concatenate(total_interictal_y, axis=0)
#     #
#     # total_ictal_X = shuffle(total_ictal_X,random_state=0)
#     # total_interictal_X = shuffle(total_interictal_X,random_state=0)
#     #
#     # # val data
#     # X_val = np.concatenate((total_ictal_X [int(total_ictal_X .shape[0] * (1 - val_ratio)):], total_interictal_X [int(total_interictal_X .shape[0] * (1 - val_ratio)):]), axis=0)
#     # y_val = np.concatenate((total_ictal_y[int(total_ictal_X.shape[0] * (1 - val_ratio)):], total_interictal_y[int(total_interictal_X.shape[0] * (1 - val_ratio)):]), axis=0)
#     #
#     # #-----concentrate train(source) dataset
#     # X_train = np.concatenate((total_ictal_X [:int(total_ictal_X.shape[0]*(1-val_ratio))],total_interictal_X[:int(total_interictal_X.shape[0]*(1-val_ratio))]),axis=0)
#     # y_train = np.concatenate((total_ictal_y[:int(total_ictal_X.shape[0]*(1-val_ratio))], total_interictal_y[:int(total_interictal_X.shape[0]*(1-val_ratio))]),axis=0)
#     #
#     # y_train[y_train == 2] = 1
#     # y_val[y_val == 2] = 1
#     #
#     # source_dataloader = train_build_dataload(X_train, y_train)   #train_build_dataload(ndarray1,ndarray2)
#     # val_dataloader =build_dataload(X_val, y_val)  # train_build_dataload(ndarray1,ndarray2)
#
#     # define DDCNet model
#     # Withtrain_model = DDCNet(num_classes=args.num_classes)
#     Withtrain_model = CNN(args.num_classes,  21)  # X_train.shape[2]
#     '''model use GPU'''
#     Withtrain_model = Withtrain_model.cuda()
#
#     # ------------------------get source only model
#     # 'weight_files/updated_online_DA_checkpoint' + target_pt + '.pth'
#     weight_path = '/home/mtt/desktop/Online-Donain-Adaption/seizure-prediction-CNN-master/weight_files/updated_online_DA_checkpoint' + target_pt + '.pth'  # just save parmeter of model into dic
#     # source_only(Withtrain_model, source_dataloader, val_dataloader, weight_path)
#
#     #----take the first one seizure from target domain as training data
#     ictal_X, ictal_y = PrepData(target_pt, type='ictal', settings=settings).apply()
#     interictal_X, interictal_y = PrepData(target_pt, type='interictal', settings=settings).apply()
#
#     # -------based_load 模型
#     Withtrain_model.load_state_dict(torch.load(weight_path))   #updated_online_DA_checkpoint.pth
#     src_only_net =copy.deepcopy(Withtrain_model)
#
#     '''model use GPU'''
#     src_only_net= src_only_net.cuda()
#     Withtrain_model=Withtrain_model.cuda()
#
#
#     #----------prepare set up model
#     # Withtrain_model= configure_model(Withtrain_model)
#     # Withtrain_model = Withtrain_model.cuda()
#
#     ACC,AUC,Sensitivity,FPR= [], [], [], []
#     '''prepare test data flow and then test'''
#     loo_folds = spilt_test_data_flow(ictal_X, ictal_y, interictal_X, interictal_y)
#     seizure = 0
#     for X_test, y_test, X_trtest,y_trtest,mi,ma in loo_folds:
#         # print('mi,ma',mi,ma)
#         seizure=seizure+1
#         test_dataloader=build_dataload(X_test,y_test)
#         trtest_dataloader=build_dataload(X_trtest,y_trtest)
#
#         # ----- 测试开始
#         Withtrain_model.eval()
#         with torch.no_grad():
#             test_outputs,test_labels=[],[]
#             for test_data, test_label in test_dataloader:
#                 ''' data use GPU'''
#                 test_data = test_data.cuda()
#                 test_label = test_label.cuda()
#                 _,test_output = Withtrain_model(test_data)
#                 test_outputs.append(test_output)
#                 test_labels.append(test_label)
#             ypred = torch.cat(test_outputs)
#             target = torch.cat(test_labels)
#
#             acc, auc, sensitivity, fpr = evaluation1(ypred, target,1e-5)
#             ACC.append(acc.item() / len(target))
#             AUC.append(auc)
#             Sensitivity.append(sensitivity.item())
#             FPR.append(fpr.item())
#         print('ACC', ACC)
#         print('AUC', AUC)
#         print('Sensitivity', Sensitivity)
#         print('FPR', FPR)
#
#         # --------------------
#         a=0
#         if a==0:
#             #-------prepare updata
#             steps=1
#             Withtrain_model.train()
#             #==============EMA updata
#             # Withtrain_model=EMA_model_params(Withtrain_model,src_only_net, 0.7)
#             #==============EMA updata
#
#             for _ in range(steps):
#                 trtest_outputs, trtest_labels,feature,src_trtest_outputs,src_feature= [], [],[],[],[]
#                 for trtest_data, trtest_label in trtest_dataloader:
#                     ''' data use GPU'''
#                     trtest_data = trtest_data.cuda()
#                     src_only_net.eval()
#                     src_feat,src_logit=src_only_net(trtest_data)
#                     feat,logit = Withtrain_model(trtest_data)
#                     all_logits=src_logit+logit
#
#                     trtest_outputs.append(all_logits)
#                     feature.append(feat)
#                     src_trtest_outputs.append(src_logit)
#                     src_feature.append(src_feat)
#
#                 output = torch.cat(trtest_outputs)
#                 features=torch.cat(feature)
#                 src_output = torch.cat(src_trtest_outputs)
#                 src_features = torch.cat(src_feature)
#
#                 # speudo_labels, probs = soft_k_nearest_neighbors(features,output)
#                 # speudo_labels, probs = soft_k_nearest_neighbors_weight(features,output)
#                 # speudo_labels, probs = soft_k_nearest_neighbors_bank(output,features,src_output, src_features)
#                 speudo_labels, probs = soft_k_nearest_neighbors_weigbank(output,features,src_output, src_features)  #logits,features,src_output, src_features
#                 # speudo_labels=soft_k_nearest_neighbors_feat(features,output,Withtrain_model)
#                 # speudo_labels, probs = soft_k_nearest_neighbors_src(output,features,src_output,src_features)     #logits,features,src_logits, src_features
#
#                 Withtrain_model=selftrain1(Withtrain_model,X_trtest,speudo_labels,probs)
#
#     print('patient{}-average AUC is--{},Accuracy is--{},Sensitivity is{}-FPR is--{}'.format(target_pt,
#         sum(AUC) / len(AUC),sum(ACC) / len(ACC),sum(Sensitivity) / len(Sensitivity),sum(FPR ) / len(FPR)))
#
#     end_time2 = time.time()
#     print('total spend time :', end_time2 - start_time)

