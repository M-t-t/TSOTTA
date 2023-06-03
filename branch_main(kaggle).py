import json
import os
import os.path
import torch.optim
from sklearn.metrics import roc_auc_score
from utils.load_signals import PrepData
from utils.prep_data import preparekaggle_targetdata
from models.model import kaggle_CNN
import torch
import time
import torch.nn.functional as F
from self_train import tent
from evaluation import accuracy
from postprocess import interictal_post,ictal_post,P_value
from sklearn.metrics import confusion_matrix
from methods import tsotta,tent,sar,eata,t3a,ecotta,sam,shot

def makedirs(dir):  #创建路径
    try:
        os.makedirs(dir)
    except:
        pass

def main(dataset='Kaggle2014Pred'):
    with open('SETTINGS_%s.json' %dataset) as f:
        settings = json.load(f)
    makedirs(str(settings['cachedir']))
    makedirs(str(settings['resultdir']))

    if settings['dataset']=='Kaggle2014Pred':
        source_pt = [ 'Dog_1', 'Dog_2','Dog_3', 'Dog_4']#, 'Dog_5', 'Patient_1', 'Patient_2' ]
        target_pt = 'Dog_1'
        source_pt.remove(target_pt)
    elif settings['dataset']=='FB':
        targets = [
            '1',
            '3',
            #'4',
            #'5',
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
    else:
        source_pts=['1','2','3','5','6','7','8','9','10','11','13','14','16','17','18','19','20','21','22','23']
        target_pt='1'
        source_pts.remove(target_pt)

    # val_ratio=0.1
    # '''prepare source data'''
    # total_ictal_X,total_ictal_y, total_interictal_X, total_interictal_y=[],[],[],[]
    # for source_pt in source_pts:
    #     #-----get source data
    #     ictal_X, ictal_y = PrepData(source_pt, type='ictal', settings=settings).apply()
    #     nfold=len(ictal_y)
    #     print('patient {} has {} seizures'.format(source_pt,nfold))
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
    # end_time1 = time.time()
    # # print('prepare source data spend time :',end_time1-start_time)
    # # list->ndarray
    # total_ictal_X=np.concatenate(total_ictal_X, axis=0)
    # total_ictal_y=np.concatenate(total_ictal_y, axis=0)
    # total_interictal_X=np.concatenate( total_interictal_X, axis=0)
    # total_interictal_y=np.concatenate(total_interictal_y, axis=0)
    # # total_ictal_X = shuffle(total_ictal_X,random_state=0)
    # # total_interictal_X = shuffle(total_interictal_X,random_state=0)
    #
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
    #
    # # if settings['dataset']=='CHBMIT':
    # source_dataloader = train_build_dataload(X_train, y_train)   #train_build_dataload(ndarray1,ndarray2)
    # val_dataloader =val_build_dataload(X_val, y_val)  # train_build_dataload(ndarray1,ndarray2)
    # # else:
    # #     source_dataloader = kaggletrain_build_dataload(X_train, y_train)  # train_build_dataload(ndarray1,ndarray2)
    # #     val_dataloader = kaggleval_build_dataload(X_val, y_val)  # train_build_dataload(ndarray1,ndarray2)
    # define model
    base_model = kaggle_CNN(2, 16)
    # base_model = CNN(args.num_classes,  18)  # X_train.shape[1]=18->(n,18,59,46)
    base_model = base_model.cuda()
    # ------------------------get source only model  chbmit_weight_files   kaggle3-1schler
    weight_path = '/home/mtt/desktop/paper2/modify/TSOTTA_modify/weight_files/kaggle3-1schler' \
                  '/updated_online_DA_checkpoint' + target_pt + '.pth'  # just save parmeter of model into dic

    # source_only(based_model, source_dataloader, val_dataloader, weight_path)

    #----take the first one seizure from target domain as training data
    ictal_X, ictal_y = PrepData(target_pt, type='ictal', settings=settings).apply()
    interictal_X, interictal_y = PrepData(target_pt, type='interictal', settings=settings).apply()

    base_model.load_state_dict(torch.load(weight_path))
    base_model=base_model.cuda()
    sei_acc,sei_AUC,sei_Sen,sei_FPR,sei_Spec= [], [], [], [], []
    total_logit,total_label=[],[]
    # total_ictalogit,total_interictalogit=[],[]

    ADAPTATION = "t3a"
    if ADAPTATION == "src":
        adapt_model = setup_source(base_model)
    elif ADAPTATION == "tent":
        adapt_model = setup_tent(base_model)
    elif ADAPTATION == "tsotta":
        adapt_model = setup_tsotta(base_model)
    elif ADAPTATION == 'dlta':
        adapt_model = setup_dlta(base_model)
    elif ADAPTATION == 'sar':
        adapt_model = setup_sar(base_model)
    elif ADAPTATION == 't3a':
        adapt_model = setup_t3a(base_model)
    elif ADAPTATION == 'shot':
        adapt_model = setup_shot(base_model)
    elif ADAPTATION == 'eata':
        model=setup_eata(base_model)

    start_time=time.time()
    '''prepare target test data flow and then test'''
    loo_folds = preparekaggle_targetdata(ictal_X, ictal_y, interictal_X, interictal_y) #preparekaggle_targetdata(ictal_X, ictal_y, interictal_X, interictal_y)

    for X_test, y_test, shift,nfold in loo_folds:
        if args.ex_type=='non continuous' and ADAPTATION !='src' and ADAPTATION !='eata':
            adapt_model.reset()

        elif args.ex_type=='continuous':
            print('continuous')
            # debug_params = list(base_model.classifier.parameters())
        # if shift==1:
            # debug_params2 = list(base_model.classifier.parameters())

        #-----test
        BATCH_SIZE=1
        X_test, y_test = torch.tensor(X_test).to(torch.float32).cuda(), torch.tensor(y_test).to(torch.float32).cuda()
        if  ADAPTATION=='eata':
            adapt_model=eata.prepare_fisher(X_test, y_test, model, BATCH = BATCH_SIZE)
            adapt_model.reset()
        acc, TP, FP, TN, nfold_logit, nfold_labels = accuracy(adapt_model, X_test, y_test, BATCH_SIZE, device=X_test.device,mode=ADAPTATION)

        total_label.append(nfold_labels)
        total_logit.append(nfold_logit)
        # -----evaluation
        segment_P = sum(y_test)
        segment_N = len(y_test) - segment_P

        if shift < nfold:
            Sen = TP / segment_P
            sei_Sen.append(Sen.item())

        else:
            FPR = FP / segment_N
            Spec=TN/segment_N
            sei_FPR.append(FPR.item())
            sei_Spec.append(Spec.item())

        sei_acc.append((acc / len(nfold_labels)).item())
        print('seizure{}-- Accuracy is--{}'.format(shift, (acc / len(nfold_labels)).item()))

    total_ictalogit=total_logit[:nfold]
    total_interictalogit =total_logit[nfold:]

    total_label = torch.cat(total_label)
    total_logit = torch.cat(total_logit)

    preprob = F.softmax(total_logit, dim=1)
    y_scores = preprob[:, 1]  # 取预测标签为1的概率
    try:
        auc = roc_auc_score(total_label.cpu(), y_scores.cpu())  # 评价模型预测概率与期望概率之间的差异，类似交叉熵
    except ValueError:
        pass
    '''metrics'''
    print('online DA mode: {}--patient{}---average acc is{}'.format(ADAPTATION, target_pt, sum(sei_acc) / len(sei_acc)))
    print('average AUC is', auc)
    # print('average sen is', sum(sei_Sen) / len(sei_Sen))
    # print('average specity is', sum(sei_Spec) / len(sei_Spec))
    # print('average FPR is', sum(sei_FPR) / len(sei_FPR))
    print('*********************')

    '''segment-based'''
    marix = confusion_matrix(total_label.cpu().numpy(), total_logit.cpu().numpy().argmax(1)).ravel()
    otn, ofp, ofn, otp = marix
    # print('confusion marix:tn, fp, fn, tp is ', marix)  # (tn, fp, fn, tp)
    # right = (otp + otn) / (otn + ofp + ofn + otp)
    precision = otp / (otp + ofp)
    sensity = otp / (otp + ofn)
    # print('segment-based acc 为:', right)
    # print('segment-based sensity 为:', sensity)
    # print('segment-based specifity 为:', specifity)
    # print('segment-based fpr debug 为:', ofp / (otn + ofp))
    # # ------F-Measure
    F1_Measure = 2 * sensity * precision / (sensity+precision)
    print('event F_Measure', F1_Measure)
    # print('*********************')

    '''event-based'''
    event_TP,event_FP=0,0
    for ictal in range(len(total_ictalogit)):
        tpalarm=ictal_post(total_ictalogit[ictal])
        if tpalarm!=0:
            event_TP=event_TP+1

    # for interictal in range(len(total_interictalogit)):
    total_interictalogit=torch.cat(total_interictalogit)
    fpalarm=interictal_post(total_interictalogit)
    event_FP = event_FP + fpalarm

    inter_samples = len(total_label) - sum(total_label)
    inter_timeh = inter_samples * 30 / (60 * 60)

    event_sensity = event_TP / (nfold)
    event_fpr = event_FP / (inter_timeh)
    # ----p-value
    p = P_value(event_fpr, event_TP, nfold)
    print('event-based Sensitivity为:', event_sensity)
    print('event-based FPR为:', event_fpr)
    print('event-based p value值为：', p)
    end_time2 = time.time()
    print('total spend time epoch :', end_time2 - start_time)

#-------------------set model

def setup_source(model):
    model.eval()
    return model

def setup_tent(model):
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = torch.optim.SGD(params, 0.005, momentum=0.9)
    # optimizer = torch.optim.Adam(params, lr=1e-3, betas=[0.9, 0.999], eps=1e-8)
    tent_model = tent.Tent(model, optimizer, steps=1, episodic=False)
    return tent_model


def setup_tsotta(model):
    # model_net=deepcopy(model)
    # model = tsotta.configure_model(model)
    # params, param_names = tsotta.collect_params(model)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=1e-4)  # kaggle
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=[0.9, 0.999],  eps=1e-8)  #chbmit
    tsotta_model = tsotta.TSOTTA(model, steps=1, episodic=False)
    return tsotta_model

def setup_dlta(model):
    model = ecotta.configure_model(model)
    params, param_names = ecotta.collect_params(model)
    optimizer = torch.optim.SGD(params, 0.00025, momentum=0.9)
    dlta_model = ecotta.DLTA(model, optimizer, steps=1, episodic=False,
                             mt_alpha=0.999, rst_m=0.01, ap=0.92)
    return dlta_model

def setup_sar(model):

    model = sar.configure_model(model)
    params, param_names = sar.collect_params(model)
    base_optimizer = torch.optim.SGD
    optimizer = sam.SAM(params, base_optimizer, lr=0.001, momentum=0.9)   #0.00025
    sar_model = sar.SAR(model, optimizer)
    return sar_model

def setup_t3a(model):
    model = t3a.configure_model(model)

    t3a_model = t3a.T3A(model, steps=1, episodic=False)
    return t3a_model

def setup_shot(model):
    model = shot.configure_model(model)
    optimizer = torch.optim.SGD(model.parameters(), 1e-2, momentum=0.9,weight_decay=1e-3)
    shot_model = shot.SHOT(model, optimizer)
    return shot_model

def setup_eata(model):
    model = eata.configure_model(model)
    # params, param_names = eata.collect_params(model)
    # optimizer = torch.optim.SGD(params, 0.00025, momentum=0.9)
    # eata_model = eata.EATA(model, optimizer, steps=1, episodic=False, mt_alpha=0.999,
    #                                 rst_m=0.01, ap=0.92, perc=0.03)
    return model


if __name__ == '__main__':
    # setup_seed(2022)
    import argparse
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    if torch.cuda.is_available():
        print('True')
    parser = argparse.ArgumentParser()
    # parser.add_argument("--mode", default='cv',help="cv or test. cv is for leave-one-out cross-validation")
    parser.add_argument("--dataset",default='Kaggle2014Pred', help="FB, CHBMIT or Kaggle2014Pred")
    parser.add_argument("--num_classes", default=2, type=int, help="no. classes in dataset (default 2)")
    parser.add_argument("--ex_type", default='non continuous', type=str, help="continuous or reset segment")
    args = parser.parse_args()
    # assert args.mode in ['cv','test']
    main(dataset=args.dataset)





#based on one seizure test
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

