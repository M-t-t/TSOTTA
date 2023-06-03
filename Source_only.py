import torch
from labelsmooth import CrossEntropyLabelSmooth
from utils.earlystopping import EarlyStopping
from torch import nn
import torch.nn.functional as F



#-----------------TGCNN code
def source_only(source_net,source_loader,val_dataloader,weight_path):
    source_net=source_net.cuda()
    CE_loss = nn.CrossEntropyLoss()#CrossEntropyLabelSmooth(num_classes=2)#
    CE_loss=CE_loss.cuda()
    EPOCH = 50
    patience = 10  # 当验证集连续10次训练周期中都没降，就停止
    early_stopping = EarlyStopping(patience, verbose=True,path=weight_path)
    # 优化器
    # optimizer = torch.optim.Adam(source_net.parameters(), lr=0.01, betas=[0.9, 0.999], eps=1e-8,weight_decay=1e-5)  # lr=1e-4/3e-4/5e-4/4e-3
    # optimizer = torch.optim.AdamW(source_net.parameters(), lr=3e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)   #6e-4
    optimizer = torch.optim.SGD(source_net.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)  # 1e-4 weight_decay=5e-4
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50)

    for epoch in range(EPOCH):
        source_net.train()
        for data, label in source_loader:  # torch.Size([batch_size=100, in_channel_1,H= 1024, W=22]) torch.Size([100])
            '''data use GPU'''
            source_data = data.cuda()
            source_label = label.cuda()
            _,output= source_net(source_data)  # 训练输入数据
            # if torch.isnan(output):
            #     break
            # else:
            loss= CE_loss(output, source_label)

            # 优化模型
            optimizer.zero_grad()
            loss = loss.requires_grad_()
            loss.backward()
            optimizer.step()
        # scheduler.step()
        # if (epoch // 5) == 0:
        #     print('train dataset output', output)
        # ------验证开始
        source_net.eval()
        with torch.no_grad():
            n, loss = 0, 0
            for data, label in val_dataloader:
                ''' data use GPU'''
                val_data = data.cuda()
                label = label.cuda()
                _,output= source_net(val_data)
                if n == 0:
                    outptinit = output
                    labelinit = label
                    n = n + 1
                else:
                    outptinit = torch.cat((outptinit, output), dim=0)
                    labelinit = torch.cat((labelinit, label), dim=0)
            ypred = outptinit
            target = labelinit
            val_loss= CE_loss(ypred, target)
            # val_loss = FL_loss(ypred, target)
            val_acc = (ypred.argmax(1) == target).sum()
            print('验证集损失',val_loss.item())
            print('验证集正确ratio', val_acc/len(target))
            # scheduler.step()
            early_stopping(val_loss.item(), source_net)
            if early_stopping.early_stop:
                # print('Early stopping')
                break

    print('source_only ok')



# #---------AddNet code
# def source_only(source_net, source_loader, val_dataloader, weight_path):
#     device = (torch.device('cuda:0')if torch.cuda.is_available() else torch.device('cpu'))
#     projection_layer = model_projection.ProjectionModel()
#     classifier = model_classifier.Classifier()
#     source_net = source_net.to(device)
#     projection_layer = projection_layer.to(device)
#     classifier=classifier.to(device)
#     loss_fn = HybridLoss(alpha=0.5, temperature=0.06)
#     loss_fn = loss_fn.to(device)
#     CE_loss = nn.CrossEntropyLoss()#CrossEntropyLabelSmooth(num_classes=2)  # nn.CrossEntropyLoss()#
#     CE_loss = CE_loss.to(device)
#     EPOCH = 100
#     patience = 10  # 当验证集连续10次训练周期中都没降，就停止
#     early_stopping = EarlyStopping(patience, verbose=True, path=weight_path)
#     # 优化器
#     # optimizer = torch.optim.Adam(source_net.parameters(), lr=3e-3, betas=[0.9, 0.999], eps=1e-8)  # lr=1e-4/3e-4/5e-4/4e-3
#     optimizer = torch.optim.AdamW(source_net.parameters(), lr=3e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)  # 6e-4
#     # optimizer = torch.optim.SGD(source_net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)  # 1e-4 weight_decay=5e-4
#     scheduler = WarmUpExponentialLR(optimizer, cold_epochs=0., warm_epochs=0, gamma=0.98)
#
#     for epoch in range(EPOCH):
#         source_net.train()
#         projection_layer.train()
#         classifier.train()
#         for data, label in source_loader:  # torch.Size([batch_size=100, in_channel_1,H= 1024, W=22]) torch.Size([100])
#             '''data use GPU'''
#             source_data = data.to(device)
#             label = label.to(device)
#             feat= source_net(source_data)  # 训练输入数据
#             y_proj = projection_layer(feat)
#             y_proj = F.normalize(y_proj, dim=1)
#             y_pred = classifier(feat)
#
#             loss1= loss_fn(y_proj,label)
#             loss2=CE_loss(y_pred,label)
#             train_loss = 0.5*loss1 + 0.5*loss2
#             # 优化模型
#             optimizer.zero_grad()
#             train_loss = train_loss.requires_grad_()
#             train_loss.backward()
#             optimizer.step()
#         if (epoch//10)==0:
#             print('train dataset loss', train_loss)
#
#         # ------验证开始
#         source_net.eval()
#         projection_layer.eval()
#         classifier.eval()
#         with torch.no_grad():
#             n, loss = 0, 0
#             for data, label in val_dataloader:
#                 ''' data use GPU'''
#                 val_data = data.to(device)
#                 label = label.to(device)
#                 feat = source_net(val_data)  # 训练输入数据
#                 y_proj = projection_layer(feat)
#                 y_proj = F.normalize(y_proj, dim=1)
#                 output = classifier(feat)
#
#                 if n == 0:
#                     outptinit = output
#                     labelinit = label
#                     featinit=y_proj
#                     n = n + 1
#                 else:
#                     outptinit = torch.cat((outptinit, output), dim=0)
#                     labelinit = torch.cat((labelinit, label), dim=0)
#                     featinit = torch.cat((featinit, y_proj), dim=0)
#             valout = outptinit
#             target = labelinit
#             feats=featinit
#
#             loss1 = loss_fn(feats, target)
#             loss2 = CE_loss(valout, target)
#             val_loss = 0.5*loss1 + 0.5*loss2
#             val_acc = (valout.argmax(1) == target).sum()
#             print('验证集损失', val_loss.item())
#             print('验证集正确ratio', val_acc / len(target))
#             # scheduler.step()
#             early_stopping(val_loss.item(), source_net)
#             if early_stopping.early_stop:
#                 # print('Early stopping')
#                 break
#         scheduler.step()
#     print('source_only ok')




#----------------STFT+CNN  code
# def source_only(source_net,source_loader,val_dataloader,weight_path):
#     source_net=source_net.cuda()
#     loss_fn = HybridLoss(alpha=0.5, temperature=0.06)
#     loss_fn=loss_fn.cuda()
#     CE_loss = CrossEntropyLabelSmooth(num_classes=2)#nn.CrossEntropyLoss()#
#     CE_loss=CE_loss.cuda()
#     EPOCH = 100
#     patience = 10  # 当验证集连续10次训练周期中都没降，就停止
#     early_stopping = EarlyStopping(patience, verbose=True,path=weight_path)
#     # 优化器
#     # optimizer = torch.optim.Adam(source_net.parameters(), lr=3e-3, betas=[0.9, 0.999], eps=1e-8)  # lr=1e-4/3e-4/5e-4/4e-3
#     optimizer = torch.optim.AdamW(source_net.parameters(), lr=3e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)   #6e-4
#     # optimizer = torch.optim.SGD(source_net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)  # 1e-4 weight_decay=5e-4
#     scheduler = WarmUpExponentialLR(optimizer, cold_epochs=0., warm_epochs=0, gamma=0.98)
#
#     for epoch in range(EPOCH):
#         source_net.train()
#         for data, label in source_loader:  # torch.Size([batch_size=100, in_channel_1,H= 1024, W=22]) torch.Size([100])
#             '''data use GPU'''
#             source_data = data.cuda()
#             source_label = label.cuda()
#             _,output= source_net(source_data)  # 训练输入数据
#
#             # loss= CE_loss(output, source_label)
#             loss=loss_fn(output, source_label)
#             # 优化模型
#             optimizer.zero_grad()
#             loss = loss.requires_grad_()
#             loss.backward()
#             optimizer.step()
#             print('train dataset output', output)
#
#         # ------验证开始
#         source_net.eval()
#         with torch.no_grad():
#             n, loss = 0, 0
#             for data, label in val_dataloader:
#                 ''' data use GPU'''
#                 val_data = data.cuda()
#                 label = label.cuda()
#                 _,output= source_net(val_data)
#                 if n == 0:
#                     outptinit = output
#                     labelinit = label
#                     n = n + 1
#                 else:
#                     outptinit = torch.cat((outptinit, output), dim=0)
#                     labelinit = torch.cat((labelinit, label), dim=0)
#             ypred = outptinit
#             target = labelinit
#             val_loss= CE_loss(ypred, target)
#             # val_loss = FL_loss(ypred, target)
#             val_acc = (ypred.argmax(1) == target).sum()
#             print('验证集损失',val_loss.item())
#             print('验证集正确ratio', val_acc/len(target))
#             # scheduler.step()
#             early_stopping(val_loss.item(), source_net)
#             if early_stopping.early_stop:
#                 # print('Early stopping')
#                 break
#         scheduler.step()
#     print('source_only ok')

    # return source_net



#
# CE_loss = nn.CrossEntropyLoss()
#     CE_loss=CE_loss.cuda()
#     EPOCH = 1
#     patience = 3  # 当验证集连续10次训练周期中都没降，就停止
#     early_stopping = EarlyStopping(patience, verbose=True)
#     # 优化器
#     optimizer = torch.optim.Adam(source_net.parameters(), lr=1e-3, betas=[0.9, 0.999], eps=1e-8)  # lr=1e-4/3e-4/5e-4/1e-3
#     # optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
#
#     source_net.train()
#     for epoch in range(EPOCH):
#         for data, label in source_loader:  # torch.Size([batch_size=100, in_channel_1,H= 1024, W=22]) torch.Size([100])
#             '''data use GPU'''
#             source_data = data.cuda()
#             source_label = label.cuda()
#             output,_ ,_= source_net(source_data,source_data)  # 训练输入数据
#             loss= CE_loss(output, source_label)
#             # 优化模型
#             optimizer.zero_grad()
#             loss = loss.requires_grad_()
#             loss.backward()
#             optimizer.step()
#             # print('source_only model loss',loss.item())
#         # ------验证开始
#         source_net.eval()
#         with torch.no_grad():
#             n, loss = 0, 0
#             for data, label in val_dataloader:
#                 ''' data use GPU'''
#                 val_data = data.cuda()
#                 label = label.cuda()
#                 output,_ ,_= source_net(val_data,val_data)
#                 if n == 0:
#                     outptinit = output
#                     labelinit = label
#                     n = n + 1
#                 else:
#                     outptinit = torch.cat((outptinit, output), dim=0)
#                     labelinit = torch.cat((labelinit, label), dim=0)
#             ypred = outptinit
#             target = labelinit
#             val_loss= CE_loss(ypred, target)
#             # val_acc = (ypred.argmax(1) == target).sum()
#             # print('验证集损失',loss.item())
#             # print('验证集正确次数', val_acc)
#             early_stopping(val_loss.item(), source_net)
#             if early_stopping.early_stop:
#                 # print('Early stopping')
#                 break
#     print('source_only ok')
#
#     # return source_net