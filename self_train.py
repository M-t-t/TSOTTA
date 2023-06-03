import torch
from config_model import configure_model,collect_modelparams
import torch.nn.functional as F
from methods.MyLoss import entropy,logit_distill
import math


'''metheds'''

@torch.enable_grad()
def TSOTTA(model,ema_model,data,model_state):
    EPOCH =1
    model=model.cuda()
    ema_model=ema_model.cuda()

    # 优化器
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=1e-4)   #kaggle
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=[0.9, 0.999],  eps=1e-8)  #chbmit lr=1e-4/3e-4/5e-4/1e-3  model.classifier.parameters()
    for step in range(EPOCH):
        model.train()
        ema_model.eval()
        trfeat,troutput = model(data)  # features[35,64,5,2]
        ema_feat, ema_output = ema_model(data)
        # -------method
        prob = F.softmax(troutput, dim=1)
        firstlabel, firstindex = prob.max(1)
        son = prob[:, (1 - firstindex)]
        gm= son.div(firstlabel).sum()

        # -----  different loss
        ent_loss=entropy(troutput).mean(0)
        distillogit_loss= logit_distill(ema_output,troutput)

      # '''PETAL'''
        # loss =ent_loss + distillogit_loss  # loss_distl_ent+ewc_loss+distilfeat_loss#loss_distlogit#+
        # loss.backward()
        #
        # # Fisher Information, line 13 in Algorithm 2 in the main paper
        # fisher_dict = {}
        # for nm, m in model.named_modules():  ## previously used model, but now using self.model
        #     for npp, p in m.named_parameters():
        #         if npp in ['weight', 'bias'] and p.requires_grad:
        #             fisher_dict[f"{nm}.{npp}"] = p.grad.data.clone().pow(2)
        # fisher_list = []
        # for name in fisher_dict:
        #     fisher_list.append(fisher_dict[name].reshape(-1))
        # fisher_flat = torch.cat(fisher_list)
        # threshold = find_quantile(fisher_flat, 0.03)  # 0.03
        #
        # optimizer.step()
        # optimizer.zero_grad()
        # # Teacher update, see line 12 in Algorithm 2 in the main paper
        # ema_model = update_ema_variables(ema_model=ema_model, model=model, alpha_teacher=gm)  #gm1  0.99  0.9
        #
        # # FIM based restore, line 13-15 in Algorithm 2 in the main paper
        # if True:
        #     for nm, m in model.named_modules():
        #         for npp, p in m.named_parameters():
        #             if npp in ['weight', 'bias'] and p.requires_grad:
        #                 mask_fish = (fisher_dict[
        #                                  f"{nm}.{npp}"] < threshold).float().cuda()  # masking makes it restore candidate
        #                 mask = mask_fish
        #                 with torch.no_grad():
        #                     p.data = model_state[f"{nm}.{npp}"] * mask + p * (1. - mask)
        # '''PETAL'''

        # 优化模型
        loss=ent_loss+distillogit_loss           #CHBMIT
        # loss=ent_loss +0.001*distillogit_loss    #kaggle
        optimizer.zero_grad()
        loss = loss.requires_grad_()
        loss.backward()
        optimizer.step()
    # Teacher model update
    ema_model = update_ema_variables(ema_model=ema_model, model=model, alpha_teacher=gm)  #gm1  0.99  0.9

    #----test
    ema_model.eval()
    model.eval()
    with torch.no_grad():
        feat, output = model(data)
        return output

@torch.enable_grad()
def tent(model,data):
    step=1
    model = configure_model(model)
    param,nm=collect_modelparams(model)
    optimizer = torch.optim.SGD(param, lr=1e-5, momentum=0.9, weight_decay=1e-4)
    # optimizer = torch.optim.Adam(param, lr=1e-3, betas=[0.9, 0.999], eps=1e-8)

    for epoch in range(step):
        trfeat,troutput = model(data)  # features[35,64,5,2]
        loss=entropy(troutput).mean(0)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    with torch.no_grad():
        feat, output = model(data)
    return output

    '''metheds'''


def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


def update_model_probs(current_model_probs, new_probs):
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        else:
            with torch.no_grad():
                return new_probs.mean(0)
    else:
        if new_probs.size(0) == 0:
            with torch.no_grad():
                return current_model_probs
        else:
            with torch.no_grad():
                return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)


def find_quantile(arr, perc):
    arr_sorted = torch.sort(arr).values
    frac_idx = perc*(len(arr_sorted)-1)
    frac_part = frac_idx - int(frac_idx)
    low_idx = int(frac_idx)
    high_idx = low_idx + 1
    quant = arr_sorted[low_idx] + (arr_sorted[high_idx]-arr_sorted[low_idx]) * frac_part # linear interpolation

    return quant
