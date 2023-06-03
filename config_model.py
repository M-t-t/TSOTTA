from torch import nn
#============================== updata whole model
def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            m.requires_grad_(True)
            m.track_running_stats=False
            m.running_mean=None
            m.running_var = None
        # else:
        #     m.requires_grad_(True)
    return model

def collect_modelparams(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names

#==============================just updata classifier
def collect_classifier_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for cp in model.classifier.parameters():
        params.append(cp)
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params

#==============================just updata feature extractor=freeze classifier
def collect_FeatExtrator_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for b1 in model.CNN_block1.parameters():
        params.append(b1)
    for b2 in model.CNN_block2.parameters():
        params.append(b2)
    return params

#====================by EMA updata

#----------EMA method
def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):

        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

def EMA_model_params(ema_model,model,factor):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = factor * ema_param[:].data[:] + (1 - factor) * param[:].data[:]
    return ema_model

def updata_emamodel(model,ema_model,alpha):
    for ema_param,param in zip(ema_model.parameters(),model.parameters()):
        ema_param.data.mul_(alpha).add_(1-alpha,param.data)



def EMA_classifier_params(model,src_model,factor):

    for param, src_param in zip(model.classifier.parameters(), src_model.classifier.parameters()):
        d = param[:].data[:].data[:]
        src_d=src_param[:].data[:]
        param.data[:] = factor * param[:].data[:] + (1 - factor) * src_param[:].data[:]
    return model

def EMA_feat_params(model,src_model,factor):

    for param, src_param in zip(model.parameters(), src_model.parameters()):
        d = param[:].data[:].data[:]
        src_d=src_param[:].data[:]
        param.data[:] = factor * param[:].data[:] + (1 - factor) * src_param[:].data[:]
    return model