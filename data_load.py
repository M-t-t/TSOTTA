import torch
from torch.utils.data import TensorDataset, DataLoader
#--------------train source model
chb_num=64 #128
def train_build_dataload(train_x,y):
    # convert numpy.ndarrary to tensor
    # X = torch.tensor(train_x)
    # X = X.to(torch.float32)
    # # y= torch.cuda.LongTensor(y)
    # y=torch.tensor(y).to(torch.float32).to(device)
    train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
    y = torch.from_numpy(y).type(torch.LongTensor)
    dataset = TensorDataset(train_x, y)

    # 利用dataloader加载数据集
    train_dataloader = DataLoader(dataset, batch_size=chb_num,drop_last=False, shuffle=True)

    return train_dataloader

def val_build_dataload(train_x,y):
    # convert numpy.ndarrary to tensor
    # X = torch.tensor(train_x)
    # X = X.to(torch.float32)
    # y= torch.cuda.LongTensor(y)
    train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
    y = torch.from_numpy(y).type(torch.LongTensor)
    dataset = TensorDataset(train_x, y)

    # 利用dataloader加载数据集
    val_dataloader = DataLoader(dataset, batch_size=chb_num,drop_last=False, shuffle=False)

    return val_dataloader

#-------------kaggle  train
kaggle_num=64#128
def kaggletrain_build_dataload(train_x,y):
    # convert numpy.ndarrary to tensor
    X = torch.tensor(train_x)
    X = X.to(torch.float32)
    y= torch.cuda.LongTensor(y)
    dataset = TensorDataset(X, y)

    # 利用dataloader加载数据集
    train_dataloader = DataLoader(dataset, batch_size=kaggle_num,drop_last=False, shuffle=True)

    return train_dataloader


def kaggleval_build_dataload(train_x,y):
    # convert numpy.ndarrary to tensor
    X = torch.tensor(train_x)
    X = X.to(torch.float32)
    y= torch.cuda.LongTensor(y)
    dataset = TensorDataset(X, y)

    # 利用dataloader加载数据集
    val_dataloader = DataLoader(dataset, batch_size=kaggle_num,drop_last=False, shuffle=False)

    return val_dataloader


#--------------when adaptation ,data come one by one

def build_dataload(X, y,test_num=1):
    # convert numpy.ndarrary to tensor
    # X = torch.as_tensor(train_x).to(torch.float32)
    # y = torch.cuda.LongTensor(y)
    dataset = TensorDataset(X, y)

    # 利用dataloader加载数据集
    dataloader = DataLoader(dataset, batch_size=test_num, drop_last=False, shuffle=False)

    return dataloader



