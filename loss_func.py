import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F

class FocalLoss_BCE_withlogit(nn.Module):##マルチラベルのとき
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super(FocalLoss_BCE_withlogit, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.reduction = reduction
    def forward(self, input, target):
        pt = torch.sigmoid(input)
        error = torch.abs(pt - target)
        log_error = torch.log(error)
        loss = -1 * (1-error)**self.gamma * log_error
        if self.reduction=="mean": 
            return loss.mean()
        elif self.reduction=="sum":
            return loss.sum()
        else:
            return loss

class Kloss(nn.Module):
    """
    QWKの最適化、
    df:train_df,valid_dfをいれる
    label_name:targetの尺度変数の列名
    Ng:何段階の尺度変数か(0~5ならNg=6)
    """
    def __init__(self,df,label_name,Ng):
        super(Kloss, self).__init__()
        self.y_shift = df[label_name].mean()
        self.Ng = Ng
    def forward(self, input, target):
        input = self.Ng*torch.sigmoid(input.float()).view(-1) - 0.5
        target = target.float()
        loss = 1.0 - (2.0*((input-self.y_shift)*(target-self.y_shift)).sum() - 1e-3)/\
        (((input-self.y_shift)**2).sum() + ((target-self.y_shift)**2).sum() + 1e-3)
        return loss
