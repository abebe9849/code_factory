import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from torch.nn import Parameter
import math
from torch.nn.modules.loss import _WeightedLoss


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
class OneHotCrossEntropy(_WeightedLoss):

    def __init__(self, weight=None, reduction='mean'):
        super().__init__(weight=weight, reduction=reduction)
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()
        return loss
class SmoothCrossEntropy(nn.Module):
    
    # From https://www.kaggle.com/shonenkov/train-inference-gpu-baseline
    def __init__(self, smoothing = 0.05,one_hotted=False):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.one_hotted = one_hotted

    def forward(self, x, target):
        if self.training:
            x = x.float()
            if self.one_hotted!=True:
                target = F.one_hot(target.long(), x.size(1))
            target = target.float()
            logprobs = F.log_softmax(x, dim = -1)

            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)
    
            smooth_loss = -logprobs.mean(dim=-1)
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss
            return loss.mean()
        else:
            if self.one_hotted!=True:
                loss = F.cross_entropy(x, target.long())
            else:
                loss = OneHotCrossEntropy(x, target)
            return loss
 


class DenseCrossEntropy(nn.Module):

    def forward(self, x, target):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()
    
    
class ArcFaceLoss(nn.Module):

    def __init__(self, s=30.0, m=0.5):
        super().__init__()
        self.crit = DenseCrossEntropy()
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, logits, labels):
        labels = F.one_hot(labels.long(), logits.size(1)).float().to(labels.device)
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = self.crit(output, labels)
        return loss
    
    
##adacos...https://github.com/4uiiurz1/pytorch-adacos/blob/master/metrics.py
class AdaCos(nn.Module):
    def __init__(self, num_features, num_classes, m=0.50):
        super(AdaCos, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = math.sqrt(2) * math.log(num_classes - 1)
        self.m = m
        self.W = Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label=None):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        if label is None:
            return logits
        # feature re-scale
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        with torch.no_grad():
            B_avg = torch.where(one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits))
            B_avg = torch.sum(B_avg) / input.size(0)
            # print(B_avg)
            theta_med = torch.median(theta[one_hot == 1])
            self.s = torch.log(B_avg) / torch.cos(torch.min(math.pi/4 * torch.ones_like(theta_med), theta_med))
        output = self.s * logits

        return output
    
    

class MyCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean'):
        super().__init__(weight=weight, reduction=reduction)
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss


from torch.nn.parameter import Parameter
from torch.autograd import Function
class FocalLoss_ce(nn.Module):
    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss_ce, self).__init__()
        self.gamma = gamma
        self.eps = eps
    def forward(self, input, target):
        logit = torch.nn.functional.softmax(input, dim=1)
        logit = logit.clamp(self.eps, 1. - self.eps)
        logit_ls = torch.log(logit)
        loss = torch.nn.functional.nll_loss(logit_ls, target.long(), reduction="none")
        view = target.size() + (1,)
        index = target.view(*view)
        loss = loss * (1 - logit.gather(1, index).squeeze(1)) ** self.gamma # focal loss
        return loss.sum()

class FocalLoss_BCE_withlogit(nn.Module):
    #
    def __init__(self, alpha=1, gamma=2, logits=True, reduction=False):
        super(FocalLoss_BCE_withlogit, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction
    #    
    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs.float(), targets.float(), reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs.float(), targets.float(), reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        if self.reduction:
            return torch.mean(F_loss)
        else:
            return torch.sum(F_loss)

class FocalLoss_CE(nn.Module):
    #
    def __init__(self, alpha=1, gamma=2, reduction=False):
        super(FocalLoss_CE, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    #    
    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets.long(),reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * CE_loss
        if self.reduction:
            return torch.mean(F_loss)
        else:
            return torch.sum(F_loss)

class FocalCosineLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, xent=.1):
        super(FocalCosineLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

        self.xent = xent

        self.y = torch.Tensor([1]).cuda()

    def forward(self, input, target, reduction="mean"):
        cosine_loss = F.cosine_embedding_loss(input, F.one_hot(target.long(), num_classes=input.size(-1)), self.y, reduction=reduction)

        cent_loss = F.cross_entropy(F.normalize(input), target.long(), reduce=False)
        pt = torch.exp(-cent_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * cent_loss

        if reduction == "mean":
            focal_loss = torch.mean(focal_loss)

        return cosine_loss + self.xent * focal_loss

    



