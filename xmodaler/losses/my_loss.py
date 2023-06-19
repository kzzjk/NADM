import torch
import torch.nn as nn
import torch.nn.functional as F
from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.config import CfgNode as CN
from .build import LOSSES_REGISTRY
import numpy as np
@LOSSES_REGISTRY.register()
class MYLoss(nn.Module):
    @configurable
    def __init__(self, abnormal_weight, normal_weight, num_classes):
        super(MYLoss, self).__init__()
        self.normal_weight = normal_weight
        self.abnormal_weight = abnormal_weight
        self.num_classes = num_classes
        self.gamma_neg = 5
        self.gamma_pos = 0
        self.clip = 0.05
        self.disable_torch_grad_focal_loss = True
        self.eps = 1e-8

    @classmethod
    def from_config(cls, cfg):
        return {
            "abnormal_weight": cfg.MODEL.SEMICNET.ABNORMAL_WEIGHT,
            "normal_weight": cfg.MODEL.SEMICNET.NORMAL_WEIGHT,
            "num_classes": cfg.MODEL.SEMICNET.NUM_CLASSES
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, outputs_dict):
        ret = {}
        abnormal_logits = outputs_dict[kfg.SEMANTICS_ABNORMAL_PRED]
        abnormal_labels = outputs_dict[kfg.ABNORMAL_LABEL].squeeze(1)


        normal_logits = outputs_dict[kfg.SEMANTICS_NORMAL_PRED]
        normal_labels = outputs_dict[kfg.NORMAL_LABEL].squeeze(1)



        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(abnormal_logits)
        #x_sigmoid, _ = torch.max(x_sigmoid, dim=1)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = abnormal_labels * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - abnormal_labels) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * abnormal_labels
            pt1 = xs_neg * (1 - abnormal_labels)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * abnormal_labels + self.gamma_neg * (1 - abnormal_labels)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w
        abnormal_loss = -loss.sum(-1).mean()


        # Calculating Probabilities
        if normal_labels.size(0) != normal_logits.size(0):
            expand_n = int(normal_logits.size(0)/normal_labels.size(0))
            normal_labels = normal_labels.repeat(expand_n,1)
            logsigma = outputs_dict[kfg.SEMANTICS_NORMAL_MA]
            feat_dim = logsigma.shape[-1]
            margin = 300
            entropy = float(feat_dim/ 2 * (np.log(2 * np.pi) + 1)) + torch.sum(logsigma, -1) / 2
            zero = torch.zeros_like(entropy)
            margin_loss = torch.max(margin - entropy,zero)
            margin_loss = torch.mean(margin_loss)
        x_sigmoid = torch.sigmoid(normal_logits)
        #x_sigmoid, _ = torch.max(x_sigmoid, dim=1)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = normal_labels * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - normal_labels) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * normal_labels
            pt1 = xs_neg * (1 - normal_labels)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * normal_labels + self.gamma_neg * (1 - normal_labels)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w
        normal_loss = -loss.sum(-1).mean()
        if kfg.MEM_LOSS in outputs_dict:
            mem_loss = outputs_dict[kfg.MEM_LOSS]
            ret.update({
                'mem_loss': mem_loss
            })
        if kfg.SEMANTICS_NORMAL_MA in outputs_dict:
            ret.update({
                "normal_loss": normal_loss * self.normal_weight,
                "abnormal_loss": abnormal_loss * self.abnormal_weight,
                "margin_loss": 0.01 * margin_loss
            })
        else:
            ret.update({
                "normal_loss": normal_loss * self.normal_weight,
                "abnormal_loss": abnormal_loss * self.abnormal_weight,
            })
        return ret