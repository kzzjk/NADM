
import torch
import torch.nn as nn
import torch.nn.functional as F
from xmodaler.config import configurable
from xmodaler.config import kfg
from .build import LOSSES_REGISTRY

@LOSSES_REGISTRY.register()
class myLabelSmoothing(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        label_smoothing
    ):
        super(myLabelSmoothing, self).__init__()
        self.label_smoothing = label_smoothing
        self.confidence = 1.0 - self.label_smoothing
        self.criterion = nn.KLDivLoss(reduction='none')

    @classmethod
    def from_config(cls, cfg):
        return {
            "label_smoothing": cfg.LOSSES.LABELSMOOTHING
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def Forward(self, logits, targets,token_weight):
        logP = F.log_softmax(logits.view(-1, logits.shape[-1]), dim=-1)
        targets = targets.view(-1)
        mask = targets >= 0

        assign_seq = targets  #.type(torch.cuda.LongTensor)
        assign_seq[assign_seq < 0] = 0

        size = logP.size(1)
        true_dist = logP.clone()
        true_dist.fill_(self.label_smoothing / (size - 1))
        true_dist.scatter_(1, assign_seq.data.unsqueeze(1), self.confidence)
        loss = self.criterion(logP, true_dist).sum(1)
        if token_weight ==None:
            loss = torch.masked_select(loss, mask).mean()
        else:
            token_weight = token_weight.view(-1)
            mask_normal = token_weight == 0
            mask_abnormal = token_weight == 1
            mask_abnormal_semic = token_weight == 2
            mask_normal_semic = token_weight == 3
            loss1 = torch.masked_select(loss, mask_normal)
            loss1_num = loss1.size(0)
            loss1 = loss1.mean()
            loss2 = torch.masked_select(loss, mask_abnormal)
            loss2_num = loss2.size(0)
            loss2 = loss2.mean()
            loss3 = torch.masked_select(loss, mask_abnormal_semic)
            loss3_num = loss3.size(0)
            loss3 = loss3.mean()
            loss4 = torch.masked_select(loss, mask_normal_semic)
            loss4_num = loss4.size(0)
            loss4 = loss4.mean()
            num = loss1_num+loss2_num+loss3_num+loss4_num
            weight1 = loss1_num / num
            weight2 = loss2_num / num
            weight3 = loss3_num / num
            weight4 = loss4_num / num
            loss = 0.691 * weight1 * loss1 + 0.993 * weight4 * loss4 + 1.196 * weight2 * loss2 + 1.502 * weight3 * loss3
        return loss

    def forward(self, outputs_dict):
        ret  = {}
        if kfg.G_LOGITS in outputs_dict:
            logits = outputs_dict[kfg.G_LOGITS]
            targets = outputs_dict[kfg.G_TARGET_IDS]
            if kfg.TOKEN_WEIGHT in outputs_dict:
                token_weight = outputs_dict[kfg.TOKEN_WEIGHT]
            else:
                token_weight = None
            loss = self.Forward(logits, targets,token_weight)
            ret.update({ 'LabelSmoothing(G) loss': loss })

        if kfg.U_LOGITS in outputs_dict:
            logits = outputs_dict[kfg.U_LOGITS]
            targets = outputs_dict[kfg.U_TARGET_IDS]
            loss = self.Forward(logits, targets)
            ret.update({ 'LabelSmoothing(U) loss': loss })
        return ret