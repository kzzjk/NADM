
import torch
from torch import nn

from xmodaler.config import configurable
from xmodaler.config import kfg
from .build import PREDICTOR_REGISTRY
import torch.nn.functional as F
__all__ = ["MYPredictor2"]

@PREDICTOR_REGISTRY.register()
class MYPredictor2(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        hidden_size: int,
        vocab_size: int,   # include <BOS>/<EOS>
        dropout: float,
        label_smoothing: float

    ):
        super(MYPredictor2, self).__init__()
        self.logits = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self.label_smoothing = label_smoothing
        self.confidence = 1.0 - self.label_smoothing
        self.criterion = nn.KLDivLoss(reduction='none')
        params = torch.ones(4, requires_grad=True)
        self.params = torch.nn.Parameter(params)



    @classmethod
    def from_config(cls, cfg):

        return {
            "hidden_size": cfg.MODEL.DECODER_DIM,
            "vocab_size": cfg.MODEL.VOCAB_SIZE,
            "dropout": cfg.MODEL.PRED_DROPOUT,
            "label_smoothing": cfg.LOSSES.LABELSMOOTHING
        }

    @classmethod
    def add_config(cls, cfg):
        pass


    def forward(self, batched_inputs):
        hidden_states = batched_inputs[kfg.G_HIDDEN_STATES]
        if isinstance(hidden_states, list):
            hidden_states = hidden_states[-1]
        if self.dropout:
            hidden_states = self.dropout(hidden_states)
        logits = self.logits(hidden_states)
        if batched_inputs[kfg.STAGE] == 'train':
            targets = batched_inputs[kfg.G_TARGET_IDS]
            token_weight = batched_inputs[kfg.TOKEN_WEIGHT]
            logP = F.log_softmax(logits.view(-1, logits.shape[-1]), dim=-1)
            targets = targets.view(-1)


            assign_seq = targets  # .type(torch.cuda.LongTensor)
            assign_seq[assign_seq < 0] = 0

            size = logP.size(1)
            true_dist = logP.clone()
            true_dist.fill_(self.label_smoothing / (size - 1))
            true_dist.scatter_(1, assign_seq.data.unsqueeze(1), self.confidence)
            loss = self.criterion(logP, true_dist).sum(1)

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
            num = loss1_num + loss2_num + loss3_num + loss4_num
            weight1 = loss1_num / num
            weight2 = loss2_num / num
            weight3 = loss3_num / num
            weight4 = loss4_num / num
            loss_sum = 0
            for i, loss in enumerate([weight1 * loss1+weight2 * loss2,weight3 * loss3+weight4 * loss4]):
                loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)

            return { kfg.G_LOGITS: logits,'LabelSmoothing(G) loss': loss_sum}


        return { kfg.G_LOGITS: logits}