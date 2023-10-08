
import torch
from torch import nn

from xmodaler.config import configurable
from xmodaler.config import CfgNode as CN
from xmodaler.config import kfg
from ..layers.bert import BertLayer, BertGenerationLayer
from .build import ENCODER_REGISTRY
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from xmodaler.modeling.layers import myLowRankBilinearLayer
from xmodaler.modeling.layers import get_act_layer
__all__ = ["MYEncoder"]


@ENCODER_REGISTRY.register()
class MYEncoder(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        num_hidden_layers: int,
        bert_layers,
        semic_normal_layers,
        semic_abnormal_layers,
        hidden_size: int,
        num_classes: int,
        slot_size: int,
        memory_size: int,
        semic_begin_layers: int,

        embed_dim: int,
        att_heads: int,
        att_mid_dim: int,
        att_mid_drop: float,
        dropout: float,
        bifeat_emb_dropout: float,
        layer_num: int,
        emb_act_type: str,
        act_type: str,
        elu_alpha: float
    ):
        super(MYEncoder, self).__init__()
        self.num_hidden_layers = num_hidden_layers

        self.semic_normal_layers = semic_normal_layers
        self.semic_abnormal_layers = semic_abnormal_layers
        self.num_classes = num_classes
        self.semic_begin_layers = semic_begin_layers
        self.slot_size = slot_size

        self.layers = bert_layers

        self.gvfeat_embed = nn.Sequential(
            nn.Linear(hidden_size * (num_hidden_layers + 1), hidden_size),
            torch.nn.LayerNorm(hidden_size)
        )
        self.semantics_normal_pred = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes)
        )
        self.semantics_abnormal_pred = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes)
        )
        self.embeddings = nn.Sequential(
            nn.Embedding(num_classes, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1),
        )

        self.slot_embeddings = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1),
        )
        self.slot = nn.Parameter(torch.FloatTensor(1, slot_size, hidden_size))
        self.memory_space = nn.Embedding(memory_size, hidden_size)
        nn.init.xavier_uniform_(self.slot)

        self.layers = nn.ModuleList([])
        self.bifeat_emb = nn.ModuleList([])
        self.layer_norms = nn.ModuleList([])
        for _ in range(layer_num):
            sublayer = myLowRankBilinearLayer(
                embed_dim = embed_dim,
                att_heads = att_heads,
                att_mid_dim = att_mid_dim,
                att_mid_drop = att_mid_drop,
                dropout = dropout,
                act_type= act_type,
                elu_alpha = elu_alpha,
                memory_num = memory_size,
                memory_space = self.memory_space.weight
            )
            self.layers.append(sublayer)

            self.bifeat_emb.append(nn.Sequential(
                nn.Linear(2 * embed_dim, embed_dim),
                get_act_layer(emb_act_type)(),
                nn.Dropout(bifeat_emb_dropout)
            ))

            self.layer_norms.append(torch.nn.LayerNorm(embed_dim))

        self.proj = nn.Linear(embed_dim * (layer_num + 1), embed_dim)
        self.layer_norm = torch.nn.LayerNorm(embed_dim)


    @classmethod
    def from_config(cls, cfg):
        bert_layers = nn.ModuleList(
            [BertLayer(cfg) for _ in range(cfg.MODEL.BERT.NUM_HIDDEN_LAYERS)]
        )
        semic_normal_layers = nn.ModuleList(
            [BertGenerationLayer(cfg) for _ in range(cfg.MODEL.SEMICNET.NUM_SEMCOMPHDER_LAYERS)]
        )
        semic_abnormal_layers = nn.ModuleList(
            [BertLayer(cfg) for _ in range(cfg.MODEL.SEMICNET.NUM_SEMCOMPHDER_LAYERS)]
        )
        return {
            "num_hidden_layers": cfg.MODEL.BERT.NUM_HIDDEN_LAYERS,
            "bert_layers": bert_layers,
            "semic_normal_layers": semic_normal_layers,
            "semic_abnormal_layers": semic_abnormal_layers,

            "hidden_size": cfg.MODEL.BERT.HIDDEN_SIZE,
            "semic_begin_layers": cfg.MODEL.SEMICNET.BEGIN_LAYERS,
            "slot_size": cfg.MODEL.SEMICNET.SLOT_SIZE,
            "num_classes": cfg.MODEL.SEMICNET.NUM_CLASSES,

            "memory_size": cfg.MODEL.SEMICNET.MEMORY_SIZE,

            "embed_dim": cfg.MODEL.BILINEAR.DIM,
            "att_heads": cfg.MODEL.BILINEAR.HEAD,
            "att_mid_dim": cfg.MODEL.BILINEAR.ENCODE.ATT_MID_DIM,
            "att_mid_drop": cfg.MODEL.BILINEAR.ENCODE.ATT_MID_DROPOUT,
            "dropout": cfg.MODEL.BILINEAR.ENCODE.DROPOUT,
            "bifeat_emb_dropout": cfg.MODEL.BILINEAR.ENCODE.BIFEAT_EMB_DROPOUT,
            "layer_num": cfg.MODEL.BILINEAR.ENCODE.LAYERS,
            "emb_act_type": cfg.MODEL.BILINEAR.BIFEAT_EMB_ACT,
            "act_type": cfg.MODEL.BILINEAR.ACT,
            "elu_alpha": cfg.MODEL.BILINEAR.ELU_ALPHA

        }

    @classmethod
    def add_config(cls, cfg):
        cfg.MODEL.SEMICNET = CN()
        cfg.MODEL.SEMICNET.NUM_SEMCOMPHDER_LAYERS = 3
        cfg.MODEL.SEMICNET.BEGIN_LAYERS = 1
        cfg.MODEL.SEMICNET.SLOT_SIZE = 1
        cfg.MODEL.SEMICNET.NUM_CLASSES = 906
        cfg.MODEL.SEMICNET.NORMAL_WEIGHT = 1.0
        cfg.MODEL.SEMICNET.ABNORMAL_WEIGHT = 1.0

        cfg.MODEL.SEMICNET.MEMORY_SIZE = 98

        cfg.MODEL.BILINEAR = CN()
        cfg.MODEL.BILINEAR.DIM = 512
        cfg.MODEL.BILINEAR.HEAD = 8
        cfg.MODEL.BILINEAR.BIFEAT_EMB_ACT = "relu"
        cfg.MODEL.BILINEAR.ACT = "celu"
        cfg.MODEL.BILINEAR.ELU_ALPHA = 1.3

        cfg.MODEL.BILINEAR.ENCODE = CN()
        cfg.MODEL.BILINEAR.ENCODE.ATT_MID_DIM = [64, 32, 64]
        cfg.MODEL.BILINEAR.ENCODE.ATT_MID_DROPOUT = 0.1
        cfg.MODEL.BILINEAR.ENCODE.DROPOUT = 0.5
        cfg.MODEL.BILINEAR.ENCODE.BIFEAT_EMB_DROPOUT = 0.3
        cfg.MODEL.BILINEAR.ENCODE.LAYERS = 3

    def forward(self, batched_inputs, mode=None):
        ret = {}
        if mode == None or mode == 'v':
            vfeats = batched_inputs[kfg.ATT_FEATS]
            ext_vmasks = batched_inputs[kfg.EXT_ATT_MASKS]
            ext_vmasks = torch.cat([ext_vmasks[:,:,:,0:1], ext_vmasks], dim=-1)
            ret.update({ kfg.EXT_ATT_MASKS: ext_vmasks })
            gfeats = []
            vfeats_outs = []
            gfeats.append(vfeats[:, 0])


            if batched_inputs[kfg.STAGE] == 'train':
                abnormal_label = batched_inputs[kfg.ABNORMAL_LABEL].squeeze(1).detach()
                order_label, order = torch.sort(abnormal_label, 1, True)
                maxN = torch.sum(order_label >= 0.5, dim=1)
                em_feats = []
                for i in range(len(maxN)):
                    if maxN[i]>0:
                        em_feats.append(vfeats[i, 1:])
                if len(em_feats)>0:
                    em_feats = torch.stack(em_feats)
                    mem_loss, image_quantized = self.codebook(em_feats, self.memory_space)
                else:
                    mem_loss = None
                ret.update({kfg.MEM_LOSS: mem_loss})
            gv_feat = vfeats[:, 0]
            att_feats = vfeats[:, 1:].contiguous()
            att_mask = batched_inputs[kfg.ATT_MASKS].squeeze(1).squeeze(1)
            feat_arr = [gv_feat]

            for i, layer in enumerate(self.layers):
                gv_feat = layer(gv_feat, att_feats, att_mask, gv_feat, att_feats)
                att_feats_cat = torch.cat([gv_feat.unsqueeze(1).expand_as(att_feats), att_feats], dim=-1)

                att_feats = self.bifeat_emb[i](att_feats_cat) + att_feats
                att_feats = self.layer_norms[i](att_feats)
                feat_arr.append(gv_feat)
                vfeats_outs.append(att_feats)

            gv_feat = torch.cat(feat_arr, dim=-1)
            gv_feat = self.proj(gv_feat)
            gv_feat = self.layer_norm(gv_feat)
            encoder_vfeats = torch.cat([gv_feat.unsqueeze(1), att_feats], dim=1)
            ret.update({ kfg.ATT_FEATS: encoder_vfeats.detach()})

            semic_vfeats = vfeats_outs[self.semic_begin_layers - 1]
            semic_input = torch.cat([gv_feat.unsqueeze(1), semic_vfeats], dim=1)

            for layer_module in self.semic_abnormal_layers:
                semic_input, _ = layer_module(semic_input, ext_vmasks)
            semantics_abnormal_pred = self.semantics_abnormal_pred(semic_input[:, 0, :])

            tag_abnormal_embedding,pred_abnormal_topk = self.semic_label(batched_inputs,semantics_abnormal_pred,'ABNORMAL')

            slot_embed = self.slot_embeddings(self.slot)
            slot_embed = slot_embed.expand(semic_input.shape[0], slot_embed.shape[1], slot_embed.shape[2])
            semantics_normal_embed = torch.cat([tag_abnormal_embedding.detach(), slot_embed], dim=1)
            normal_mask = torch.ones((semantics_normal_embed.shape[0], semantics_normal_embed.shape[1]),
                                     device=semantics_normal_embed.device).to(dtype=next(self.parameters()).dtype)
            normal_mask = (1.0 - normal_mask) * -10000.0
            normal_mask = normal_mask.unsqueeze(1).unsqueeze(2)
            for layer_module in self.semic_normal_layers:
                semantics_normal_embed = layer_module(semantics_normal_embed, encoder_vfeats.detach(), normal_mask, ext_vmasks)
            semantics_normal_pred = self.semantics_normal_pred(semantics_normal_embed[:, -1, :])

            tag_normal_embedding,pred_normal_topk = self.semic_label(batched_inputs,semantics_normal_pred,'NORMAL')



            tag_embedding = torch.cat((tag_abnormal_embedding,tag_normal_embedding),dim = 1)


            ret.update({
                kfg.SEMANTICS_FEATS: tag_embedding,
                kfg.SEMANTICS_ABNORMAL_PRED: semantics_abnormal_pred,
                kfg.SEMANTICS_NORMAL_PRED: semantics_normal_pred,
            })
        return ret
    def semic_label(self, batched_inputs, semantics_pred, type):
        if batched_inputs[kfg.STAGE] == 'train':

            if type == 'ABNORMAL':
                label = batched_inputs[kfg.ABNORMAL_LABEL].squeeze(1).detach()

                topk = 25
            else:
                label = batched_inputs[kfg.NORMAL_LABEL].squeeze(1).detach()
                topk = 25

            prob, pred_topk_label = label.topk(topk, dim=1,largest=True)
            tag_embedding = self.embeddings(pred_topk_label)
            return tag_embedding, pred_topk_label
        else:
            with torch.no_grad():
                if type == 'ABNORMAL':
                    topk = 25
                else:
                    topk = 25

                offline_logit = torch.sigmoid(semantics_pred.detach())
                prob, pred_topk = offline_logit.topk(topk, dim=1, largest=True)


            tag_embedding = self.embeddings(pred_topk)
            return tag_embedding, pred_topk
    def codebook(self,mode_emb,embedding_memory):
        split =  mode_emb.size(1)
        mode_emb_flatten = mode_emb.reshape(-1, mode_emb.size(2))
        distances = (torch.sum(mode_emb_flatten ** 2, dim=1, keepdim=True)
                           + torch.sum(embedding_memory.weight ** 2, dim=1)
                           - 2 * torch.matmul(mode_emb_flatten, embedding_memory.weight.t())).sqrt()
        distances = distances.split(split)
        indices = [
            linear_sum_assignment(d.detach().cpu().numpy())[1]
            for d in distances
        ]
        indices = torch.from_numpy(np.concatenate(indices))
        indices = indices.to(mode_emb_flatten.device)
        quantized = embedding_memory(indices)

        q_latent_loss = F.mse_loss(mode_emb_flatten.detach(), quantized)
        orl_log = F.normalize(embedding_memory.weight,p=2,dim=-1)@F.normalize(embedding_memory.weight.t(),p=2,dim=-1)
        orl_label = torch.zeros(orl_log.size()).to(orl_log)
        orl_label.fill_diagonal_(1)
        orl_loss=F.mse_loss(orl_log,orl_label)
        loss = q_latent_loss +orl_loss

        quantized = mode_emb_flatten + (quantized - mode_emb_flatten).detach()
        quantized = quantized.view(mode_emb.size(0),mode_emb.size(1),-1)

        return loss,quantized
