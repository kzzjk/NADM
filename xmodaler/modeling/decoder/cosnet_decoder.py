# Copyright 2022 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import random
import torch
import torch.nn as nn

from xmodaler.config import configurable
from xmodaler.config import kfg
from .decoder import Decoder
from ..layers.cosnet_layer import COSNetDecBlock
from .build import DECODER_REGISTRY

__all__ = ["COSNetDecoder"]

@DECODER_REGISTRY.register()
class COSNetDecoder(Decoder):
    @configurable
    def __init__(
        self,
        *,
        layer_drop: float,
        num_generation_layers: int,
        cos_generation_layers
    ):
        super(COSNetDecoder, self).__init__()
        self.num_generation_layers = num_generation_layers
        if self.num_generation_layers > 0:
            self.g_layers = cos_generation_layers
        self.layer_drop = layer_drop
        
    @classmethod
    def from_config(cls, cfg):
        cos_generation_layers = nn.ModuleList(
            [COSNetDecBlock(cfg) for _ in range(cfg.MODEL.BERT.NUM_GENERATION_LAYERS)]
        )
        return {
            "num_generation_layers": cfg.MODEL.BERT.NUM_GENERATION_LAYERS,
            "cos_generation_layers": cos_generation_layers,
            "layer_drop": cfg.MODEL.BERT.LAYER_DROP,
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs):
        ret = {}
        vfeats = batched_inputs[kfg.ATT_FEATS]
        ofeats = batched_inputs[kfg.SEMANTICS_FEATS]
        ext_vmasks = batched_inputs[kfg.EXT_ATT_MASKS]
        ext_omasks = torch.ones((ofeats.shape[0], ofeats.shape[1]),
                                device=ofeats.device).to(dtype=next(self.parameters()).dtype)
        ext_omasks = (1.0 - ext_omasks) * -10000.0
        ext_omasks = ext_omasks.unsqueeze(1).unsqueeze(2)
        history_states = batched_inputs.get(kfg.HISTORY_STATES, None)

        g_tfeats_arr = []
        g_tfeats = batched_inputs[kfg.G_TOKEN_EMBED]
        ext_g_tmasks = batched_inputs[kfg.EXT_G_TOKENS_MASKS]
        if len(g_tfeats.size()) == 2:
            g_tfeats = g_tfeats.unsqueeze(1)
        
        if kfg.TIME_STEP in batched_inputs:
            time_step = batched_inputs[kfg.TIME_STEP]
            ext_g_tmasks = ext_g_tmasks[:,:, time_step:time_step+1, 0:time_step+1]
            if kfg.HISTORY_STATES not in batched_inputs:
                shape = list(g_tfeats.size())
                shape[1] = 0
                history_states = [g_tfeats.new(torch.Size(shape))] * self.num_generation_layers
                batched_inputs[kfg.HISTORY_STATES] = history_states
        else:
            history_states = [None] * self.num_generation_layers

        for i, layer_module in enumerate(self.g_layers):
            if history_states[i] is not None:
                history_states[i] = torch.cat([history_states[i], g_tfeats], dim=1)

            dropout_probability = random.uniform(0, 1)
            this_layer_drop = self.layer_drop * (i+1)/len(self.g_layers)
            if self.training and (dropout_probability < this_layer_drop):
                g_tfeats_arr.append(g_tfeats)
            else:
                g_tfeats = layer_module(
                    g_tfeats, 
                    vfeats, 
                    ofeats,
                    ext_g_tmasks, 
                    ext_vmasks, 
                    ext_omasks,
                    history_states[i])
                g_tfeats_arr.append(g_tfeats)

        g_hidden_states = g_tfeats_arr[-1]
        ret.update({ kfg.G_HIDDEN_STATES: g_hidden_states })
        return ret