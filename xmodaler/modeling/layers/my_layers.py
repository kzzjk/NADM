
import torch
import torch.nn as nn
from xmodaler.modeling.layers import get_act_layer
from .scattention import SCAttention
import numpy as np
import math
import torch.nn.functional as F
from xmodaler.config import configurable
from ..layers.create_act import get_activation
from .bert import BertAttention, BertIntermediate, BertOutput
__all__ = ["myLowRankBilinearLayer","myLowRankBilineardecoerLayer","DisTrans"]


class LowRank(nn.Module):
    def __init__(
            self,
            *,
            embed_dim: int,
            att_heads: int,
            att_mid_dim: list,
            att_mid_drop: float,
            act_type: str,
            elu_alpha: float,
            memory_num: int,
            memory_space: torch.Tensor

    ):
        super(LowRank, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = att_heads
        self.head_dim = embed_dim // self.num_heads
        self.scaling = self.head_dim ** -0.5
        self.memory_num = memory_num
        self.m_k = memory_space
        self.m_v = memory_space


        output_dim = 2 * embed_dim if act_type == 'GLU' else embed_dim

        sequential = []
        sequential.append(nn.Linear(embed_dim, output_dim))
        act = get_act_layer(act_type)(elu_alpha)
        if act is not None:
            sequential.append(act)
        sequential.append(torch.nn.GroupNorm(self.num_heads, embed_dim))
        self.in_proj_q = nn.Sequential(*sequential)

        sequential = []
        sequential.append(nn.Linear(embed_dim, output_dim))
        act = get_act_layer(act_type)(elu_alpha)
        if act is not None:
            sequential.append(act)
        sequential.append(torch.nn.GroupNorm(self.num_heads, embed_dim))
        self.in_proj_k = nn.Sequential(*sequential)

        sequential = []
        sequential.append(nn.Linear(embed_dim, output_dim))
        act = get_act_layer(act_type)(elu_alpha)
        if act is not None:
            sequential.append(act)
        sequential.append(torch.nn.GroupNorm(self.num_heads, embed_dim))
        self.in_proj_v1 = nn.Sequential(*sequential)

        sequential = []
        sequential.append(nn.Linear(embed_dim, output_dim))
        act = get_act_layer(act_type)(elu_alpha)
        # act = nn.CELU(elu_alpha)
        if act is not None:
            sequential.append(act)
        sequential.append(torch.nn.GroupNorm(self.num_heads, embed_dim))
        self.in_proj_v2 = nn.Sequential(*sequential)

        self.attn_net = SCAttention(att_mid_dim, att_mid_drop)

    # query -- batch_size * qdim
    # value -- batch_size * att_num * vdim
    def forward(self, query, key, mask, value1, value2, precompute=False):
        batch_size = query.size()[0]
        q = self.in_proj_q(query)
        v1 = self.in_proj_v1(value1)

        q = q.view(batch_size, self.num_heads, self.head_dim)
        v1 = v1.view(batch_size, self.num_heads, self.head_dim)

        if precompute == False:
            key = key.view(-1, key.size()[-1])
            value2 = value2.view(-1, value2.size()[-1])
            k = self.in_proj_k(key)
            v2 = self.in_proj_v2(value2)
            k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v2 = v2.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            k = key
            v2 = value2
        m_k = np.sqrt(self.head_dim) * self.m_k.expand(batch_size, self.memory_num, self.num_heads * self.head_dim).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        m_v = np.sqrt(self.memory_num) * self.m_v.expand(batch_size, self.memory_num, self.num_heads * self.head_dim).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = torch.cat([k, m_k], -2)
        v2 = torch.cat([v2, m_v], -2)

        if mask.size(-1)<self.memory_num:
            mask_m = torch.cat((mask, mask[:, :self.memory_num-mask.size(-1)]), dim=-1)
        else:
            mask_m = mask[:, :self.memory_num]
        mask = torch.cat((mask, mask_m), dim=-1)

        attn_map = q.unsqueeze(-2) * k
        attn = self.attn_net(attn_map, mask, v1, v2)
        attn = attn.view(batch_size, self.num_heads * self.head_dim)
        return attn

    # query -- batch_size * seq_num * qdim
    # value -- batch_size * att_num * vdim
    def forward2(self, query, key, mask, value1, value2, precompute=False):
        batch_size = query.size()[0]
        query = query.view(-1, query.size()[-1])
        value1 = value1.view(-1, value1.size()[-1])

        q = self.in_proj_q(query)
        v1 = self.in_proj_v1(value1)

        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v1 = v1.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        if precompute == False:
            key = key.view(-1, key.size()[-1])
            value2 = value2.view(-1, value2.size()[-1])
            k = self.in_proj_k(key)
            v2 = self.in_proj_v2(value2)
            k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v2 = v2.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            k = key
            v2 = value2
        m_k = np.sqrt(self.head_dim) * self.m_k.expand(batch_size, self.memory_num, self.num_heads * self.head_dim).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        m_v = np.sqrt(self.memory_num) * self.m_v.expand(batch_size, self.memory_num, self.num_heads * self.head_dim).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = torch.cat([k, m_k], -2)
        v2 = torch.cat([v2, m_v], -2)

        mask = torch.cat((mask, torch.ones(mask.size(0), mask.size(1), self.memory_num).to(mask)), dim=-1)
        attn_map = q.unsqueeze(-2) * k.unsqueeze(-3)
        attn = self.attn_net.forward(attn_map, mask, v1, v2).transpose(1, 2).contiguous()
        attn = attn.view(batch_size, -1, self.num_heads * self.head_dim)
        return attn

    def precompute(self, key, value2):
        batch_size = value2.size()[0]
        key = key.view(-1, key.size()[-1])
        value2 = value2.view(-1, value2.size()[-1])

        k = self.in_proj_k(key)
        v2 = self.in_proj_v2(value2)

        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v2 = v2.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        return k, v2


class myLowRankBilinearLayer(nn.Module):
    def __init__(
            self,
            *,
            embed_dim: int,
            att_heads: int,
            att_mid_dim: list,
            att_mid_drop: float,
            dropout: float,
            act_type: str,
            elu_alpha: float,
            memory_num: int,
            memory_space: torch.Tensor
    ):
        super(myLowRankBilinearLayer, self).__init__()
        self.encoder_attn = LowRank(
            embed_dim=embed_dim,
            att_heads=att_heads,
            att_mid_dim=att_mid_dim,
            att_mid_drop=att_mid_drop,
            act_type=act_type,
            elu_alpha=elu_alpha,
            memory_num=memory_num,
            memory_space=memory_space
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(
            self,
            x,
            key=None,
            mask=None,
            value1=None,
            value2=None,
            precompute=False
    ):
        x = self.encoder_attn(
            query=x,
            key=key if key is not None else x,
            mask=mask,
            value1=value1 if value1 is not None else x,
            value2=value2 if value2 is not None else x,
            precompute=precompute
        )
        if self.dropout is not None:
            x = self.dropout(x)
        return x

    def precompute(self, key, value2):
        return self.encoder_attn.precompute(key, value2)
class myLowRankBilineardecoerLayer(nn.Module):
    def __init__(
            self,
            *,
            embed_dim: int,
            att_heads: int,
            att_mid_dim: list,
            att_mid_drop: float,
            dropout: float,
            act_type: str,
            elu_alpha: float,
            memory_num: int,
            memory_space: torch.Tensor,
            last_layer: bool,
            emb_act_type: str,
            bifeat_emb_dropout: float
    ):
        super(myLowRankBilineardecoerLayer, self).__init__()
        self.decoder_attn = LowRank(
            embed_dim=embed_dim,
            att_heads=att_heads,
            att_mid_dim=att_mid_dim,
            att_mid_drop=att_mid_drop,
            act_type=act_type,
            elu_alpha=elu_alpha,
            memory_num=memory_num,
            memory_space=memory_space
        )
        self.last_layer = last_layer
        self.decoder_crossattn = LowRank(
            embed_dim=embed_dim,
            att_heads=att_heads,
            att_mid_dim=att_mid_dim,
            att_mid_drop=att_mid_drop,
            act_type=act_type,
            elu_alpha=elu_alpha,
            memory_num=memory_num,
            memory_space=memory_space
        )
        if self.last_layer == False:
            self.bifeat_emb = nn.Sequential(
                nn.Linear(2 * embed_dim, embed_dim),
                get_act_layer(emb_act_type)(),
                nn.Dropout(bifeat_emb_dropout)
            )
            self.layer_norm_x = torch.nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.word_dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.layer_norm_gx = torch.nn.LayerNorm(embed_dim)
        self.layer_norm_cross = torch.nn.LayerNorm(embed_dim)
    def forward(
            self,
            gx,
            x,
            encoder_out,
            att_mask,
            seq_mask,
            key=None,
            value2=None,
            precompute=False
    ):
        word_x = x
        residual = x
        x = self.decoder_attn.forward2(
            query=gx,
            key=x,
            mask=seq_mask,
            value1=gx,
            value2=x,
            precompute=precompute
        )
        if self.word_dropout is not None:
            x = self.word_dropout(x)
        x = residual + x

        residual = x
        x = self.layer_norm_cross(x)
        x = self.decoder_crossattn.forward2(
            query=x,
            key=encoder_out if precompute == False else key,
            mask=att_mask,
            value1=x,
            value2=encoder_out if precompute == False else value2,
            precompute=precompute
        )
        if self.dropout is not None:
            x = self.dropout(x)

        gx = residual + x
        gx = self.layer_norm_gx(gx)
        if self.last_layer == False:
            x_ = torch.cat([gx, word_x], dim=-1)
            x = self.bifeat_emb(x_) + word_x
            x = self.layer_norm_x(x)
        else:
            x = None
        return gx,x

    def precompute(self, key, value2):
        return self.decoder_attn.precompute(key, value2)
class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DisAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.mu_proj = nn.Linear(int(dim/2), dim)
        self.mu_proj_drop = nn.Dropout(proj_drop)
        self.logsig_proj = nn.Linear(int(dim/2), dim)
        self.logsig_proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        ) # (3, B, mu_heads_num+logsig_heads_num, n, dim_heads)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C).reshape(B, N, 2, int(C/2))

        mu = x[:,:,0,:]
        logsigma = x[:,:,1,:]
        mu = self.mu_proj(mu)
        mu = self.mu_proj_drop(mu)
        logsigma = self.logsig_proj(logsigma)
        logsigma = self.logsig_proj_drop(logsigma)
        return mu, logsigma, attn


class DisTrans(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.1,
        attn_drop=0.1,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.act = act_layer()
        self.norm1 = norm_layer(dim)
        self.attn = DisAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mu_mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.logsig_mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, mask=None):
        x_ = self.norm1(self.act(self.fc(x)))
        mu, logsigma, attn = self.attn(x_, mask=mask)
        mu = x + self.drop_path(mu)
        mu = mu + self.drop_path(self.mu_mlp(self.norm2(mu)))
        logsigma = logsigma + self.drop_path(self.logsig_mlp(self.norm3(logsigma)))
        return mu, logsigma,



class myCOSJointAttention(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        hidden_size,
        num_attention_heads,
        attention_probs_dropout_prob,
        layer_norm_eps,
        hidden_dropout_prob
    ):
        super(myCOSJointAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads)
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.attn_dropout = nn.Dropout(attention_probs_dropout_prob)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        # self.key = nn.Linear(hidden_size, self.all_head_size)
        # self.value = nn.Linear(hidden_size, self.all_head_size)

        #self.c_proj = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.resid_dropout = nn.Dropout(hidden_dropout_prob)

        ##################################################################
        self.v_attn = nn.Linear(hidden_size, hidden_size * 2)
        self.o_attn = nn.Linear(hidden_size, hidden_size * 2)
        self.vo_proj = nn.Linear(2 * hidden_size, hidden_size)
        # self.vl_proj = nn.Linear(2 * hidden_size, hidden_size)
        # self.ol_proj = nn.Linear(2 * hidden_size, hidden_size)
        self.gate_attn = nn.Linear(hidden_size*2, hidden_size)
        self.gate = nn.Sigmoid()
        self.tanh = nn.Tanh()
        ##################################################################

    @classmethod
    def from_config(cls, cfg):
        return {
            "hidden_size": cfg.MODEL.BERT.HIDDEN_SIZE,
            "num_attention_heads": cfg.MODEL.BERT.NUM_ATTENTION_HEADS,
            "attention_probs_dropout_prob": cfg.MODEL.BERT.ATTENTION_PROBS_DROPOUT_PROB,
            "layer_norm_eps": 1e-12,
            "hidden_dropout_prob": cfg.MODEL.BERT.HIDDEN_DROPOUT_PROB
        }

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)

        shape_list = list(range(len(new_x_shape)))
        shape_list[-2], shape_list[-3] = shape_list[-3], shape_list[-2]
        return x.permute(shape_list)

    def attn(self, mixed_query_layer, mixed_key_layer, mixed_value_layer, attention_mask):
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        shape_list = list(range(len(context_layer.shape)))
        shape_list[-2], shape_list[-3] = shape_list[-3], shape_list[-2]
        context_layer = context_layer.permute(shape_list).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        outputs = context_layer.view(*new_context_layer_shape)
        return outputs

    def forward(
        self,
        hidden_states,
        attention_mask,
        v_feats,
        o_feats,
        v_attention_mask,
        o_attention_mask,
        history_states=None
    ):
        input_tensor = hidden_states

        mixed_query_layer = self.query(hidden_states)

        v_mixed_key_layer, v_mixed_value_layer = self.v_attn(v_feats).split(self.all_head_size, dim=2)
        o_mixed_key_layer, o_mixed_value_layer = self.o_attn(o_feats).split(self.all_head_size, dim=2)


        v_outputs = self.attn(mixed_query_layer, v_mixed_key_layer, v_mixed_value_layer, v_attention_mask)
        o_outputs = self.attn(mixed_query_layer, o_mixed_key_layer, o_mixed_value_layer, o_attention_mask)



        vo_outputs = self.vo_proj(torch.cat([v_outputs, o_outputs], dim=-1))

        # gate = self.gate_attn(torch.cat([o_outputs, vo_outputs], dim=-1))
        # gate = self.gate(gate)
        # outputs = gate * o_outputs + (1 - gate) * vo_outputs


        outputs = self.resid_dropout(vo_outputs)
        outputs = self.LayerNorm(outputs + input_tensor)
        return outputs

class COSBertIntermediate(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        hidden_size: int,
        hidden_act: str,
        intermediate_size: int,
        intermediate_drop: float
    ):
        super(COSBertIntermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = get_activation(hidden_act)
        self.dropout = nn.Dropout(intermediate_drop)

    @classmethod
    def from_config(cls, cfg):
        return {
            "hidden_size": cfg.MODEL.BERT.HIDDEN_SIZE,
            "hidden_act": cfg.MODEL.BERT.HIDDEN_ACT,
            "intermediate_size": cfg.MODEL.BERT.INTERMEDIATE_SIZE,
            "intermediate_drop": cfg.MODEL.BERT.INTERMEDIATE_DROP
        }

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class COSBertOutput(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        layer_norm_eps: float,
        ffn_dropout_prob: float
    ):
        super(COSBertOutput, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(ffn_dropout_prob)

    @classmethod
    def from_config(cls, cfg):
        return {
            "hidden_size": cfg.MODEL.BERT.HIDDEN_SIZE,
            "intermediate_size": cfg.MODEL.BERT.INTERMEDIATE_SIZE,
            "layer_norm_eps": 1e-12,
            "ffn_dropout_prob": cfg.MODEL.BERT.FFN_DROPOUT_PROB
        }

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class myCOSNetDecBlock(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        attention,
        cross_attention,
        bert_intermediate,
        bert_output
    ):
        super(myCOSNetDecBlock, self).__init__()
        self.attn = attention
        self.cross_attn = cross_attention
        self.intermediate = bert_intermediate
        self.output = bert_output

    @classmethod
    def from_config(cls, cfg):
        return {
            'attention': BertAttention(cfg),
            'cross_attention': myCOSJointAttention(cfg),
            "bert_intermediate": COSBertIntermediate(cfg),
            "bert_output": COSBertOutput(cfg)
        }

    def forward(self,
        lang_feats,
        v_feats,
        o_feats,
        lang_attention_mask=None,
        v_attention_mask=None,
        o_attention_mask=None,
        t_history_states=None
    ):
        x, _ = self.attn(lang_feats, lang_attention_mask, t_history_states)
        x = self.cross_attn(
            x,
            lang_attention_mask,
            v_feats,
            o_feats,
            v_attention_mask,
            o_attention_mask,
            t_history_states
        )
        intermediate_output = self.intermediate(x)
        layer_output = self.output(intermediate_output, x)
        return layer_output