import torch
import torch.nn as nn
import torchvision.models as models
import os
import gzip
import logging
logger = logging.getLogger(__name__)
import numpy as np
from typing import Any, List, Tuple, Type, Union,Callable, Optional
import enum
import torch
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import model_urls, ResNet, BasicBlock, Bottleneck
from pathlib import Path

from .build import BACKBONE_REGISTRY
from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.functional import pad_tensor,dict_to_cuda
TypeSkipConnections = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
TypeImageEncoder = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


__all__ = ["BioResnet50"]

@BACKBONE_REGISTRY.register()
class BioResnet50(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        freeze_encoder: bool,
        pretrained_path: str,
        **kwargs
    ):
        super(BioResnet50, self).__init__()
        self.encoder = ImageEncoder("resnet50")
        self.freeze_encoder = freeze_encoder
        self.train()
        pretrained= Path(pretrained_path)
        if pretrained_path !='':
            if not isinstance(pretrained, (str, Path)):
                raise TypeError(f"Expected a string or Path, got {type(pretrained)}")
            state_dict = torch.load(pretrained, map_location="cpu")
            self.load_state_dict(state_dict,strict=False)

    @classmethod
    def from_config(cls, cfg):

        return {
            "freeze_encoder": cfg.MODEL.BACKBONE.FREEZE_ENCODER,
            "pretrained_path": cfg.MODEL.BACKBONE.PRETRAINED,
        }
    def train(self, mode: bool = True) -> Any:
        super().train(mode=mode)
        if self.freeze_encoder:
            self.encoder.train(mode=False)
        return self

    def forward(self, batched_inputs):
        ret = {}
        x = batched_inputs[kfg.IMAGE_FEATS]
        with torch.set_grad_enabled(not self.freeze_encoder):
            if x.size(1)==2:
                patch_x0, pooled_x0 = self.encoder(x[:, 0, :, :], return_patch_embeddings=True)
                batch_size, feat_size, _, _ = patch_x0.shape
                patch_feats0 = patch_x0.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
                patch_x1, pooled_x1 = self.encoder(x[:, 1, :, :], return_patch_embeddings=True)
                batch_size, feat_size, _, _ = patch_x1.shape
                patch_feats1 = patch_x1.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
                patch_feats = torch.cat((patch_feats0,patch_feats1), dim=1)
                pooled_x = (pooled_x0 + pooled_x1)/2
                pooled_x = pooled_x.unsqueeze(1)
            else:
                patch_x, pooled_x = self.encoder(x, return_patch_embeddings=True)
                batch_size, feat_size, _, _ = patch_x.shape
                patch_feats = patch_x.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
                pooled_x = pooled_x.unsqueeze(1)
        #patch_feats, mask = pad_tensor(patch_feats, padding_value=0, use_mask=True)
        mask = torch.ones((patch_feats.size(0),patch_feats.size(1)),dtype=torch.float32)
        ret.update({ kfg.ATT_FEATS: patch_feats, kfg.ATT_MASKS: mask,kfg.GLOBAL_FEATS: pooled_x})
        dict_to_cuda(ret)
        return ret


@enum.unique
class ResnetType(str, enum.Enum):
    RESNET50 = "resnet50"
class ImageEncoder(nn.Module):
    def __init__(self, img_model_type: str):
        super().__init__()
        self.img_model_type = img_model_type
        self.encoder = self._create_encoder()

    def _create_encoder(self, **kwargs: Any) -> nn.Module:
        supported = ResnetType.RESNET50
        if self.img_model_type not in supported:
            raise NotImplementedError(f"Image model type \"{self.img_model_type}\" must be in {supported}")
        encoder_class = resnet50
        encoder = encoder_class(pretrained=True, **kwargs)
        return encoder

    def forward(self, x: torch.Tensor, return_patch_embeddings: bool = False) -> TypeImageEncoder:
        x = self.encoder(x)
        avg_pooled_emb = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)), 1)
        if return_patch_embeddings:
            return x, avg_pooled_emb

        return avg_pooled_emb
class ResNetHIML(ResNet):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def forward(self, x: torch.Tensor,
                return_intermediate_layers: bool = False) -> Union[torch.Tensor, TypeSkipConnections]:
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)

        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        if return_intermediate_layers:
            return x0, x1, x2, x3, x4
        else:
            return x4


def _resnet(arch: str, block: Type[Union[BasicBlock, Bottleneck]], layers: List[int],
            pretrained: bool, progress: bool, **kwargs: Any) -> ResNetHIML:
    model = ResNetHIML(block=block, layers=layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model
def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetHIML:
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)
