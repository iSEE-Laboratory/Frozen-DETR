# Copyright (c) OpenMMLab. All rights reserved.
from .resnet import ResNet, ResNetV1d
from .swin import SwinTransformer
from .clip_res_backbone import CLIPCNNBackbone
from .swin_transformer import SwinTransformerV1
from .vit import ViT

__all__ = [
    'ResNet', 'ResNetV1d', 'SwinTransformer'
]
