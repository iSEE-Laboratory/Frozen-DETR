# --------------------------------------------------------
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

import math
import torch
import torch.nn as nn
from torchscale.model.BEiT3 import BEiT3
from torchscale.architecture.config import EncoderConfig


def _get_base_config(
        img_size=224, patch_size=16, drop_path_rate=0, 
        checkpoint_activations=None, mlp_ratio=4, vocab_size=64010, **kwargs
):
    return EncoderConfig(
        img_size=img_size, patch_size=patch_size, vocab_size=vocab_size, multiway=True, 
        layernorm_embedding=False, normalize_output=True, no_output_layer=True, 
        drop_path_rate=drop_path_rate, encoder_embed_dim=768, encoder_attention_heads=12, 
        encoder_ffn_embed_dim=int(768 * mlp_ratio), encoder_layers=12, 
        checkpoint_activations=checkpoint_activations, 
    )


def _get_large_config(
        img_size=224, patch_size=16, drop_path_rate=0, 
        checkpoint_activations=None, mlp_ratio=4, vocab_size=64010, **kwargs
):
    return EncoderConfig(
        img_size=img_size, patch_size=patch_size, vocab_size=vocab_size, multiway=True, 
        layernorm_embedding=False, normalize_output=True, no_output_layer=True, 
        drop_path_rate=drop_path_rate, encoder_embed_dim=1024, encoder_attention_heads=16, 
        encoder_ffn_embed_dim=int(1024 * mlp_ratio), encoder_layers=24, 
        checkpoint_activations=checkpoint_activations, 
    )


class BEiT3Wrapper(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        self.beit3 = BEiT3(args)
        # self.apply(self._init_weights)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_num_layers(self):
        return self.beit3.encoder.num_layers

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'beit3.encoder.embed_positions.A.weight', 'beit3.vision_embed.cls_token', 'logit_scale'}

    
