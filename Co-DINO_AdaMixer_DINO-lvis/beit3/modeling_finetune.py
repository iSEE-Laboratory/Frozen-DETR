# --------------------------------------------------------
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

import torch
import torch.nn as nn
from .modeling_utils import BEiT3Wrapper, _get_base_config, _get_large_config


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


# The implementation code is modified from DeiT (https://github.com/facebookresearch/deit.git)
def load_model_and_may_interpolate(ckpt_path, model, model_key='model|module', model_prefix=''):
    if ckpt_path.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            ckpt_path, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(ckpt_path, map_location='cpu')

    print("Load ckpt from %s" % ckpt_path)
    checkpoint_model = None
    for model_key in model_key.split('|'):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print("Load state_dict by model_key = %s" % model_key)
            break
    
    if checkpoint_model is None:
        checkpoint_model = checkpoint
    
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    for pos_embed_key in ("vision_pos_embed", "pos_embed", "beit3.encoder.embed_positions.A.weight"):
        if pos_embed_key in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model[pos_embed_key]
            embedding_size = pos_embed_checkpoint.shape[-1]
            if pos_embed_key == "beit3.encoder.embed_positions.A.weight":
                # being consistent with Fairseq, which starts from 2 for position embedding
                torchscale_model = True
                num_patches = model.beit3.vision_embed.num_patches
                num_extra_tokens = model.beit3.vision_embed.num_position_embeddings() + 2 - num_patches
            else:
                torchscale_model = False
                num_patches = model.patch_embed.num_patches
                num_extra_tokens = getattr(model, pos_embed_key).shape[-2] - num_patches
            # height (== width) for the checkpoint position embedding
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int(num_patches ** 0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                if torchscale_model:
                    extra_tokens = pos_embed_checkpoint[:num_extra_tokens].unsqueeze(0)
                    # only the position tokens are interpolated
                    pos_tokens = pos_embed_checkpoint[num_extra_tokens:]
                else:
                    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                    # only the position tokens are interpolated
                    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                ori_dtype = pos_tokens.dtype
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2).float()
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False).to(ori_dtype)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                if torchscale_model:
                    new_pos_embed = new_pos_embed.squeeze(0)
                checkpoint_model[pos_embed_key] = new_pos_embed

    load_state_dict(model, checkpoint_model, prefix=model_prefix)


class TwoLayerMLP(nn.Module):
    def __init__(
            self, 
            in_features, 
            hidden_features, 
            out_features, 
            norm_layer, 
            norm_input=True, 
    ):
        super().__init__()
        self.norm1 = norm_layer(in_features) if norm_input else nn.Identity()
        self.dense1 = nn.Linear(in_features, hidden_features)
        self.norm2 = norm_layer(hidden_features)
        self.act = nn.GELU()
        self.dense2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.norm1(x)
        x = self.dense1(x)
        x = self.norm2(x)
        x = self.act(x)
        return self.dense2(x)


class Pooler(nn.Module):
    def __init__(self, input_features, output_features, norm_layer):
        super().__init__()
        self.norm = norm_layer(input_features)
        self.dense = nn.Linear(input_features, output_features)
        self.activation = nn.Tanh()

    def forward(self, x):
        cls_rep = x[:, 0, :]
        cls_rep = self.norm(cls_rep)
        pooled_output = self.dense(cls_rep)
        pooled_output = self.activation(pooled_output)
        return pooled_output



class BEiT3ForImageClassification(BEiT3Wrapper):
    def __init__(
            self, 
            args, 
            num_classes, 
            norm_layer=nn.LayerNorm, 
            **kwargs
    ):
        super(BEiT3ForImageClassification, self).__init__(args=args)
        embed_dim = args.encoder_embed_dim


    def forward(self, image, **kwargs):
        B, _, H, W = image.shape
        num_patch = H // 16
        x = self.beit3(textual_tokens=None, visual_tokens=image)["encoder_out"]
        # t = x[:, 1:, :]
        # cls_x = self.fc_norm(t.mean(1))
        return x[:, 1:, :].mean(1), x[:, 1:, :].reshape(B, num_patch, num_patch, x.shape[-1])
        # return x[:, 0], x[:, 1:, :].reshape(B, num_patch, num_patch, x.shape[-1])



def beit3_base_patch16_224_imageclassification(img_size, ckpt_path, pretrained=False, **kwargs):
    args = _get_base_config(img_size=img_size, **kwargs)
    args.normalize_output = False
    model = BEiT3ForImageClassification(args, num_classes=80, **kwargs).float()
    load_model_and_may_interpolate(ckpt_path, model)
    return model


def beit3_large_patch16_224_imageclassification(img_size, ckpt_path, pretrained=False, **kwargs):
    args = _get_large_config(img_size=img_size, **kwargs)
    args.normalize_output = False
    model = BEiT3ForImageClassification(args, num_classes=80, **kwargs).float()
    load_model_and_may_interpolate(ckpt_path, model)
    return model

