# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import collections
import copy
import math

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops.boxes import batched_nms

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher, HungarianMatcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss, DeformableDETRsegm)
from .deformable_transformer import build_deforamble_transformer

from .matcher_o2m import Stage2Assigner
import myclip

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, use_global_token, clip_path, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False, mixed_selection=False, use_ms_detr=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        self.mixed_selection = mixed_selection
        self.use_ms_detr = use_ms_detr

        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
        elif mixed_selection:
            self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
        
        ###################################################################
        # begin global query
        ###################################################################
        self.use_global_token = use_global_token
        self.global_token_dim = 768
        self.num_global_token = 5
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.use_global_token:
            self.patch_token_proj = nn.Sequential(
                nn.Conv2d(1024, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim),
            )
            j = self.num_global_token
            k = 1
            while j > 0:
                j = j - k ** 2
                k = k + 1
            if j != 0:
                assert False, "Invalid number of global token"

            self.foundation_model, self.preprocess = myclip.load(clip_path, device)
            self.foundation_model = self.foundation_model.visual
            self.foundation_model.float().eval()
            if self.num_global_token > 1:
                attn_mask = torch.zeros((self.num_global_token + 24 * 24, self.num_global_token + 24 * 24), device=device, dtype=torch.bool)
                attn_mask[self.num_global_token - 1:, :self.num_global_token - 1] = True # original part
                attn_mask[:self.num_global_token - 1] = True
                attn_mask[torch.arange(self.num_global_token - 1), torch.arange(self.num_global_token - 1)] = False
                k = 2
                num_global_token = self.num_global_token - 1
                idx = 0
                while num_global_token > 0:
                    sub_attn_mask = torch.ones((k ** 2, 24, 24), device=device, dtype=torch.bool).reshape(k ** 2, k, 24 // k, k, 24 // k)
                    sub_attn_mask = sub_attn_mask.permute(0, 1, 3, 2, 4).flatten(1, 2) # (k*k, k*k, -1, -1)
                    sub_attn_mask[torch.arange(k ** 2), torch.arange(k ** 2)] = False # only one patch is visible
                    sub_attn_mask = sub_attn_mask.reshape(k ** 2, k, k, 24 // k, 24 // k).permute(0, 1, 3, 2, 4).reshape(k ** 2, 24, 24)
                    attn_mask[idx:idx+k**2, self.num_global_token:] = sub_attn_mask.flatten(1)
                    idx += k**2
                    num_global_token = num_global_token - k ** 2
                    k = k + 1
                self.attn_mask = attn_mask

            for child in self.foundation_model.children():
                for param in child.parameters():
                    param.requires_grad = False
        ###################################################################
        # end global query
        ###################################################################
    

    ###################################################################
    # begin global query
    ###################################################################
    @torch.no_grad()
    def get_image_feat(self, imgs, boxes, img_metas):
        preprocessed = []
        
        for i in range(len(boxes)):
            img = imgs[i]
            device = boxes[i].device
            boxs = boxes[i]
            if len(boxs) == 0:
                continue

            img_shape = img_metas[i] # (W, H)

            boxs = torch.stack([torch.floor(boxs[:,0]-0.001),torch.floor(boxs[:,1]-0.001),torch.ceil(boxs[:,2]),torch.ceil(boxs[:,3])], dim=1).to(torch.int)
            boxs[:,[0,2]].clamp_(min=0,max=img_shape[0])
            boxs[:,[1,3]].clamp_(min=0,max=img_shape[1])

            boxs = boxs.detach().cpu().numpy()
            
            for i, box in enumerate(boxs):
                croped = img.crop(box)
                croped = self.preprocess(croped)
                preprocessed.append(croped)

        if len(preprocessed) == 0:
            return torch.zeros((0, self.global_token_dim), device=device)

        preprocessed = torch.stack(preprocessed).to(device)
        self.foundation_model.eval()
        cls_token = self.foundation_model(preprocessed)
        return cls_token
    
    @torch.no_grad()
    def generate_image_box(self, img_metas, dtype, device):
        all_image_boxes = []
        for i in range(len(img_metas)):
            res_image_box = self.num_global_token
            W, H = img_metas[i]
            image_boxes = []
            j = 1
            while res_image_box > 0:
                x, y = torch.meshgrid([torch.arange(0, W, math.ceil(W / j)), torch.arange(0, H, math.ceil(H / j))], indexing='xy')
                x1 = x.flatten()
                y1 = y.flatten()
                x2 = x1 + math.ceil(W / j)
                y2 = y1 + math.ceil(H / j)
                image_boxes.append(torch.stack([x1, y1, x2, y2], dim=-1))
                res_image_box = res_image_box - len(x1)
                j = j + 1
            image_boxes = torch.cat(image_boxes).to(device).to(dtype)
            all_image_boxes.append(image_boxes)
        return all_image_boxes
    
    @torch.no_grad()
    def get_clip_image_feat_multi_cls_token(self, imgs, boxes, img_metas):
        preprocessed = []
        
        for i in range(len(boxes)):
            img = imgs[i]
            device = boxes[i].device
            boxs = boxes[i]
            if len(boxs) == 0:
                continue

            img_shape = img_metas[i] # (W, H)

            boxs = torch.stack([torch.floor(boxs[:,0]-0.001),torch.floor(boxs[:,1]-0.001),torch.ceil(boxs[:,2]),torch.ceil(boxs[:,3])], dim=1).to(torch.int)
            boxs[:,[0,2]].clamp_(min=0,max=img_shape[0])
            boxs[:,[1,3]].clamp_(min=0,max=img_shape[1])

            boxs = boxs.detach().cpu().numpy()
            
            for i, box in enumerate(boxs):
                croped = img.crop(box)
                croped = self.preprocess(croped)
                preprocessed.append(croped)

        if len(preprocessed) == 0:
            return torch.zeros((0, self.global_token_dim), device=device)

        preprocessed = torch.stack(preprocessed).to(device)
        self.foundation_model.eval()
        features, patch_token = self.foundation_model.forward_multi_cls_token(preprocessed, self.attn_mask, self.num_global_token)
        features = torch.cat([features[:, self.num_global_token-1:], features[:, :self.num_global_token-1]], dim=1)
        b, _, c = patch_token.shape
        
        return features, patch_token.permute(0, 2, 1).reshape(b, c, 24, 24)
    
    ###################################################################
    # end global query
    ###################################################################


    def forward(self, samples: NestedTensor, targets):
        """The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        ###################################################################
        # begin global query
        ###################################################################
        img_metas = [(tgt['w'], tgt['h']) for tgt in targets]
        img_no_normalize = [tgt['img_no_normalize'] for tgt in targets]
        image_box = None
        image_query = None
        num_imgs = len(img_metas)
        if self.use_global_token:
            image_box = self.generate_image_box(img_metas, torch.float, samples.tensors.device)
            if self.num_global_token == 1:
                image_query = self.get_image_feat(img_no_normalize, image_box, img_metas).reshape(num_imgs, self.num_global_token, self.global_token_dim)
            else:
                single_image_box = [box[:1] for box in image_box]
                image_query, patch_token = self.get_clip_image_feat_multi_cls_token(img_no_normalize, single_image_box, img_metas)
            image_box = torch.stack(image_box)
            image_box = box_ops.box_xyxy_to_cxcywh(image_box)
            for i, img_meta in enumerate(img_metas):
                img_w, img_h = img_meta
                factor = image_box.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
                image_box[i] = image_box[i] / factor
        ###################################################################
        # end global query
        ###################################################################

        
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        ###################################################################
        # begin global query
        ###################################################################
        if self.use_global_token:
            mask = torch.zeros((len(img_metas), 24, 24), device=masks[-1].device, dtype=torch.bool)
            src = self.patch_token_proj(patch_token)
            pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
            srcs.append(src)
            masks.append(mask)
            pos.append(pos_l)
        ###################################################################
        # end global query
        ###################################################################

        query_embeds = None
        if not self.two_stage or self.mixed_selection:
            query_embeds = self.query_embed.weight

        hs, hs_o2m, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, anchors = self.transformer(srcs, masks, pos, query_embeds, image_box=image_box, image_query=image_query)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.use_ms_detr:
            outputs_classes_o2m = []
            outputs_coords_o2m = []
            for lvl in range(hs_o2m.shape[0]):
                if lvl == 0:
                    reference = init_reference
                else:
                    reference = inter_references[lvl - 1]
                reference = inverse_sigmoid(reference)
                outputs_class_o2m = self.class_embed[lvl](hs_o2m[lvl])
                tmp = self.bbox_embed[lvl](hs_o2m[lvl])
                if reference.shape[-1] == 4:
                    tmp += reference
                else:
                    assert reference.shape[-1] == 2
                    tmp[..., :2] += reference
                outputs_coord_o2m = tmp.sigmoid()
                outputs_classes_o2m.append(outputs_class_o2m)
                outputs_coords_o2m.append(outputs_coord_o2m)
            outputs_class_o2m = torch.stack(outputs_classes_o2m)
            outputs_coord_o2m = torch.stack(outputs_coords_o2m)
            out['o2m_outputs'] = {'pred_logits': outputs_class_o2m[-1], 'pred_boxes': outputs_coord_o2m[-1]}
            if self.aux_loss:
                out['o2m_outputs']['aux_outputs'] = self._set_aux_loss(outputs_class_o2m, outputs_coord_o2m)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord, 'anchors': anchors}

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, num_queries, focal_alpha=0.25, o2m_matcher_threshold=0.4, o2m_matcher_k=6, use_indices_merge=False, use_anchors_enc_match=True, enc_matcher=None):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.use_indices_merge = use_indices_merge
        self.use_anchors_enc_match = use_anchors_enc_match
        if enc_matcher is None:
            self.enc_matcher = matcher
        else:
            self.enc_matcher = enc_matcher
        self.matcher_o2m = Stage2Assigner(k=o2m_matcher_k, threshold=o2m_matcher_threshold)


    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = (sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2)
                   * src_logits.shape[1])
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    @staticmethod
    def indices_merge(num_queries, o2o_indices, o2m_indices):
        bs = len(o2o_indices)
        temp_indices = torch.zeros(bs, num_queries, dtype=torch.int64).cuda() - 1
        new_one2many_indices = []

        for i in range(bs):
            one2many_fg_inds = o2m_indices[i][0].cuda()
            one2many_gt_inds = o2m_indices[i][1].cuda()
            one2one_fg_inds = o2o_indices[i][0].cuda()
            one2one_gt_inds = o2o_indices[i][1].cuda()
            temp_indices[i][one2one_fg_inds] = one2one_gt_inds
            temp_indices[i][one2many_fg_inds] = one2many_gt_inds
            fg_inds = torch.nonzero(temp_indices[i] >= 0).squeeze(1)
            # fg_inds = torch.argwhere(temp_indices[i] >= 0).squeeze(1)
            gt_inds = temp_indices[i][fg_inds]
            new_one2many_indices.append((fg_inds, gt_inds))

        return new_one2many_indices

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
        # store one-to-one indices for indices merge
        o2o_indices_list = []

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        o2o_indices_list.append(indices)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                o2o_indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # one-to-many losses
        if 'o2m_outputs' in outputs:
            o2m_outputs = outputs['o2m_outputs']
            indices = self.matcher_o2m(o2m_outputs, targets)

            if self.use_indices_merge:
                o2o_indices = o2o_indices_list.pop(0)
                indices = self.indices_merge(self.num_queries, o2o_indices, indices)

            for loss in self.losses:
                kwargs = {}
                l_dict = self.get_loss(loss, o2m_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + '_o2m': v for k, v in l_dict.items()}
                losses.update(l_dict)

            if "aux_outputs" in o2m_outputs:
                for i, aux_outputs in enumerate(o2m_outputs['aux_outputs']):
                    indices = self.matcher_o2m(aux_outputs, targets)

                    if self.use_indices_merge:
                        o2o_indices = o2o_indices_list[i]
                        indices = self.indices_merge(self.num_queries, o2o_indices, indices)
                    
                    for loss in self.losses:
                        if loss == 'masks':
                            # Intermediate masks losses are too costly to compute, we ignore them.
                            continue
                        kwargs = {}
                        if loss == 'labels':
                            # Logging is enabled only for the last layer
                            kwargs['log'] = False
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                        l_dict = {k + f'_{i}_o2m': v for k, v in l_dict.items()}
                        losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])

            # NOTE: this is a hack to use anchors for encoder matching, after matching we need to restore pred_boxes for computing loss
            if self.use_anchors_enc_match:
                enc_outputs['pred_boxes'], enc_outputs['anchors'] = enc_outputs['anchors'], enc_outputs['pred_boxes']
            indices = self.enc_matcher(enc_outputs, bin_targets)
            if self.use_anchors_enc_match:
                enc_outputs['pred_boxes'], enc_outputs['anchors'] = enc_outputs['anchors'], enc_outputs['pred_boxes']

            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + '_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, topk=100):
        super().__init__()
        self.topk = topk
        print("topk for evaluation:", self.topk)
    
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), self.topk, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class NMSPostProcess(nn.Module):
    def __init__(self, topk=100, nms_iou_threshold=0.7):
        super().__init__()
        self.topk = topk
        self.nms_iou_threshold = nms_iou_threshold
        print("Top-k for evaluation: {}, NMS IoU threshold: {}".format(self.topk, nms_iou_threshold))

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        bs, n_queries, n_cls = out_logits.shape

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()

        all_scores = prob.view(bs, n_queries * n_cls).to(out_logits.device)
        all_indexes = torch.arange(n_queries * n_cls)[None].repeat(bs, 1).to(out_logits.device)
        all_boxes = all_indexes // out_logits.shape[2]
        all_labels = all_indexes % out_logits.shape[2]

        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, all_boxes.unsqueeze(-1).repeat(1,1,4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = []
        for b in range(bs):
            box = boxes[b]
            score = all_scores[b]
            lbls = all_labels[b]

            pre_topk = score.topk(10000).indices
            box = box[pre_topk]
            score = score[pre_topk]
            lbls = lbls[pre_topk]

            keep_inds = batched_nms(box, score, lbls, self.nms_iou_threshold)[:self.topk]
            results.append({
                'scores': score[keep_inds],
                'labels': lbls[keep_inds],
                'boxes':  box[keep_inds],
            })

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        num_classes = 250
    device = torch.device(args.device)

    use_global_token = True

    backbone = build_backbone(args)
    transformer = build_deforamble_transformer(args, use_global_token)

    model = DeformableDETR(
        use_global_token,
        args.clip_path,
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        mixed_selection=args.mixed_selection,
        use_ms_detr=args.use_ms_detr,
    )

    if args.masks:
        model = DeformableDETRsegm(model, freeze_detr=(args.frozen_weights is not None))

    matcher = build_matcher(args)
    enc_matcher = HungarianMatcher(cost_class=0, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)

    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef, 'loss_giou': args.giou_loss_coef}
    weight_dict.update(
        {'loss_ce_enc': args.enc_cls_loss_coef, 'loss_bbox_enc': args.enc_bbox_loss_coef, 'loss_giou_enc': args.enc_giou_loss_coef}
    )
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    
    # TODO this is a hack for auxiliary loss weights
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    
    # NOTE: this is a hack to update the one-to-many loss weights
    o2m_weight_dict = {'loss_ce': args.o2m_cls_loss_coef, 'loss_bbox': args.o2m_bbox_loss_coef, 'loss_giou': args.o2m_giou_loss_coef}
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in o2m_weight_dict.items()})
        o2m_weight_dict.update(aux_weight_dict)
    o2m_weight_dict = {k + '_o2m': v for k, v in o2m_weight_dict.items()}
    weight_dict.update(o2m_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(
        num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha, num_queries=args.num_queries, enc_matcher=enc_matcher)
    criterion.to(device)
    
    post_process = PostProcess(topk=args.topk_eval) if args.nms_iou_threshold is None else NMSPostProcess(topk=args.topk_eval, nms_iou_threshold=args.nms_iou_threshold)
    postprocessors = {'bbox': post_process}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
