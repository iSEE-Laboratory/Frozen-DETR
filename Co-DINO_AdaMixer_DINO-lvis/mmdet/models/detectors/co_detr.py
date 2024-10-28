import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import bbox2result
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
from mmdet.models.detectors.base import BaseDetector
import myclip
import numpy as np
from PIL import Image
import math
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def interpolate_positional_embedding(positional_embedding, grid, size):
    if size == (grid, grid):
        return positional_embedding
    cls_token = positional_embedding[0]
    positional_embedding = positional_embedding[1:]
    positional_embedding = torch.reshape(positional_embedding, (grid, grid, -1)).permute(2, 0, 1).unsqueeze(0)
    positional_embedding = F.interpolate(positional_embedding, size=size, mode='bilinear')
    positional_embedding = positional_embedding.squeeze(0).flatten(1).permute(1, 0)
    positional_embedding = torch.cat([cls_token.unsqueeze(0), positional_embedding])
    return positional_embedding





@DETECTORS.register_module()
class CoDETR(BaseDetector):
    def __init__(self,
                 backbone,
                 neck=None,
                 query_head=None,
                 rpn_head=None,
                 roi_head=[None],
                 bbox_head=[None],
                 train_cfg=[None, None],
                 test_cfg=[None, None],
                 pretrained=[None, None],
                 init_cfg=None,
                 with_pos_coord=True,
                 with_attn_mask=True,
                 eval_module='detr',
                 eval_index=0,
                 use_global_token=False,
                 foundation_model_type='openai_clip',
                 foundation_model_path=None,
                 change_foundation_model_size=None,
                 global_token_dim=768,
                 num_global_token=1,
                 use_dynamic_image_token=False,
                 use_co_head=True):
        super(CoDETR, self).__init__(init_cfg)
        self.with_pos_coord = with_pos_coord
        self.with_attn_mask = with_attn_mask
        # Module for evaluation, ['detr', 'one-stage', 'two-stage']
        self.eval_module = eval_module
        # Module index for evaluation
        self.eval_index = eval_index
        self.backbone = build_backbone(backbone)

        head_idx = 0

        if neck is not None:
            self.neck = build_neck(neck)

        if query_head is not None:
            query_head.update(train_cfg=train_cfg[head_idx] if (train_cfg is not None and train_cfg[head_idx] is not None) else None)
            query_head.update(test_cfg=test_cfg[head_idx])
            self.query_head = build_head(query_head)
            self.query_head.init_weights()
            head_idx += 1

        self.use_co_head = use_co_head
        if self.use_co_head:
            if rpn_head is not None:
                rpn_train_cfg = train_cfg[head_idx].rpn if (train_cfg is not None and train_cfg[head_idx] is not None) else None
                rpn_head_ = rpn_head.copy()
                rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg[head_idx].rpn)
                self.rpn_head = build_head(rpn_head_)
                self.rpn_head.init_weights()

        self.roi_head = nn.ModuleList()
        if self.use_co_head:
            for i in range(len(roi_head)):
                if roi_head[i]:
                    rcnn_train_cfg = train_cfg[i+head_idx].rcnn if (train_cfg and train_cfg[i+head_idx] is not None) else None
                    roi_head[i].update(train_cfg=rcnn_train_cfg)
                    roi_head[i].update(test_cfg=test_cfg[i+head_idx].rcnn)
                    self.roi_head.append(build_head(roi_head[i]))
                    self.roi_head[-1].init_weights()

        self.bbox_head = nn.ModuleList()
        if self.use_co_head:
            for i in range(len(bbox_head)):
                if bbox_head[i]:
                    bbox_head[i].update(train_cfg=train_cfg[i+head_idx+len(self.roi_head)] if (train_cfg and train_cfg[i+head_idx+len(self.roi_head)] is not None) else None)
                    bbox_head[i].update(test_cfg=test_cfg[i+head_idx+len(self.roi_head)])
                    self.bbox_head.append(build_head(bbox_head[i]))  
                    self.bbox_head[-1].init_weights() 

        self.head_idx = head_idx
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        ###################################################################
        # begin global query
        ###################################################################
        self.use_global_token = use_global_token
        self.use_dynamic_image_token = use_dynamic_image_token
        self.global_token_dim = global_token_dim
        self.num_global_token = num_global_token
        self.foundation_model_type = foundation_model_type
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.use_global_token:

            j = self.num_global_token
            k = 1
            while j > 0:
                j = j - k ** 2
                k = k + 1
            if j != 0:
                assert False, "Invalid number of global token"
            
            if self.foundation_model_type == 'openai_clip':
                self.foundation_model, self.preprocess = myclip.load(foundation_model_path, device)
                self.foundation_model = self.foundation_model.visual
                self.foundation_model.float().eval()
                self.base_patch_size = 24
                if change_foundation_model_size is not None:
                    self.preprocess = transform(change_foundation_model_size)
                    positional_embedding = interpolate_positional_embedding(
                        self.foundation_model.positional_embedding, 24, (change_foundation_model_size // 14, ) * 2
                    )
                    self.foundation_model.positional_embedding = nn.Parameter(positional_embedding, requires_grad=False)
                    self.base_patch_size = change_foundation_model_size // 14
                if self.num_global_token > 1:
                    attn_mask = torch.zeros((self.num_global_token + self.base_patch_size * self.base_patch_size, self.num_global_token + self.base_patch_size * self.base_patch_size), device=device, dtype=torch.bool)
                    attn_mask[self.num_global_token - 1:, :self.num_global_token - 1] = True # original part
                    attn_mask[:self.num_global_token - 1] = True
                    attn_mask[torch.arange(self.num_global_token - 1), torch.arange(self.num_global_token - 1)] = False
                    k = 2
                    num_global_token = self.num_global_token - 1
                    idx = 0
                    while num_global_token > 0:
                        sub_attn_mask = torch.ones((k ** 2, self.base_patch_size, self.base_patch_size), device=device, dtype=torch.bool).reshape(k ** 2, k, self.base_patch_size // k, k, self.base_patch_size // k)
                        sub_attn_mask = sub_attn_mask.permute(0, 1, 3, 2, 4).flatten(1, 2) # (k*k, k*k, -1, -1)
                        sub_attn_mask[torch.arange(k ** 2), torch.arange(k ** 2)] = False # only one patch is visible
                        sub_attn_mask = sub_attn_mask.reshape(k ** 2, k, k, self.base_patch_size // k, self.base_patch_size // k).permute(0, 1, 3, 2, 4).reshape(k ** 2, self.base_patch_size, self.base_patch_size)
                        attn_mask[idx:idx+k**2, self.num_global_token:] = sub_attn_mask.flatten(1)
                        idx += k**2
                        num_global_token = num_global_token - k ** 2
                        k = k + 1
                    self.attn_mask = attn_mask
            else:
                raise NotImplementedError
            for child in self.foundation_model.children():
                for param in child.parameters():
                    param.requires_grad = False
        ###################################################################
        # end global query
        ###################################################################

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_query_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'query_head') and self.query_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None and len(self.roi_head)>0

    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self, 'roi_head') and self.roi_head[0].with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self, 'roi_head') and self.roi_head is not None and len(self.roi_head)>0)
                or (hasattr(self, 'bbox_head') and self.bbox_head is not None and len(self.bbox_head)>0))

    @property
    def with_mask(self):
        """bool: whether the detector has a mask head"""
        return (hasattr(self, 'roi_head') and self.roi_head is not None and len(self.roi_head)>0 and self.roi_head[0].with_mask)
    
    def extract_feat(self, img, img_metas=None):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    # over-write `forward_dummy` because:
    # the forward of bbox_head requires img_metas
    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        warnings.warn('Warning! MultiheadAttention in DETR does not '
                      'support flops computation! Do not use the '
                      'results in your papers!')

        batch_size, _, height, width = img.shape
        dummy_img_metas = [
            dict(
                batch_input_shape=(height, width),
                img_shape=(height, width, 3)) for _ in range(batch_size)
        ]
        x = self.extract_feat(img)
        outs = self.query_head(x, dummy_img_metas)
        return outs
    

    ###################################################################
    # begin global query
    ###################################################################
    @torch.no_grad()
    def get_image_feat(self, imgs, boxes, img_metas):
        preprocessed = []
        
        for i in range(len(boxes)):
            img = imgs[i]
            device = img.device
            boxs = boxes[i]
            if len(boxs) == 0:
                continue
            img = np.array(img.detach().cpu()).astype(np.uint8)
            img = Image.fromarray(img.transpose(1,2,0))
            img_shape = (img_metas[i]['img_shape'][1], img_metas[i]['img_shape'][0]) # (W, H)

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
        cls_token, patch_token = self.foundation_model(preprocessed)
        return cls_token
    
    @torch.no_grad()
    def generate_image_box(self, img_metas, dtype, device):
        all_image_boxes = []
        for i in range(len(img_metas)):
            res_image_box = self.num_global_token
            W, H = img_metas[i]['img_shape'][1], img_metas[i]['img_shape'][0]
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
            device = img.device
            boxs = boxes[i]
            if len(boxs) == 0:
                continue
            img = np.array(img.detach().cpu()).astype(np.uint8)
            img = Image.fromarray(img.transpose(1,2,0))
            img_shape = (img_metas[i]['img_shape'][1], img_metas[i]['img_shape'][0]) # (W, H)

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
        
        return features, patch_token.permute(0, 2, 1).reshape(b, c, self.base_patch_size, self.base_patch_size)
    
    ###################################################################
    # end global query
    ###################################################################

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        img_no_normalize = img.clone().detach()
        img_no_normalize = img_no_normalize * torch.tensor([58.395, 57.12, 57.375], device=img_no_normalize.device).reshape(1, 3, 1, 1) \
                        + torch.tensor([123.675, 116.28, 103.53], device=img_no_normalize.device).reshape(1, 3, 1, 1)
        ###################################################################
        # begin global query
        ###################################################################
        image_box = None
        image_query = None
        patch_token = None
        num_imgs = len(img_metas)
        if self.use_global_token:
            image_box = self.generate_image_box(img_metas, gt_bboxes[0].dtype, gt_bboxes[0].device)
            if self.num_global_token == 1:
                image_query = self.get_image_feat(img_no_normalize, image_box, img_metas).reshape(num_imgs, self.num_global_token, self.global_token_dim)
            else:
                single_image_box = [box[:1] for box in image_box]
                image_query, patch_token = self.get_clip_image_feat_multi_cls_token(img_no_normalize, single_image_box, img_metas)
            image_box = torch.stack(image_box)
        ###################################################################
        # end global query
        ###################################################################

        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape

        if not self.with_attn_mask: # remove attn mask for LSJ
            for i in range(len(img_metas)):
                input_img_h, input_img_w = img_metas[i]['batch_input_shape']
                img_metas[i]['img_shape'] = [input_img_h, input_img_w, 3]

        x = self.extract_feat(img, img_metas)

        losses = dict()
        def upd_loss(losses, idx, weight=1):
            new_losses = dict()
            for k,v in losses.items():
                new_k = '{}{}'.format(k,idx)
                if isinstance(v,list) or isinstance(v,tuple):
                    new_losses[new_k] = [i*weight for i in v]
                else:new_losses[new_k] = v*weight
            return new_losses

        # DETR encoder and decoder forward
        if self.with_query_head:
            bbox_losses, x = self.query_head.forward_train(x, image_box, image_query, patch_token, img_metas, gt_bboxes,
                                                          gt_labels, gt_bboxes_ignore)
            losses.update(bbox_losses)
            
        if self.use_co_head:
            # RPN forward and loss
            if self.with_rpn:
                proposal_cfg = self.train_cfg[self.head_idx].get('rpn_proposal',
                                                self.test_cfg[self.head_idx].rpn)
                rpn_losses, proposal_list = self.rpn_head.forward_train(
                    x,
                    img_metas,
                    gt_bboxes,
                    gt_labels=None,
                    gt_bboxes_ignore=gt_bboxes_ignore,
                    proposal_cfg=proposal_cfg,
                    **kwargs)
                losses.update(rpn_losses)
            else:
                proposal_list = proposals

            positive_coords = []
            for i in range(len(self.roi_head)):
                roi_losses = self.roi_head[i].forward_train(x, img_metas, proposal_list,
                                                        gt_bboxes, gt_labels,
                                                        gt_bboxes_ignore, gt_masks,
                                                        **kwargs)
                if self.with_pos_coord:
                    positive_coords.append(roi_losses.pop('pos_coords'))
                else: 
                    if 'pos_coords' in roi_losses.keys():
                        tmp = roi_losses.pop('pos_coords')     
                roi_losses = upd_loss(roi_losses, idx=i)
                losses.update(roi_losses)
                
            for i in range(len(self.bbox_head)):
                bbox_losses = self.bbox_head[i].forward_train(x, img_metas, gt_bboxes,
                                                            gt_labels, gt_bboxes_ignore)
                if self.with_pos_coord:
                    pos_coords = bbox_losses.pop('pos_coords')
                    positive_coords.append(pos_coords)
                else:
                    if 'pos_coords' in bbox_losses.keys():
                        tmp = bbox_losses.pop('pos_coords')          
                bbox_losses = upd_loss(bbox_losses, idx=i+len(self.roi_head))
                losses.update(bbox_losses)

            if self.with_pos_coord and len(positive_coords)>0:
                for i in range(len(positive_coords)):
                    bbox_losses = self.query_head.forward_train_aux(x, image_box, image_query, patch_token, img_metas, gt_bboxes,
                                                                gt_labels, gt_bboxes_ignore, positive_coords[i], i)
                    bbox_losses = upd_loss(bbox_losses, idx=i)
                    losses.update(bbox_losses)                    

        return losses


    def simple_test_roi_head(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        if not self.with_attn_mask: # remove attn mask for LSJ
            for i in range(len(img_metas)):
                input_img_h, input_img_w = img_metas[i]['batch_input_shape']
                img_metas[i]['img_shape'] = [input_img_h, input_img_w, 3]

        x = self.extract_feat(img, img_metas)
        if self.with_query_head:
            results = self.query_head.forward(x, img_metas)
            x = results[-1]
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head[self.eval_index].simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def simple_test_query_head(self, img, image_box, image_query, patch_token, img_metas, proposals=None, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        index = 0
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        if not self.with_attn_mask: # remove attn mask for LSJ
            for i in range(len(img_metas)):
                input_img_h, input_img_w = img_metas[i]['batch_input_shape']
                img_metas[i]['img_shape'] = [input_img_h, input_img_w, 3]

        x = self.extract_feat(img, img_metas)
        results_list = self.query_head.simple_test(
            x, image_box, image_query, patch_token, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.query_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def simple_test_bbox_head(self, img, img_metas, proposals=None, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        if not self.with_attn_mask: # remove attn mask for LSJ
            for i in range(len(img_metas)):
                input_img_h, input_img_w = img_metas[i]['batch_input_shape']
                img_metas[i]['img_shape'] = [input_img_h, input_img_w, 3]

        x = self.extract_feat(img, img_metas)
        if self.with_query_head:
            results = self.query_head.forward(x, img_metas)
            x = results[-1]
        results_list = self.bbox_head[self.eval_index].simple_test(
            x, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head[self.eval_index].num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        img_no_normalize = img.clone().detach()
        img_no_normalize = img_no_normalize * torch.tensor([58.395, 57.12, 57.375], device=img_no_normalize.device).reshape(1, 3, 1, 1) \
                        + torch.tensor([123.675, 116.28, 103.53], device=img_no_normalize.device).reshape(1, 3, 1, 1)
        ###################################################################
        # begin global query
        ###################################################################
        image_box = None
        image_query = None
        patch_token = None
        num_imgs = len(img_metas)
        if self.use_global_token:
            image_box = self.generate_image_box(img_metas, img.dtype, img.device)
            if self.num_global_token == 1:
                image_query = self.get_image_feat(img_no_normalize, image_box, img_metas).reshape(num_imgs, self.num_global_token, self.global_token_dim)
            else:
                single_image_box = [box[:1] for box in image_box]
                image_query, patch_token = self.get_clip_image_feat_multi_cls_token(img_no_normalize, single_image_box, img_metas)
            image_box = torch.stack(image_box)
        ###################################################################
        # end global query
        ###################################################################
        """Test without augmentation."""
        assert self.eval_module in ['detr', 'one-stage', 'two-stage']
        if self.with_bbox and self.eval_module=='one-stage':
            return self.simple_test_bbox_head(img, img_metas, proposals, rescale)
        if self.with_roi_head and self.eval_module=='two-stage':
            return self.simple_test_roi_head(img, img_metas, proposals, rescale)
        return self.simple_test_query_head(img, image_box, image_query, patch_token, img_metas, proposals, rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.query_head, 'aug_test'), \
            f'{self.query_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.query_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.query_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def onnx_export(self, img, img_metas, with_nms=True):
        """Test function without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        outs = self.query_head.forward_onnx(x, img_metas)[:2]
        # get origin input shape to support onnx dynamic shape

        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]['pad_shape_for_onnx'] = img_shape

        if len(outs) == 2:
            # add dummy score_factor
            outs = (*outs, None)
        # TODO Can we change to `get_bboxes` when `onnx_export` fail
        # TODO support NMS
        # det_bboxes, det_labels = self.query_head.onnx_export(
        #     *outs, img_metas, with_nms=with_nms)
        det_bboxes, det_labels = self.query_head.onnx_export(*outs, img_metas)

        return det_bboxes, det_labels