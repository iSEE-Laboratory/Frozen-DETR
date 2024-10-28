import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from mmdet.core import bbox2result, bbox2roi, bbox_xyxy_to_cxcywh
from mmdet.core.bbox.samplers import PseudoSampler
from ..builder import HEADS, build_roi_extractor, build_loss
from .cascade_roi_head import CascadeRoIHead
import myclip
# import open_clip
import dinov2
import deit
import mae
import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt

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



@HEADS.register_module()
class AdaMixerDecoder(CascadeRoIHead):
    _DEBUG = -1

    def __init__(self,
                 num_stages=6,
                 stage_loss_weights=(1, 1, 1, 1, 1, 1),
                 content_dim=256,
                 featmap_strides=[4, 8, 16, 32],
                 use_global_token=False,
                 foundation_model_type='openai_clip',
                 foundation_model_path=None,
                 global_token_dim=768,
                 num_global_token=1,
                 use_dynamic_image_token=False,
                 image_level_distill=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        assert bbox_head is not None
        assert len(stage_loss_weights) == num_stages
        self.featmap_strides = featmap_strides
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        self.content_dim = content_dim
        device = "cuda" if torch.cuda.is_available() else "cpu"
        super(AdaMixerDecoder, self).__init__(
            num_stages,
            stage_loss_weights,
            bbox_roi_extractor=dict(
                # This does not mean that our method need RoIAlign. We put this
                # as a placeholder to satisfy the argument for the parent class
                # 'CascadeRoIHead'.
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=2),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        # train_cfg would be None when run the test.py
        if train_cfg is not None:
            for stage in range(num_stages):
                assert isinstance(self.bbox_sampler[stage], PseudoSampler)
        self.use_global_token = use_global_token
        self.use_dynamic_image_token = use_dynamic_image_token
        self.global_token_dim = global_token_dim
        self.num_global_token = num_global_token
        self.foundation_model_type = foundation_model_type
        self.use_image_level_distill = True if image_level_distill is not None else False # dict(type='CrossEntropyLoss', loss_weight=0.2)
        if self.use_global_token:
            self.attn_weight_roi = build_roi_extractor(
                dict(
                    type='SingleRoIExtractor',
                    roi_layer=dict(type='RoIAlign', output_size=24, sampling_ratio=2),
                    out_channels=1,
                    featmap_strides=[1]),)
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
            elif self.foundation_model_type == 'dinov2':
                self.foundation_model = dinov2.vit_large()
                checkpoint = torch.load(foundation_model_path, map_location='cpu')
                print("Load pre-trained checkpoint from: %s" % (foundation_model_path))
                # load pre-trained model
                msg = self.foundation_model.load_state_dict(checkpoint, strict=False)
                print(msg)
                self.foundation_model.float().eval()
                self.preprocess = dinov2.transform(336)
            elif self.foundation_model_type == 'deit':
                self.foundation_model = deit.deit_large_patch16_LS(True, 336, model_path=foundation_model_path)
                self.foundation_model.float().eval()
                self.preprocess = deit.transform(336)
            elif self.foundation_model_type == 'mae':
                self.foundation_model = mae.vit_large_patch16(global_pool=True, img_size=336, pretrained=True, model_path=foundation_model_path)
                self.foundation_model.float().eval()
                self.preprocess = deit.transform(336)
            elif self.foundation_model_type == 'beit':
                from beit3 import beit3_large_patch16_224_imageclassification
                self.foundation_model = beit3_large_patch16_224_imageclassification(img_size=336, ckpt_path=foundation_model_path, pretrained=True)
                self.foundation_model.float().eval()
                self.preprocess = deit.transform(336)
            elif self.foundation_model_type == 'self':
                self.img_distill_roi = build_roi_extractor(
                    dict(
                        type='SingleRoIExtractor',
                        roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
                        out_channels=content_dim,
                        featmap_strides=[32]),)
            else:
                raise NotImplementedError
            if self.foundation_model_type != 'self':
                for child in self.foundation_model.children():
                    for param in child.parameters():
                        param.requires_grad = False
            
        self.image_id = 0
    
    def draw_boxes(self, stage, image, boxes):
        image = image.to(torch.uint8).cpu()
        H, W = image.shape[-2:]
        boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=W)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=H)
        img = image.permute(1, 2, 0).cpu().numpy()
        img = Image.fromarray(img)
        img.save('%d_%d_ori_img.jpeg'%(self.image_id, stage))
        img = torchvision.utils.draw_bounding_boxes(image, boxes, width=3)
        # img = torchvision.utils.draw_bounding_boxes(img, gt_bboxes, colors='red', width=6)
        img = img.permute(1, 2, 0).cpu().numpy()
        img = Image.fromarray(img)
        img.save('%d_%d_all_box.jpeg'%(self.image_id, stage))

        self.image_id += 1

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
        features, _ = self.foundation_model.forward_multi_cls_token(preprocessed, self.attn_mask, self.num_global_token)
        features = torch.cat([features[:, self.num_global_token-1:], features[:, :self.num_global_token-1]], dim=1)
        
        return features
    

    def _bbox_forward(self, stage, img_feat, query_xyzr, query_content, img_metas, image_box, image_query):
        num_imgs = len(img_metas)
        bbox_head = self.bbox_head[stage]

        cls_score, delta_xyzr, query_content = bbox_head(img_feat, query_xyzr, query_content, featmap_strides=self.featmap_strides
                                                         , image_box=image_box, global_token=image_query)

        query_xyzr, decoded_bboxes = self.bbox_head[stage].refine_xyzr(
            query_xyzr,
            delta_xyzr)

        bboxes_list = [bboxes for bboxes in decoded_bboxes]

        bbox_results = dict(
            cls_score=cls_score,
            query_xyzr=query_xyzr,
            decode_bbox_pred=decoded_bboxes,
            query_content=query_content,
            detach_cls_score_list=[
                cls_score[i].detach() for i in range(num_imgs)
            ],
            detach_bboxes_list=[item.detach() for item in bboxes_list],
            bboxes_list=bboxes_list,
        )
        return bbox_results
    
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
    


    def forward_train(self,
                      x,
                      img_no_normalize,
                      query_xyzr,
                      query_content,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      imgs_whwh=None,
                      gt_masks=None):

        num_imgs = len(img_metas)
        num_queries = query_xyzr.size(1)
        imgs_whwh = imgs_whwh.repeat(1, num_queries, 1)
        all_stage_bbox_results = []
        all_stage_loss = {}
        image_box = None
        image_query = None
        image_box_stack = None
        if self.use_global_token:
            image_box = self.generate_image_box(img_metas, gt_bboxes[0].dtype, gt_bboxes[0].device)
            if self.foundation_model_type == 'self':
                image_box_roi = bbox2roi(image_box)
                backbone_image_features = self.img_distill_roi([x[3]], image_box_roi).reshape(len(image_box_roi), self.content_dim, 49)
                image_query = torch.mean(backbone_image_features, dim=-1)
                image_query = image_query.reshape(num_imgs, self.num_global_token, self.content_dim)
            else:
                if self.num_global_token == 1:
                    image_query = self.get_image_feat(img_no_normalize, image_box, img_metas).reshape(num_imgs, self.num_global_token, self.global_token_dim)
                else:
                    single_image_box = [box[:1] for box in image_box]
                    image_query = self.get_clip_image_feat_multi_cls_token(img_no_normalize, single_image_box, img_metas).reshape(num_imgs, self.num_global_token, self.global_token_dim)
            image_box_stack = torch.stack(image_box)

        for stage in range(self.num_stages):
            bbox_results = self._bbox_forward(stage, x, query_xyzr, query_content,
                                              img_metas, image_box_stack, image_query)
            all_stage_bbox_results.append(bbox_results)
            if gt_bboxes_ignore is None:
                # TODO support ignore
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            cls_pred_list = bbox_results['detach_cls_score_list']
            bboxes_list = bbox_results['detach_bboxes_list']

            query_xyzr = bbox_results['query_xyzr'].detach()
            query_content = bbox_results['query_content']

            if self.stage_loss_weights[stage] <= 0:
                continue

            for i in range(num_imgs):
                normalize_bbox_ccwh = bbox_xyxy_to_cxcywh(bboxes_list[i] /
                                                          imgs_whwh[i])
                assign_result = self.bbox_assigner[stage].assign(
                    normalize_bbox_ccwh, cls_pred_list[i], gt_bboxes[i],
                    gt_labels[i], img_metas[i])
                sampling_result = self.bbox_sampler[stage].sample(
                    assign_result, bboxes_list[i], gt_bboxes[i])
                sampling_results.append(sampling_result)
            bbox_targets = self.bbox_head[stage].get_targets(
                sampling_results, gt_bboxes, gt_labels, self.train_cfg[stage],
                True)

            cls_score = bbox_results['cls_score']
            decode_bbox_pred = bbox_results['decode_bbox_pred']

            single_stage_loss = self.bbox_head[stage].loss(
                cls_score.view(-1, cls_score.size(-1)),
                decode_bbox_pred.view(-1, 4),
                *bbox_targets,
                imgs_whwh=imgs_whwh)
            for key, value in single_stage_loss.items():
                all_stage_loss[f'stage{stage}_{key}'] = value * \
                    self.stage_loss_weights[stage]

        return all_stage_loss

    def simple_test(self,
                    x,
                    img_no_normalize,
                    query_xyzr,
                    query_content,
                    img_metas,
                    imgs_whwh,
                    rescale=False):
        assert self.with_bbox, 'Bbox head must be implemented.'

        num_imgs = len(img_metas)
        image_box = None
        image_query = None
        image_box_stack = None
        if self.use_global_token:
            image_box = self.generate_image_box(img_metas, query_xyzr.dtype, query_xyzr.device)
            if self.foundation_model_type == 'self':
                image_box_roi = bbox2roi(image_box)
                backbone_image_features = self.img_distill_roi([x[3]], image_box_roi).reshape(len(image_box_roi), self.content_dim, 49)
                image_query = torch.mean(backbone_image_features, dim=-1)
                image_query = image_query.reshape(num_imgs, self.num_global_token, self.content_dim)
            else:
                if self.num_global_token == 1:
                    image_query = self.get_image_feat(img_no_normalize, image_box, img_metas).reshape(num_imgs, self.num_global_token, self.global_token_dim)
                else:
                    single_image_box = [box[:1] for box in image_box]
                    image_query = self.get_clip_image_feat_multi_cls_token(img_no_normalize, single_image_box, img_metas).reshape(num_imgs, self.num_global_token, self.global_token_dim)
            image_box_stack = torch.stack(image_box)

        for stage in range(self.num_stages):    
            bbox_results = self._bbox_forward(stage, x, query_xyzr, query_content,
                                              img_metas, image_box_stack, image_query)
            query_content = bbox_results['query_content']
            cls_score = bbox_results['cls_score']
            bboxes_list = bbox_results['detach_bboxes_list']
            query_xyzr = bbox_results['query_xyzr']
            cls_pred_list = bbox_results['detach_cls_score_list']
            # for i in range(num_imgs):
            #     if self.image_id < 60:
            #         self.draw_boxes(stage, img_no_normalize[i], bboxes_list[i])

        num_classes = self.bbox_head[-1].num_classes
        det_bboxes = []
        det_labels = []

        if self.bbox_head[-1].loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
        else:
            cls_score = cls_score.softmax(-1)[..., :-1]

        for img_id in range(num_imgs):
            cls_score_per_img = cls_score[img_id]
            scores_per_img, topk_indices = cls_score_per_img.flatten(
                0, 1).topk(
                    self.test_cfg.max_per_img, sorted=False)
            labels_per_img = topk_indices % num_classes
            bbox_pred_per_img = bboxes_list[img_id][topk_indices //
                                                    num_classes]
            if rescale:
                scale_factor = img_metas[img_id]['scale_factor']
                bbox_pred_per_img /= bbox_pred_per_img.new_tensor(scale_factor)
            det_bboxes.append(
                torch.cat([bbox_pred_per_img, scores_per_img[:, None]], dim=1))
            det_labels.append(labels_per_img)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i], num_classes)
            for i in range(num_imgs)
        ]

        return bbox_results

    def aug_test(self, x, bboxes_list, img_metas, rescale=False):
        raise NotImplementedError()

    def forward_dummy(self, x,
                      query_xyzr,
                      query_content,
                      img_metas):
        """Dummy forward function when do the flops computing."""
        all_stage_bbox_results = []

        num_imgs = len(img_metas)
        if self.with_bbox:
            for stage in range(self.num_stages):
                bbox_results = self._bbox_forward(stage, x,
                                                  query_xyzr,
                                                  query_content,
                                                  img_metas)
                all_stage_bbox_results.append(bbox_results)
                query_content = bbox_results['query_content']
                query_xyzr = bbox_results['query_xyzr']
        return all_stage_bbox_results
