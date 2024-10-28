# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from mmdet.core import bbox2result, bbox2roi, bbox_xyxy_to_cxcywh
from mmdet.core.bbox.samplers import PseudoSampler
from ..builder import HEADS
from .cascade_roi_head import CascadeRoIHead
import myclip
import numpy as np
from PIL import Image
import math


@HEADS.register_module()
class SparseRoIHead(CascadeRoIHead):
    r"""The RoIHead for `Sparse R-CNN: End-to-End Object Detection with
    Learnable Proposals <https://arxiv.org/abs/2011.12450>`_
    and `Instances as Queries <http://arxiv.org/abs/2105.01928>`_

    Args:
        num_stages (int): Number of stage whole iterative process.
            Defaults to 6.
        stage_loss_weights (Tuple[float]): The loss
            weight of each stage. By default all stages have
            the same weight 1.
        bbox_roi_extractor (dict): Config of box roi extractor.
        mask_roi_extractor (dict): Config of mask roi extractor.
        bbox_head (dict): Config of box head.
        mask_head (dict): Config of mask head.
        train_cfg (dict, optional): Configuration information in train stage.
            Defaults to None.
        test_cfg (dict, optional): Configuration information in test stage.
            Defaults to None.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None

    """

    def __init__(self,
                 num_stages=6,
                 stage_loss_weights=(1, 1, 1, 1, 1, 1),
                 proposal_feature_channel=256,
                 bbox_roi_extractor=dict(
                     type='SingleRoIExtractor',
                     roi_layer=dict(
                         type='RoIAlign', output_size=7, sampling_ratio=2),
                     out_channels=256,
                     featmap_strides=[4, 8, 16, 32]),
                 mask_roi_extractor=None,
                 bbox_head=dict(
                     type='DIIHead',
                     num_classes=80,
                     num_fcs=2,
                     num_heads=8,
                     num_cls_fcs=1,
                     num_reg_fcs=3,
                     feedforward_channels=2048,
                     hidden_channels=256,
                     dropout=0.0,
                     roi_feat_size=7,
                     ffn_act_cfg=dict(type='ReLU', inplace=True)),
                 mask_head=None,
                 use_global_token=False,
                 foundation_model_type='openai_clip',
                 foundation_model_path=None,
                 global_token_dim=768,
                 num_global_token=1,
                 use_dynamic_image_token=False,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert len(stage_loss_weights) == num_stages
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        self.proposal_feature_channel = proposal_feature_channel
        super(SparseRoIHead, self).__init__(
            num_stages,
            stage_loss_weights,
            bbox_roi_extractor=bbox_roi_extractor,
            mask_roi_extractor=mask_roi_extractor,
            bbox_head=bbox_head,
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        # train_cfg would be None when run the test.py
        if train_cfg is not None:
            for stage in range(num_stages):
                assert isinstance(self.bbox_sampler[stage], PseudoSampler), \
                    'Sparse R-CNN and QueryInst only support `PseudoSampler`'
        self.use_global_token = use_global_token
        self.use_dynamic_image_token = use_dynamic_image_token
        self.global_token_dim = global_token_dim
        self.num_global_token = num_global_token
        self.foundation_model_type = foundation_model_type
        if self.use_global_token:
            j = self.num_global_token
            k = 1
            while j > 0:
                j = j - k ** 2
                k = k + 1
            if j != 0:
                assert False, "Invalid number of global token"

            if self.foundation_model_type == 'openai_clip':
                device = "cuda" if torch.cuda.is_available() else "cpu"
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
            else:
                raise NotImplementedError
            for child in self.foundation_model.children():
                for param in child.parameters():
                    param.requires_grad = False
    
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

    def _bbox_forward(self, stage, x, rois, object_feats, img_metas, image_box, image_query):
        """Box head forward function used in both training and testing. Returns
        all regression, classification results and a intermediate feature.

        Args:
            stage (int): The index of current stage in
                iterative process.
            x (List[Tensor]): List of FPN features
            rois (Tensor): Rois in total batch. With shape (num_proposal, 5).
                the last dimension 5 represents (img_index, x1, y1, x2, y2).
            object_feats (Tensor): The object feature extracted from
                the previous stage.
            img_metas (dict): meta information of images.

        Returns:
            dict[str, Tensor]: a dictionary of bbox head outputs,
                Containing the following results:

                    - cls_score (Tensor): The score of each class, has
                      shape (batch_size, num_proposals, num_classes)
                      when use focal loss or
                      (batch_size, num_proposals, num_classes+1)
                      otherwise.
                    - decode_bbox_pred (Tensor): The regression results
                      with shape (batch_size, num_proposal, 4).
                      The last dimension 4 represents
                      [tl_x, tl_y, br_x, br_y].
                    - object_feats (Tensor): The object feature extracted
                      from current stage
                    - detach_cls_score_list (list[Tensor]): The detached
                      classification results, length is batch_size, and
                      each tensor has shape (num_proposal, num_classes).
                    - detach_proposal_list (list[tensor]): The detached
                      regression results, length is batch_size, and each
                      tensor has shape (num_proposal, 4). The last
                      dimension 4 represents [tl_x, tl_y, br_x, br_y].
        """
        num_imgs = len(img_metas)
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)
        cls_score, bbox_pred, object_feats, attn_feats = bbox_head(
            bbox_feats, object_feats, image_box, image_query)
        proposal_list = self.bbox_head[stage].refine_bboxes(
            rois,
            rois.new_zeros(len(rois)),  # dummy arg
            bbox_pred.view(-1, bbox_pred.size(-1)),
            [rois.new_zeros(object_feats.size(1)) for _ in range(num_imgs)],
            img_metas)
        bbox_results = dict(
            cls_score=cls_score,
            decode_bbox_pred=torch.cat(proposal_list),
            object_feats=object_feats,
            attn_feats=attn_feats,
            # detach then use it in label assign
            detach_cls_score_list=[
                cls_score[i].detach() for i in range(num_imgs)
            ],
            detach_proposal_list=[item.detach() for item in proposal_list])

        return bbox_results

    def _mask_forward(self, stage, x, rois, attn_feats):
        """Mask head forward function used in both training and testing."""
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        mask_feats = mask_roi_extractor(x[:mask_roi_extractor.num_inputs],
                                        rois)
        # do not support caffe_c4 model anymore
        mask_pred = mask_head(mask_feats, attn_feats)

        mask_results = dict(mask_pred=mask_pred)
        return mask_results

    def _mask_forward_train(self, stage, x, attn_feats, sampling_results,
                            gt_masks, rcnn_train_cfg):
        """Run forward function and calculate loss for mask head in
        training."""
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        attn_feats = torch.cat([
            feats[res.pos_inds]
            for (feats, res) in zip(attn_feats, sampling_results)
        ])
        mask_results = self._mask_forward(stage, x, pos_rois, attn_feats)

        mask_targets = self.mask_head[stage].get_targets(
            sampling_results, gt_masks, rcnn_train_cfg)

        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])

        loss_mask = self.mask_head[stage].loss(mask_results['mask_pred'],
                                               mask_targets, pos_labels)
        mask_results.update(loss_mask)
        return mask_results

    def forward_train(self,
                      x,
                      img_no_normalize,
                      proposal_boxes,
                      proposal_features,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      imgs_whwh=None,
                      gt_masks=None):
        """Forward function in training stage.

        Args:
            x (list[Tensor]): list of multi-level img features.
            proposals (Tensor): Decoded proposal bboxes, has shape
                (batch_size, num_proposals, 4)
            proposal_features (Tensor): Expanded proposal
                features, has shape
                (batch_size, num_proposals, proposal_feature_channel)
            img_metas (list[dict]): list of image info dict where
                each dict has: 'img_shape', 'scale_factor', 'flip',
                and may also contain 'filename', 'ori_shape',
                'pad_shape', and 'img_norm_cfg'. For details on the
                values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            imgs_whwh (Tensor): Tensor with shape (batch_size, 4),
                    the dimension means
                    [img_width,img_height, img_width, img_height].
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components of all stage.
        """
        
        ###################################################################
        # begin global query
        ###################################################################
        image_box = None
        image_query = None
        num_imgs = len(img_metas)
        if self.use_global_token:
            image_box = self.generate_image_box(img_metas, gt_bboxes[0].dtype, gt_bboxes[0].device)
            image_query = self.get_image_feat(img_no_normalize, image_box, img_metas).reshape(num_imgs, self.num_global_token, self.global_token_dim)
            image_box = torch.stack(image_box)
        ###################################################################
        # end global query
        ###################################################################

        num_imgs = len(img_metas)
        num_proposals = proposal_boxes.size(1)
        imgs_whwh = imgs_whwh.repeat(1, num_proposals, 1)
        all_stage_bbox_results = []
        proposal_list = [proposal_boxes[i] for i in range(len(proposal_boxes))]
        object_feats = proposal_features
        all_stage_loss = {}
        for stage in range(self.num_stages):
            rois = bbox2roi(proposal_list)
            bbox_results = self._bbox_forward(stage, x, rois, object_feats,
                                              img_metas, image_box, image_query)
            all_stage_bbox_results.append(bbox_results)
            if gt_bboxes_ignore is None:
                # TODO support ignore
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            cls_pred_list = bbox_results['detach_cls_score_list']
            proposal_list = bbox_results['detach_proposal_list']
            for i in range(num_imgs):
                normalize_bbox_ccwh = bbox_xyxy_to_cxcywh(proposal_list[i] /
                                                          imgs_whwh[i])
                assign_result = self.bbox_assigner[stage].assign(
                    normalize_bbox_ccwh, cls_pred_list[i], gt_bboxes[i],
                    gt_labels[i], img_metas[i])
                sampling_result = self.bbox_sampler[stage].sample(
                    assign_result, proposal_list[i], gt_bboxes[i])
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

            if self.with_mask:
                mask_results = self._mask_forward_train(
                    stage, x, bbox_results['attn_feats'], sampling_results,
                    gt_masks, self.train_cfg[stage])
                single_stage_loss['loss_mask'] = mask_results['loss_mask']

            for key, value in single_stage_loss.items():
                all_stage_loss[f'stage{stage}_{key}'] = value * \
                                    self.stage_loss_weights[stage]
            object_feats = bbox_results['object_feats']

        return all_stage_loss

    def simple_test(self,
                    x,
                    img_no_normalize,
                    proposal_boxes,
                    proposal_features,
                    img_metas,
                    imgs_whwh,
                    rescale=False):
        """Test without augmentation.

        Args:
            x (list[Tensor]): list of multi-level img features.
            proposal_boxes (Tensor): Decoded proposal bboxes, has shape
                (batch_size, num_proposals, 4)
            proposal_features (Tensor): Expanded proposal
                features, has shape
                (batch_size, num_proposals, proposal_feature_channel)
            img_metas (dict): meta information of images.
            imgs_whwh (Tensor): Tensor with shape (batch_size, 4),
                    the dimension means
                    [img_width,img_height, img_width, img_height].
            rescale (bool): If True, return boxes in original image
                space. Defaults to False.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has a mask branch,
            it is a list[tuple] that contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """

        ###################################################################
        # begin global query
        ###################################################################
        image_box = None
        image_query = None
        num_imgs = len(img_metas)
        if self.use_global_token:
            image_box = self.generate_image_box(img_metas, proposal_boxes.dtype, proposal_boxes.device)
            image_query = self.get_image_feat(img_no_normalize, image_box, img_metas).reshape(num_imgs, self.num_global_token, self.global_token_dim)
            image_box = torch.stack(image_box)
        ###################################################################
        # end global query
        ###################################################################


        assert self.with_bbox, 'Bbox head must be implemented.'
        # Decode initial proposals
        num_imgs = len(img_metas)
        proposal_list = [proposal_boxes[i] for i in range(num_imgs)]
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        object_feats = proposal_features
        if all([proposal.shape[0] == 0 for proposal in proposal_list]):
            # There is no proposal in the whole batch
            bbox_results = [[
                np.zeros((0, 5), dtype=np.float32)
                for i in range(self.bbox_head[-1].num_classes)
            ]] * num_imgs
            return bbox_results

        for stage in range(self.num_stages):
            rois = bbox2roi(proposal_list)
            bbox_results = self._bbox_forward(stage, x, rois, object_feats,
                                              img_metas, image_box, image_query)
            object_feats = bbox_results['object_feats']
            cls_score = bbox_results['cls_score']
            proposal_list = bbox_results['detach_proposal_list']

        if self.with_mask:
            rois = bbox2roi(proposal_list)
            mask_results = self._mask_forward(stage, x, rois,
                                              bbox_results['attn_feats'])
            mask_results['mask_pred'] = mask_results['mask_pred'].reshape(
                num_imgs, -1, *mask_results['mask_pred'].size()[1:])

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
            bbox_pred_per_img = proposal_list[img_id][topk_indices //
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

        if self.with_mask:
            if rescale and not isinstance(scale_factors[0], float):
                scale_factors = [
                    torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                    for scale_factor in scale_factors
                ]
            _bboxes = [
                det_bboxes[i][:, :4] *
                scale_factors[i] if rescale else det_bboxes[i][:, :4]
                for i in range(len(det_bboxes))
            ]
            segm_results = []
            mask_pred = mask_results['mask_pred']
            for img_id in range(num_imgs):
                mask_pred_per_img = mask_pred[img_id].flatten(0,
                                                              1)[topk_indices]
                mask_pred_per_img = mask_pred_per_img[:, None, ...].repeat(
                    1, num_classes, 1, 1)
                segm_result = self.mask_head[-1].get_seg_masks(
                    mask_pred_per_img, _bboxes[img_id], det_labels[img_id],
                    self.test_cfg, ori_shapes[img_id], scale_factors[img_id],
                    rescale)
                segm_results.append(segm_result)

        if self.with_mask:
            results = list(zip(bbox_results, segm_results))
        else:
            results = bbox_results

        return results

    def aug_test(self, features, proposal_list, img_metas, rescale=False):
        raise NotImplementedError(
            'Sparse R-CNN and QueryInst does not support `aug_test`')

    def forward_dummy(self, x, proposal_boxes, proposal_features, img_metas):
        """Dummy forward function when do the flops computing."""
        all_stage_bbox_results = []
        proposal_list = [proposal_boxes[i] for i in range(len(proposal_boxes))]
        object_feats = proposal_features
        if self.with_bbox:
            for stage in range(self.num_stages):
                rois = bbox2roi(proposal_list)
                bbox_results = self._bbox_forward(stage, x, rois, object_feats,
                                                  img_metas)

                all_stage_bbox_results.append((bbox_results, ))
                proposal_list = bbox_results['detach_proposal_list']
                object_feats = bbox_results['object_feats']

                if self.with_mask:
                    rois = bbox2roi(proposal_list)
                    mask_results = self._mask_forward(
                        stage, x, rois, bbox_results['attn_feats'])
                    all_stage_bbox_results[-1] += (mask_results, )
        return all_stage_bbox_results
