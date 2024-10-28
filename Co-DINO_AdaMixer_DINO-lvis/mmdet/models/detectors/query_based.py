from ..builder import DETECTORS
from .two_stage import TwoStageDetector
from .sparse_rcnn import SparseRCNN
import torch


@DETECTORS.register_module()
class QueryBased(SparseRCNN):
    '''
    We hack and build our model into Sparse RCNN framework implementation
    in mmdetection.
    '''
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """Forward function of SparseR-CNN in train stage.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (List[Tensor], optional) : Segmentation masks for
                each box. But we don't support it in this architecture.
            proposals (List[Tensor], optional): override rpn proposals with
                custom proposals. Use when `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        assert proposals is None, 'Sparse R-CNN does not support' \
                                  ' external proposals'
        assert gt_masks is None, 'Sparse R-CNN does not instance segmentation'

        img_no_normalize = img.clone().detach()
        img_no_normalize = img_no_normalize * torch.tensor([58.395, 57.12, 57.375], device=img_no_normalize.device).reshape(1, 3, 1, 1) \
                        + torch.tensor([123.675, 116.28, 103.53], device=img_no_normalize.device).reshape(1, 3, 1, 1)
        
        x = self.extract_feat(img)
        proposal_boxes, proposal_features, imgs_whwh = \
            self.rpn_head.forward_train(x, img_metas)
        roi_losses = self.roi_head.forward_train(
            x,
            img_no_normalize,
            proposal_boxes,
            proposal_features,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore=gt_bboxes_ignore,
            gt_masks=gt_masks,
            imgs_whwh=imgs_whwh)
        return roi_losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        img_no_normalize = img.clone().detach()
        img_no_normalize = img_no_normalize * torch.tensor([58.395, 57.12, 57.375], device=img_no_normalize.device).reshape(1, 3, 1, 1) \
                        + torch.tensor([123.675, 116.28, 103.53], device=img_no_normalize.device).reshape(1, 3, 1, 1)
        
        x = self.extract_feat(img)
        proposal_boxes, proposal_features, imgs_whwh = \
            self.rpn_head.simple_test_rpn(x, img_metas)
        bbox_results = self.roi_head.simple_test(
            x,
            img_no_normalize,
            proposal_boxes,
            proposal_features,
            img_metas,
            imgs_whwh=imgs_whwh,
            rescale=rescale)
        return bbox_results
