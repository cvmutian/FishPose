import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


class GIoULoss(nn.Module):

    def __init__(self, reduction='mean'):
        super(GIoULoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):

        return torch.tensor(0.0, device=pred.device)


class OKSLoss(nn.Module):

    def __init__(self, sigmas: torch.Tensor = None, reduction: str = 'mean'):
        super(OKSLoss, self).__init__()
        if sigmas is None:

            sigmas = torch.tensor([0.025, 0.026, 0.026, 0.035, 0.035,
                                   0.079, 0.079, 0.072, 0.072, 0.062, 0.062])
        self.register_buffer('sigmas', sigmas)
        self.reduction = reduction

    def forward(
        self,
        pred_keypoints: torch.Tensor,
        gt_keypoints: torch.Tensor,
        gt_bbox: torch.Tensor = None
    ) -> torch.Tensor:

        batch_size, num_kpts, _ = pred_keypoints.shape


        pred_xy = pred_keypoints
        gt_xy = gt_keypoints[:, :, :2]
        visibility = gt_keypoints[:, :, 2] > 0


        if gt_bbox is not None:
            scale = gt_bbox[:, 2] * gt_bbox[:, 3]
            scale = scale.unsqueeze(1).unsqueeze(2)
        else:

            scale = torch.ones(batch_size, 1, 1, device=pred_keypoints.device)


        variances = (self.sigmas * 2).pow(2).view(1, -1, 1)
        dx = pred_xy[:, :, 0] - gt_xy[:, :, 0]
        dy = pred_xy[:, :, 1] - gt_xy[:, :, 1]
        dist_sq = (dx.pow(2) + dy.pow(2)) / (variances * scale + 1e-8)


        oks_per_kpt = torch.exp(-dist_sq / 2.0)


        oks_per_kpt = oks_per_kpt * visibility.float()


        num_visible = visibility.sum(dim=1, keepdim=True).float()
        oks_per_sample = oks_per_kpt.sum(dim=1) / (num_visible + 1e-8)


        loss = 1.0 - oks_per_sample

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DeformationConsistencyLoss(nn.Module):

    def __init__(self, consistency_type: str = 'offset', smoothness_weight: float = 0.1):
        super(DeformationConsistencyLoss, self).__init__()
        self.consistency_type = consistency_type
        self.smoothness_weight = smoothness_weight

    def forward(
        self,
        offsets: torch.Tensor = None,
        keypoints: torch.Tensor = None,
        visibility: torch.Tensor = None
    ) -> torch.Tensor:
        if self.consistency_type == 'offset' and offsets is not None:
            return self._offset_consistency_loss(offsets)
        elif self.consistency_type == 'keypoint' and keypoints is not None:
            return self._keypoint_consistency_loss(keypoints, visibility)
        else:

            device = offsets.device if offsets is not None else (keypoints.device if keypoints is not None else 'cpu')
            return torch.tensor(0.0, device=device)

    def _offset_consistency_loss(self, offsets: torch.Tensor) -> torch.Tensor:



        grad_x = offsets[:, :, :, 1:] - offsets[:, :, :, :-1]

        grad_y = offsets[:, :, 1:, :] - offsets[:, :, :-1, :]


        loss_x = grad_x.pow(2).mean()
        loss_y = grad_y.pow(2).mean()

        return self.smoothness_weight * (loss_x + loss_y)

    def _keypoint_consistency_loss(
        self,
        keypoints: torch.Tensor,
        visibility: torch.Tensor = None
    ) -> torch.Tensor:

        batch_size, num_kpts, _ = keypoints.shape

        if num_kpts < 2:
            return torch.tensor(0.0, device=keypoints.device)


        kpt_diff = keypoints[:, 1:, :] - keypoints[:, :-1, :]
        distances = kpt_diff.norm(dim=2)


        if num_kpts >= 3:
            second_diff = distances[:, 1:] - distances[:, :-1]
            consistency_loss = second_diff.pow(2).mean()
        else:
            consistency_loss = distances.pow(2).mean()


        if visibility is not None:

            visible_mask = visibility[:, :-1] & visibility[:, 1:]
            if visible_mask.any():
                consistency_loss = consistency_loss * visible_mask.float().mean()

        return self.smoothness_weight * consistency_loss


class FishPoseLoss(nn.Module):

    def __init__(
        self,
        heatmap_loss_weight: float = 1.0,
        offset_loss_weight: float = 1.0,
        oks_loss_weight: float = 1.0,
        deformation_loss_weight: float = 1.0,
        giou_loss_weight: float = 1.0,
        num_keypoints: int = 11,
        oks_sigmas: torch.Tensor = None
    ):
        super().__init__()
        self.loss_weights = {
            'heatmap': heatmap_loss_weight,
            'offset': offset_loss_weight,
            'oks': oks_loss_weight,
            'deformation': deformation_loss_weight,
            'giou': giou_loss_weight
        }
        self.num_keypoints = num_keypoints
        self.heatmap_loss_fn = FocalLoss()
        self.offset_loss_fn = nn.L1Loss(reduction='sum')
        self.oks_loss_fn = OKSLoss(sigmas=oks_sigmas, reduction='mean')
        self.deformation_loss_fn = DeformationConsistencyLoss(
            consistency_type='offset',
            smoothness_weight=0.1
        )
        self.giou_loss_fn = GIoULoss(reduction='mean')

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:


        pred_heatmap = predictions['pose_heatmap']
        pred_offsets = predictions['offsets']


        gt_heatmap = targets['heatmap_target']
        gt_offsets = targets['offset_target']
        kpt_mask = targets['kpt_mask']


        loss_heatmap = self.heatmap_loss_fn(pred_heatmap, gt_heatmap)


        pred_offsets_reshaped = pred_offsets.view(
            pred_offsets.shape[0], self.num_keypoints, 2, pred_offsets.shape[2], pred_offsets.shape[3]
        )



        gt_offsets_y = gt_offsets[:, :self.num_keypoints, :, :]
        gt_offsets_x = gt_offsets[:, self.num_keypoints:, :, :]
        gt_offsets_reshaped = torch.stack([gt_offsets_y, gt_offsets_x], dim=2)


        kpt_mask_y = kpt_mask[:, :self.num_keypoints, :, :]
        kpt_mask_expanded = kpt_mask_y.unsqueeze(2).expand_as(pred_offsets_reshaped)


        loss_offset_unnormalized = self.offset_loss_fn(
            pred_offsets_reshaped[kpt_mask_expanded],
            gt_offsets_reshaped[kpt_mask_expanded]
        )


        num_visible_kpts = kpt_mask_expanded.sum()
        if num_visible_kpts > 0:
            offset_loss = loss_offset_unnormalized / num_visible_kpts
        else:
            offset_loss = torch.tensor(0.0, device=pred_heatmap.device)


        loss_oks = torch.tensor(0.0, device=pred_heatmap.device)
        if self.loss_weights['oks'] > 0 and 'pred_keypoints' in predictions:


            pred_kpts = predictions.get('pred_keypoints', None)
            gt_kpts = targets.get('keypoints', None)
            gt_bbox = targets.get('bbox', None)
            if pred_kpts is not None and gt_kpts is not None:
                loss_oks = self.oks_loss_fn(pred_kpts, gt_kpts, gt_bbox)


        loss_deformation = torch.tensor(0.0, device=pred_heatmap.device)
        if self.loss_weights['deformation'] > 0:

            loss_deformation = self.deformation_loss_fn(offsets=pred_offsets)


        loss_giou = torch.tensor(0.0, device=pred_heatmap.device)
        if self.loss_weights['giou'] > 0 and 'det_preds' in predictions:


            pred_bbox = predictions.get('det_preds', None)
            gt_bbox = targets.get('bbox', None)
            if pred_bbox is not None and gt_bbox is not None:
                loss_giou = self.giou_loss_fn(pred_bbox, gt_bbox)


        total_loss = (
            self.loss_weights['heatmap'] * loss_heatmap +
            self.loss_weights['offset'] * offset_loss +
            self.loss_weights['oks'] * loss_oks +
            self.loss_weights['deformation'] * loss_deformation +
            self.loss_weights['giou'] * loss_giou
        )

        return {
            'total_loss': total_loss,
            'loss_heatmap': loss_heatmap,
            'loss_offset': offset_loss,
            'loss_oks': loss_oks,
            'loss_deformation': loss_deformation,
            'loss_giou': loss_giou
        }
