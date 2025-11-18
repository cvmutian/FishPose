import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

from models.head import CoarseHead, LightweightHead, DetectionHead, RTMOHead
from models.pc_sdf import PCSDFModule
from models.ssm_pyramid import SSMPyramidBackbone


class FishPoseModel(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        
        self.backbone = SSMPyramidBackbone(**config['model']['backbone']['ssm_pyramid'])

        backbone_out_channels = config['model']['backbone']['ssm_pyramid']['mamba_config']['output_dims']
        fpn_out_channels = config['model']['fpn']['out_channels']

        self.fpn_convs = nn.ModuleList()
        for in_channels in backbone_out_channels:
            self.fpn_convs.append(nn.Conv2d(in_channels, fpn_out_channels, 1))

        self.coarse_head = CoarseHead(**config['model']['coarse_head'])
        self.fusion_module = PCSDFModule(**config['model']['fusion_module'])

        pose_head_type = config['model'].get('pose_head_type', 'heatmap')
        if pose_head_type == 'heatmap':
            self.pose_head = LightweightHead(**config['model']['pose_head'])
        elif pose_head_type == 'rtmo':
            pose_head_cfg = config['model']['pose_head']
            pose_head_cfg['img_size'] = config['data']['img_size']
            self.pose_head = RTMOHead(**pose_head_cfg)
        else:
            raise ValueError(f"Unknown pose_head_type: {pose_head_type}")

        self.det_head = DetectionHead(**config['model']['det_head'])

    def forward(self, x):
        features_p_list = self.backbone(x)
        
        fpn_features = []
        for i, feature in enumerate(features_p_list):
            fpn_features.append(self.fpn_convs[i](feature))

        target_size = fpn_features[0].shape[2:]
        fused_fpn_feature = fpn_features[0]
        for feature in fpn_features[1:]:
            fused_fpn_feature += F.interpolate(
                feature, size=target_size, mode='bilinear', align_corners=False
            )

        coarse_heatmap = self.coarse_head(fpn_features[-1])

        final_feature_map_dict = self.fusion_module(fused_fpn_feature, coarse_heatmap)
        
        final_features = final_feature_map_dict["fused_features"]
        pose_outputs = self.pose_head(final_features)
        det_preds = self.det_head(final_features)
        
        return {
            'pose_heatmap': pose_outputs['heatmaps'],
            'offsets': pose_outputs['coords_offset'],
            'det_preds': det_preds,
            'coarse_heatmap': coarse_heatmap,
            'final_feature_map_dict': final_feature_map_dict
        }
