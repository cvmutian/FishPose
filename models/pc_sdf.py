import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
from typing import Dict

class PCSDFModule(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, num_keypoints: int,
        deformable_groups: int = 8, kernel_size: int = 3
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_keypoints = num_keypoints
        self.deformable_groups = deformable_groups
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1)
        )

        offset_channels = deformable_groups * 2 * kernel_size * kernel_size
        mask_channels = deformable_groups * 1 * kernel_size * kernel_size
        
        predictor_in_channels = in_channels + num_keypoints + 1
        
        self.offset_mask_predictor = nn.Sequential(
            nn.Conv2d(predictor_in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, offset_channels + mask_channels, kernel_size=1)
        )
        self.offset_mask_predictor[-1].weight.data.zero_()
        self.offset_mask_predictor[-1].bias.data.zero_()

        self.deform_conv = DeformConv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            padding=padding, groups=deformable_groups, bias=False
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.identity_proj = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, feature_map: torch.Tensor, coarse_heatmap: torch.Tensor) -> Dict[str, torch.Tensor]:
        uncertainty_map = self.uncertainty_head(feature_map)

        if coarse_heatmap.shape[2:] != feature_map.shape[2:]:
            heatmap_resized = F.interpolate(
                coarse_heatmap, size=feature_map.shape[2:],
                mode='bilinear', align_corners=False
            )
        else:
            heatmap_resized = coarse_heatmap

        predictor_input = torch.cat([feature_map, heatmap_resized, uncertainty_map], dim=1)
        
        offset_mask = self.offset_mask_predictor(predictor_input)
        
        split_size = self.deformable_groups * self.kernel_size * self.kernel_size
        offset, mask = torch.split(
            offset_mask,
            [2 * split_size, 1 * split_size],
            dim=1
        )
        
        mask = torch.sigmoid(mask)
        
        deformed_features_raw = self.deform_conv(feature_map, offset, mask)
        deformed_features_activated = self.relu(self.norm(deformed_features_raw))
        
        gate_input = torch.cat([feature_map, deformed_features_activated], dim=1)
        gate = self.gate_conv(gate_input)
        
        identity_features = self.identity_proj(feature_map)
        
        fused_features = gate * deformed_features_activated + (1 - gate) * identity_features
        
        return {"fused_features": fused_features, "offsets": offset, "uncertainty_map": uncertainty_map}
