import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class LightweightHead(nn.Module):
    def __init__(self, in_channels: int, num_keypoints: int):
        super().__init__()
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_keypoints, kernel_size=1, stride=1, padding=0)
        )
        
        self.coords_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_keypoints * 2, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x: torch.Tensor):
        heatmaps = self.head(x)
        coords_offset = self.coords_head(x)
        return {'heatmaps': heatmaps, 'coords_offset': coords_offset}


class DetectionHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 1, feat_channels: int = 256, stacked_convs: int = 4):
        super().__init__()
        self.num_classes = num_classes
        
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(stacked_convs):
            chn = in_channels if i == 0 else feat_channels
            self.cls_convs.append(
                nn.Sequential(
                    nn.Conv2d(chn, feat_channels, 3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(feat_channels),
                    nn.ReLU(inplace=True)
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    nn.Conv2d(chn, feat_channels, 3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(feat_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        self.det_cls_pred = nn.Conv2d(feat_channels, self.num_classes, 3, padding=1)
        self.det_reg_pred = nn.Conv2d(feat_channels, 4, 3, padding=1)
        self.det_cen_pred = nn.Conv2d(feat_channels, 1, 3, padding=1)

    def forward(self, x: torch.Tensor):
        cls_feat = x
        reg_feat = x

        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)

        cls_score = self.det_cls_pred(cls_feat)
        bbox_pred = F.relu(self.det_reg_pred(reg_feat))
        centerness = self.det_cen_pred(reg_feat)
        
        return {
            'cls_score': cls_score, 
            'bbox_pred': bbox_pred, 
            'centerness': centerness
        }


class RTMOHead(nn.Module):
    """RTMO-style head for 1D coordinate classification."""
    def __init__(
        self,
        in_channels: int,
        num_keypoints: int,
        img_size: List[int], # Add img_size as a required parameter
        feat_channels: int = 256
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.img_size = img_size
        img_h, img_w = self.img_size

        self.heatmap_x_layers = self._make_head_layers(in_channels, feat_channels)
        self.heatmap_y_layers = self._make_head_layers(in_channels, feat_channels)

        # Dynamically set output channels based on image size
        self.heatmap_x_pred = nn.Conv2d(feat_channels, num_keypoints * img_w, 1)
        self.heatmap_y_pred = nn.Conv2d(feat_channels, num_keypoints * img_h, 1)

    def _make_head_layers(self, in_channels: int, feat_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor):
        B, _, H, W = x.shape
        feat_x = self.heatmap_x_layers(x)
        pred_x_flat = self.heatmap_x_pred(feat_x).mean(dim=2) # Global Average Pooling on H, shape (B, K*Wp_out)
        pred_x = pred_x_flat.view(B, self.num_keypoints, -1) # Reshape to (B, K, Wp_out)

        feat_y = self.heatmap_y_layers(x)
        pred_y_flat = self.heatmap_y_pred(feat_y).mean(dim=3) # Global Average Pooling on W, shape (B, K*Hp_out)
        pred_y = pred_y_flat.view(B, self.num_keypoints, -1) # Reshape to (B, K, Hp_out)
        
        return {'heatmap_x': pred_x, 'heatmap_y': pred_y}


class CoarseHead(nn.Module):
    def __init__(self, in_channels: int, num_keypoints: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_keypoints, kernel_size=1, stride=1, padding=0)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)
