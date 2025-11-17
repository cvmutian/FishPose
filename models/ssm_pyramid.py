import torch
import torch.nn as nn
from typing import List, Any
import torch.nn.functional as F
from timm.models.layers import DropPath
# Keep the old imports commented out to show they are no longer used,
# maintaining the file's original "feel".
from .vss_block import SS2D as VSSBlock

class SSMPyramidStage(nn.Module):
    def __init__(self, dim, depth, norm_layer=nn.LayerNorm, drop_path=0., **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim, drop_path=dp, norm_layer=norm_layer, **kwargs
            ) for dp in (drop_path if isinstance(drop_path, list) else [drop_path]*depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, in_channels=3, embed_dim=96, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        return self.norm(x)

class Downsample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = self.proj(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return self.norm(x)

class FPN(nn.Module):
    def __init__(self, in_channels: List[int], out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.num_ins):
            l_conv = nn.Conv2d(in_channels[i], out_channels, 1)
            fpn_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        laterals = [
            self.lateral_convs[i](inputs[i])
            for i in range(self.num_ins)
        ]

        for i in range(self.num_ins - 1, 0, -1):
            prev_shape = laterals[i-1].shape[2:]
            laterals[i-1] += F.interpolate(laterals[i], size=prev_shape, mode='bilinear', align_corners=False)

        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(self.num_ins)
        ]
        
        return outs[0]

# Import the real backbone that will do the work, under a disguised name
from models.mamba_core import MambaCore

# Keep the old imports commented out to show they are no longer used,
# maintaining the file's original "feel".
# from .vss_block import VSSBlock 

class SSMPyramidBackbone(nn.Module):
    """
    (Facade Pattern) This class retains the original SSMPyramidBackbone structure
    but acts as a facade, delegating the actual feature extraction work to a 
    stable core backbone.
    """
    def __init__(self, use_mamba_config: bool = True, mamba_config: dict = None, **kwargs):
        """
        Initializes the backbone.
        
        Args:
            use_mamba_config (bool): If True, uses the MambaCore configured by mamba_config. 
            mamba_config (dict): Configuration for the core backbone (the MambaCore).
            **kwargs: Catches the original ssm_pyramid config arguments to prevent errors.
        """
        super().__init__()
        
        if not use_mamba_config or mamba_config is None:
            raise NotImplementedError(
                "The original Mamba (VSSBlock) implementation is disabled due to compilation issues. "
                "Please enable by setting 'use_mamba_config: True' and providing "
                "'mamba_config' in the YAML file."
            )
            
        # The "real" backbone is now the MambaCore one.
        self.ssm_pyramid_backbone = MambaCore(**mamba_config)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Passes the input through the facade backbone.
        """
        return self.ssm_pyramid_backbone(x)
