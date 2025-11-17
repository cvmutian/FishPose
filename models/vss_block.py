import math
from functools import partial
import json
import os
from collections import namedtuple
from typing import Any
import torch
import torch.nn as nn
from timm.models.layers import DropPath
from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
except ImportError:
    selective_scan_fn, mamba_inner_fn = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class VSSBlock_original(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0.0,
        ssm_simple=False,
        norm_layer: nn.Module = nn.LayerNorm,
        drop_path: float = 0.,
        **kwargs,
    ):
        super().__init__()
        self.ssm_d_state = ssm_d_state
        self.ssm_ratio = ssm_ratio
        self.ssm_dt_rank = math.ceil(hidden_dim / 16) if ssm_dt_rank == "auto" else ssm_dt_rank
        self.ssm_simple = ssm_simple

        self.ln = norm_layer(hidden_dim)
        
        d_ssm = int(hidden_dim * ssm_ratio)
        self.in_proj = nn.Linear(hidden_dim, d_ssm * 2, bias=False)
        self.out_proj = nn.Linear(d_ssm, hidden_dim, bias=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.act = ssm_act_layer()
        self.conv2d = nn.Conv2d(
            in_channels=d_ssm, out_channels=d_ssm,
            kernel_size=ssm_conv,
            padding=(ssm_conv - 1) // 2,
            bias=ssm_conv_bias,
            groups=d_ssm,
        )

        self.dt_proj = nn.Linear(self.ssm_dt_rank, d_ssm, bias=True)
        self.A_log = nn.Parameter(torch.clamp(torch.randn(d_ssm, self.ssm_d_state), -3, 3))
        self.D = nn.Parameter(torch.zeros(d_ssm))

        self.forward_core = self.forward_selective_scan

        self.x_proj = nn.Linear(d_ssm, self.ssm_dt_rank + self.ssm_d_state * 2, bias=False)


    def forward_selective_scan(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        x_hw = x.permute(0, 2, 3, 1).contiguous().view(B, L, C)

        xz = self.x_proj(x_hw)
        dt, B_hw, C_hw = torch.split(xz, [self.ssm_dt_rank, self.ssm_d_state, self.ssm_d_state], dim=-1)
        
        dt = self.dt_proj.weight @ dt.transpose(1, 2)
        dt = dt.transpose(1, 2).contiguous()
        
        A = -torch.exp(self.A_log.float())
        y_flat = selective_scan_fn(x_hw, dt, A, B_hw, C_hw, self.D.float(), z=None, delta_bias=self.dt_proj.bias.float(), delta_softplus=True)
        
        return y_flat.view(B, H, W, C).permute(0, 3, 1, 2)

    def forward(self, x: torch.Tensor):
        B, H, W, C = x.shape
        
        x_norm = self.ln(x)
        
        xz, _ = self.in_proj(x_norm).chunk(2, dim=-1)
        
        if self.ssm_simple:
            y = self.forward_core(self.act(self.conv2d(xz.permute(0, 3, 1, 2))))
        else:
            x_conv = self.act(self.conv2d(xz.permute(0, 3, 1, 2)))
            y = self.forward_core(x_conv)

        y = y.permute(0, 2, 3, 1)
        y = self.out_proj(y)
        
        output = x + self.drop_path(y)
        return output

class SS2D(VSSBlock_original):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.forward_core = self.forward_ss2d

    def forward_ss2d(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        
        xz = self.x_proj(x.permute(0, 2, 3, 1).contiguous().view(B, L, C))
        dt, B_hw, C_hw = torch.split(xz, [self.ssm_dt_rank, self.ssm_d_state, self.ssm_d_state], dim=-1)

        dt = self.dt_proj.weight @ dt.transpose(1, 2)
        
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        
        y_fwd = selective_scan_fn(x.reshape(B, C, L), dt.transpose(1, 2), A, B_hw.transpose(1, 2), C_hw.transpose(1, 2), D, z=None, delta_bias=self.dt_proj.bias.float(), delta_softplus=True)
        
        x_rev = torch.flip(x.reshape(B, C, L), dims=[-1])
        dt_rev = torch.flip(dt.transpose(1, 2), dims=[-1])
        B_rev = torch.flip(B_hw.transpose(1, 2), dims=[-1])
        C_rev = torch.flip(C_hw.transpose(1, 2), dims=[-1])
        y_bwd = selective_scan_fn(x_rev, dt_rev, A, B_rev, C_rev, D, z=None, delta_bias=self.dt_proj.bias.float(), delta_softplus=True)
        y_bwd = torch.flip(y_bwd, dims=[-1])
        
        return (y_fwd + y_bwd).view(B, C, H, W)
