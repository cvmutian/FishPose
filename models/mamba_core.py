import torch
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor
from .mamba_backbone import MambaBackbone
import os

class MambaCore(nn.Module):
    def __init__(self, checkpoint_path: str = None, in_channels: int = 3, output_dims: list = None, feature_keys: dict = None, **kwargs):
        super().__init__()
        
        if output_dims is None:
            output_dims = [24, 40, 112, 320]
        if len(output_dims) != 4:
            raise ValueError("output_dims must be a list of 4 integers.")

        if feature_keys is None:
            feature_keys = {
                'blocks.1': 'p2',
                'blocks.2': 'p3',
                'blocks.4': 'p4',
                'blocks.6': 'p5',
            }
        
        self.feature_keys = feature_keys

        model_structure = MambaBackbone(in_chans=in_channels)
        
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            try:
                state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            except:
                state_dict = torch.load(checkpoint_path, map_location='cpu')

            model_structure.load_state_dict(state_dict, strict=False)

        self.feature_extractor = create_feature_extractor(model_structure, return_nodes=self.feature_keys)
        
        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, 256, 256)
            dummy_features = self.feature_extractor(dummy_input)
            self.output_dims = [dummy_features[key].shape[1] for key in self.feature_keys.values()]

        self.adapter_convs = nn.ModuleList()
        for i in range(len(output_dims)):
            adapter = nn.Conv2d(self.output_dims[i], output_dims[i], kernel_size=1)
            self.adapter_convs.append(adapter)

    def forward(self, x: torch.Tensor) -> list:
        features_dict = self.feature_extractor(x)
        features_list = list(features_dict.values())
        
        adapted_features = []
        for i in range(len(features_list)):
            adapted_features.append(self.adapter_convs[i](features_list[i]))
            
        return adapted_features
