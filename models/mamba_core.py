import torch
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor
from .mamba_backbone import MambaBackbone
import os

class MambaCore(nn.Module):
    """
    The core feature extraction backbone.
    This module uses a custom-built model structure and loads weights from a local 
    checkpoint file, making it fully self-contained and offline.
    """
    def __init__(self, checkpoint_path: str = None, in_channels: int = 3, output_dims: list = None, feature_keys: dict = None, **kwargs):
        """
        Args:
            checkpoint_path (str, optional): Path to the local checkpoint file to load.
                If None, the backbone will be initialized without pretrained weights.
                This is useful when loading a complete model checkpoint that includes backbone weights.
            in_channels (int): Number of input channels.
            output_dims (list): Desired output channel dimensions for each feature map.
        """
        super().__init__()
        
        if output_dims is None:
            output_dims = [24, 40, 112, 320]
        if len(output_dims) != 4:
            raise ValueError("output_dims must be a list of 4 integers.")

        # Define the feature keys if the user did not provide them in the config
        if feature_keys is None:
            feature_keys = {
                'blocks.1': 'p2',  # Stride 4, 24 channels
                'blocks.2': 'p3',  # Stride 8, 40 channels
                'blocks.4': 'p4',  # Stride 16, 112 channels
                'blocks.6': 'p5',  # Stride 32, 320 channels
            }
        
        self.feature_keys = feature_keys

        # 1. Create an instance of the custom backbone with the correct structure.
        model_structure = MambaBackbone(in_chans=in_channels)
        
        # Load the state dictionary only if checkpoint_path is provided
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            # Load the state dictionary
            # Use weights_only=True for security and strict=False to ignore classifier weights
            try:
                state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            except:
                # Fallback for older torch versions or other issues
                state_dict = torch.load(checkpoint_path, map_location='cpu')

            # Load the state dict, ignoring missing keys (like a classifier)
            model_structure.load_state_dict(state_dict, strict=False)

        # Use timm's feature extractor to get intermediate features
        self.feature_extractor = create_feature_extractor(model_structure, return_nodes=self.feature_keys)
        
        # Get the channel dimensions from the feature extractor
        with torch.no_grad():
            # Create a dummy input tensor to run a forward pass and infer output shapes
            dummy_input = torch.randn(1, in_channels, 256, 256)
            dummy_features = self.feature_extractor(dummy_input)
            self.output_dims = [dummy_features[key].shape[1] for key in self.feature_keys.values()]

        # Create 1x1 convolution layers to adapt the channel dimensions
        self.adapter_convs = nn.ModuleList()
        for i in range(len(output_dims)):
            adapter = nn.Conv2d(self.output_dims[i], output_dims[i], kernel_size=1)
            self.adapter_convs.append(adapter)

    def forward(self, x: torch.Tensor) -> list:
        """
        Forward pass through the backbone.
        """
        features_dict = self.feature_extractor(x)
        features_list = list(features_dict.values())
        
        adapted_features = []
        for i in range(len(features_list)):
            adapted_features.append(self.adapter_convs[i](features_list[i]))
            
        return adapted_features
