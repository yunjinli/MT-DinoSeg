from dataclasses import dataclass, field
from typing import List, Optional, Dict

from .base import DecoderConfig
from .registry import ConfigRegistry
from .base import ShapeSpec

# Create a registry for probing head configurations
DecoderRegistry = ConfigRegistry[DecoderConfig]("DecoderRegistry")

@DecoderRegistry.register("multitask_mask2former_head")
@dataclass
class MultitaskMask2FormerHeadConfig(DecoderConfig):
    """mask2former head."""
    name: str = 'multitask_mask2former_head'
    input_dim: int = -1
    # num_classes: int = -1
    num_class_dict: Dict = None
    # Input shape information (will be set automatically)
    input_shape: Dict[str, ShapeSpec] = field(default_factory=dict)
    
    # Pixel decoder configuration
    conv_dim: int = 256
    mask_dim: int = 256
    norm: str = "GN"  # GroupNorm
    
    # Deformable transformer pixel decoder config
    transformer_dropout: float = 0.0
    transformer_nheads: int = 8
    transformer_dim_feedforward: int = 1024
    transformer_enc_layers: int = 6
    transformer_in_features: List[str] = field(default_factory=lambda: ["res2", "res3", "res4", "res5"])
    common_stride: int = 4
    
    # Transformer decoder config
    maskformer_hidden_dim: int = 256
    # num_obj_queries: int = 150
    # num_obj_queries_dict: Dict = field(default_factory=lambda: {'sem_seg': 100, 'lane': 25, 'drivable': 25})
    num_obj_queries_dict: Dict = None
    maskformer_nheads: int = 8
    maskformer_dim_feedforward: int = 2048
    dec_layers: int = 9
    pre_norm: bool = False
    enforce_input_project: bool = False
    
    # Input feature selection for transformer
    maskformer_in_feature: str = "multi_scale_pixel_decoder"  # or "transformer_encoder", "pixel_embedding"
    
    # Training specific
    mask_classification: bool = True

    ## Optimization 
    # Loss parameters:
    deep_supervision: bool = True
    # no_object_weight: float = 1.0
    no_object_weight: float = 0.1
    class_weight: float = 2.0
    dice_weight: float = 5.0
    mask_weight: float = 5.0
    train_num_points: int = 64 * 64
    oversample_ratio: float = 3.0
    importance_sample_ratio: float = 0.75

    @property
    def num_obj_queries(self):
        counter = 0
        for _, num_obj_q in self.num_obj_queries_dict.items():
            counter += num_obj_q
        return counter
    @property
    def num_classes(self):
        counter = 0
        for _, num in self.num_class_dict.items():
            counter += num
        return counter

@DecoderRegistry.register("open_mask2former_head")
@dataclass
class OpenMask2FormerHeadConfig(DecoderConfig):
    """mask2former head."""
    name: str = 'open_mask2former_head'
    input_dim: int = -1
    num_classes: int = -1
    class_names = None
    # Input shape information (will be set automatically)
    input_shape: Dict[str, ShapeSpec] = field(default_factory=dict)
    
    # Pixel decoder configuration
    conv_dim: int = 256
    mask_dim: int = 256
    norm: str = "GN"  # GroupNorm
    
    # Deformable transformer pixel decoder config
    transformer_dropout: float = 0.0
    transformer_nheads: int = 8
    transformer_dim_feedforward: int = 1024
    transformer_enc_layers: int = 6
    transformer_in_features: List[str] = field(default_factory=lambda: ["res2", "res3", "res4", "res5"])
    common_stride: int = 4
    
    # Transformer decoder config
    maskformer_hidden_dim: int = 256
    num_obj_queries: int = 100
    maskformer_nheads: int = 8
    maskformer_dim_feedforward: int = 2048
    dec_layers: int = 9
    pre_norm: bool = False
    enforce_input_project: bool = False
    
    # Input feature selection for transformer
    maskformer_in_feature: str = "multi_scale_pixel_decoder"  # or "transformer_encoder", "pixel_embedding"
    
    # Training specific
    mask_classification: bool = True

    ## Optimization 
    # Loss parameters:
    deep_supervision: bool = True
    # no_object_weight: float = 1.0
    no_object_weight: float = 0.1
    class_weight: float = 2.0
    dice_weight: float = 5.0
    mask_weight: float = 5.0
    train_num_points: int = 64 * 64
    oversample_ratio: float = 3.0
    importance_sample_ratio: float = 0.75

@DecoderRegistry.register("mask2former_head")
@dataclass
class Mask2FormerHeadConfig(DecoderConfig):
    """mask2former head."""
    name: str = 'mask2former_head'
    input_dim: int = -1
    num_classes: int = -1
    # Input shape information (will be set automatically)
    input_shape: Dict[str, ShapeSpec] = field(default_factory=dict)
    
    # Pixel decoder configuration
    conv_dim: int = 256
    mask_dim: int = 256
    norm: str = "GN"  # GroupNorm
    
    # Deformable transformer pixel decoder config
    transformer_dropout: float = 0.0
    transformer_nheads: int = 8
    transformer_dim_feedforward: int = 1024
    transformer_enc_layers: int = 6
    transformer_in_features: List[str] = field(default_factory=lambda: ["res2", "res3", "res4", "res5"])
    common_stride: int = 4
    
    # Transformer decoder config
    maskformer_hidden_dim: int = 256
    num_obj_queries: int = 100
    maskformer_nheads: int = 8
    maskformer_dim_feedforward: int = 2048
    dec_layers: int = 9
    pre_norm: bool = False
    enforce_input_project: bool = False
    
    # Input feature selection for transformer
    maskformer_in_feature: str = "multi_scale_pixel_decoder"  # or "transformer_encoder", "pixel_embedding"
    
    # Training specific
    mask_classification: bool = True

    ## Optimization 
    # Loss parameters:
    deep_supervision: bool = True
    # no_object_weight: float = 1.0
    no_object_weight: float = 0.1
    class_weight: float = 2.0
    dice_weight: float = 5.0
    mask_weight: float = 5.0
    train_num_points: int = 64 * 64
    oversample_ratio: float = 3.0
    importance_sample_ratio: float = 0.75

@DecoderRegistry.register("segformer_head")
@dataclass
class SegFormerHeadConfig(DecoderConfig):
    """SegFormer head."""
    name: str = 'segformer_head'
    input_dim: int = -1
    num_classes: int = -1
    in_channels: List[int] = field(default_factory=lambda: [384, 384, 384, 384])
    embed_dim: int = 512
    encoder_name: str = None
    dropout: float = 0.1

@DecoderRegistry.register("linear_probing")
@dataclass
class LinearProbingConfig(DecoderConfig):
    """Linear probing head."""
    name: str = 'linear_probing'
    input_dim: int = -1
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.1
    activation: str = "relu"
    num_classes: int = -1
    # spatial_size: int = -1