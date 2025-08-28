from dataclasses import dataclass, field
from typing import Optional, Tuple

from .base import ModelConfig, EncoderConfig, DecoderConfig, ShapeSpec
from .registry import ConfigRegistry
from .encoder import EncoderRegistry
from .decoder import DecoderRegistry

# Create registry for full model configurations
ModelRegistry = ConfigRegistry[ModelConfig]("ModelRegistry")

@ModelRegistry.register("dino_vits8-linear_probing")
@dataclass
class DinoViTS8LinearProbingConfig(ModelConfig):
    name: str = 'dino_vits8-linear_probing'
    input_size: Tuple[int, int] = (224, 224)
    encoder: EncoderConfig = field(
        default_factory=lambda: EncoderRegistry.get("dino_vits8")()
    )
    decoder: DecoderConfig = field(
        default_factory=lambda: DecoderRegistry.get("linear_probing")()
    )
    
    def __post_init__(self):
        # Ensure decoder input_dim matches encoder output_dim
        self.decoder.input_dim = self.encoder.output_dim

@ModelRegistry.register("lora_dino_vits8-linear_probing")
@dataclass
class LoRADinoViTS8LinearProbingConfig(ModelConfig):
    name: str = 'lora_dino_vits8-linear_probing'
    input_size: Tuple[int, int] = (224, 224)
    encoder: EncoderConfig = field(
        default_factory=lambda: EncoderRegistry.get("lora_dino_vits8")()
    )
    decoder: DecoderConfig = field(
        default_factory=lambda: DecoderRegistry.get("linear_probing")()
    )
    
    def __post_init__(self):
        # Ensure decoder input_dim matches encoder output_dim
        self.decoder.input_dim = self.encoder.output_dim

@ModelRegistry.register("dinov2_vits14-linear_probing")
@dataclass
class Dinov2ViTS14LinearProbingConfig(ModelConfig):
    name: str = 'dinov2_vits14-linear_probing'
    input_size: Tuple[int, int] = (392, 392) ## So that the output spatial size from the encoder is still (28, 28)

    encoder: EncoderConfig = field(
        default_factory=lambda: EncoderRegistry.get("dinov2_vits14")()
    )
    decoder: DecoderConfig = field(
        default_factory=lambda: DecoderRegistry.get("linear_probing")()
    )
    
    def __post_init__(self):
        # Ensure decoder input_dim matches encoder output_dim
        self.decoder.input_dim = self.encoder.output_dim

@ModelRegistry.register("lora_dinov2_vits14-linear_probing")
@dataclass
class LoRADinov2ViTS14LinearProbingConfig(ModelConfig):
    name: str = 'lora_dinov2_vits14-linear_probing'
    input_size: Tuple[int, int] = (392, 392) ## So that the output spatial size from the encoder is still (28, 28)

    encoder: EncoderConfig = field(
        default_factory=lambda: EncoderRegistry.get("lora_dinov2_vits14")()
    )
    decoder: DecoderConfig = field(
        default_factory=lambda: DecoderRegistry.get("linear_probing")()
    )
    
    def __post_init__(self):
        # Ensure decoder input_dim matches encoder output_dim
        self.decoder.input_dim = self.encoder.output_dim

@ModelRegistry.register("dinov2_vits14_px224-linear_probing")
@dataclass
class Dinov2ViTS14Px224LinearProbingConfig(ModelConfig):
    name: str = 'dinov2_vits14_px224-linear_probing'
    input_size: Tuple[int, int] = (224, 224) ## So that the output spatial size from the encoder is still (28, 28)

    encoder: EncoderConfig = field(
        default_factory=lambda: EncoderRegistry.get("dinov2_vits14")()
    )
    decoder: DecoderConfig = field(
        default_factory=lambda: DecoderRegistry.get("linear_probing")()
    )
    
    def __post_init__(self):
        # Ensure decoder input_dim matches encoder output_dim
        self.decoder.input_dim = self.encoder.output_dim

@ModelRegistry.register("lora_dinov2_vits14_px224-linear_probing")
@dataclass
class LoRADinov2ViTS14Px224LinearProbingConfig(ModelConfig):
    name: str = 'lora_dinov2_vits14_px224-linear_probing'
    input_size: Tuple[int, int] = (224, 224) ## So that the output spatial size from the encoder is still (28, 28)

    encoder: EncoderConfig = field(
        default_factory=lambda: EncoderRegistry.get("lora_dinov2_vits14")()
    )
    decoder: DecoderConfig = field(
        default_factory=lambda: DecoderRegistry.get("linear_probing")()
    )
    
    def __post_init__(self):
        # Ensure decoder input_dim matches encoder output_dim
        self.decoder.input_dim = self.encoder.output_dim

@ModelRegistry.register("clip_vitb16-linear_probing")
@dataclass
class CLIPViTB16LinearProbingConfig(ModelConfig):
    name: str = 'clip_vitb16-linear_probing'
    input_size: Tuple[int, int] = (224, 224) ## So that the output spatial size from the encoder is still (28, 28)

    encoder: EncoderConfig = field(
        default_factory=lambda: EncoderRegistry.get("clip_vitb16")()
    )
    decoder: DecoderConfig = field(
        default_factory=lambda: DecoderRegistry.get("linear_probing")()
    )
    
    def __post_init__(self):
        # Ensure decoder input_dim matches encoder output_dim
        self.decoder.input_dim = self.encoder.output_dim

@ModelRegistry.register("dino_vits8-segformer_head")
@dataclass
class DinoViTS8SegFormerHeadConfig(ModelConfig):
    name: str = 'dino_vits8-segformer_head'
    input_size: Tuple[int, int] = (224, 224)
    encoder: EncoderConfig = field(
        default_factory=lambda: EncoderRegistry.get("dino_vits8")()
    )
    decoder: DecoderConfig = field(
        default_factory=lambda: DecoderRegistry.get("segformer_head")()
    )
    
    def __post_init__(self):
        # Ensure decoder input_dim matches encoder output_dim
        self.decoder.input_dim = self.encoder.output_dim
        self.decoder.in_channels = [self.encoder.output_dim, self.encoder.output_dim, self.encoder.output_dim, self.encoder.output_dim] 
        self.decoder.encoder_name = self.encoder.name

@ModelRegistry.register("dinov2_vits14-segformer_head")
@dataclass
class Dinov2ViTS14SegFormerHeadConfig(ModelConfig):
    name: str = 'dinov2_vits14-segformer_head'
    input_size: Tuple[int, int] = (392, 392)
    encoder: EncoderConfig = field(
        default_factory=lambda: EncoderRegistry.get("dinov2_vits14")()
    )
    decoder: DecoderConfig = field(
        default_factory=lambda: DecoderRegistry.get("segformer_head")()
    )
    
    def __post_init__(self):
        # Ensure decoder input_dim matches encoder output_dim
        self.decoder.input_dim = self.encoder.output_dim
        self.decoder.in_channels = [self.encoder.output_dim, self.encoder.output_dim, self.encoder.output_dim, self.encoder.output_dim] 
        self.decoder.encoder_name = self.encoder.name

@ModelRegistry.register("dinov2_vits14_px224-segformer_head")
@dataclass
class Dinov2ViTS14Px224SegFormerHeadConfig(ModelConfig):
    name: str = 'dinov2_vits14_px224-segformer_head'
    input_size: Tuple[int, int] = (224, 224)
    encoder: EncoderConfig = field(
        default_factory=lambda: EncoderRegistry.get("dinov2_vits14")()
    )
    decoder: DecoderConfig = field(
        default_factory=lambda: DecoderRegistry.get("segformer_head")()
    )
    
    def __post_init__(self):
        # Ensure decoder input_dim matches encoder output_dim
        self.decoder.input_dim = self.encoder.output_dim
        self.decoder.in_channels = [self.encoder.output_dim, self.encoder.output_dim, self.encoder.output_dim, self.encoder.output_dim] 
        self.decoder.encoder_name = self.encoder.name

@ModelRegistry.register("clip_vitb16-segformer_head")
@dataclass
class CLIPViTB16SegFormerHeadConfig(ModelConfig):
    name: str = 'clip_vitb16-segformer_head'
    input_size: Tuple[int, int] = (224, 224)
    encoder: EncoderConfig = field(
        default_factory=lambda: EncoderRegistry.get("clip_vitb16")()
    )
    decoder: DecoderConfig = field(
        default_factory=lambda: DecoderRegistry.get("segformer_head")()
    )
    
    def __post_init__(self):
        # Ensure decoder input_dim matches encoder output_dim
        self.decoder.input_dim = self.encoder.output_dim
        self.decoder.in_channels = [self.encoder.output_dim, self.encoder.output_dim, self.encoder.output_dim, self.encoder.output_dim] 
        self.decoder.encoder_name = self.encoder.name

@ModelRegistry.register("swinv2_tiny_window8_256-segformer_head")
@dataclass
class SwinV2TinyWindow8SegFormerHeadConfig(ModelConfig):
    name: str = 'swinv2_tiny_window8_256-segformer_head'
    input_size: Tuple[int, int] = (256, 256)
    encoder: EncoderConfig = field(
        default_factory=lambda: EncoderRegistry.get("swinv2_tiny_window8_256")()
    )
    decoder: DecoderConfig = field(
        default_factory=lambda: DecoderRegistry.get("segformer_head")()
    )
    
    def __post_init__(self):
        # Ensure decoder input_dim matches encoder output_dim
        self.decoder.in_channels = self.encoder.output_dim
        self.decoder.input_dim = self.encoder.output_dim
        self.decoder.encoder_name = self.encoder.name

@ModelRegistry.register("lora_swinv2_tiny_window8_256-segformer_head")
@dataclass
class LoRASwinV2TinyWindow8SegFormerHeadConfig(ModelConfig):
    name: str = 'lora_swinv2_tiny_window8_256-segformer_head'
    input_size: Tuple[int, int] = (256, 256)
    encoder: EncoderConfig = field(
        default_factory=lambda: EncoderRegistry.get("lora_swinv2_tiny_window8_256")()
    )
    decoder: DecoderConfig = field(
        default_factory=lambda: DecoderRegistry.get("segformer_head")()
    )
    
    def __post_init__(self):
        # Ensure decoder input_dim matches encoder output_dim
        self.decoder.in_channels = self.encoder.output_dim
        self.decoder.input_dim = self.encoder.output_dim
        self.decoder.encoder_name = self.encoder.name

@ModelRegistry.register("swinv2_small_window8_256-segformer_head")
@dataclass
class SwinV2SmallWindow8SegFormerHeadConfig(ModelConfig):
    name: str = 'swinv2_small_window8_256-segformer_head'
    input_size: Tuple[int, int] = (256, 256)
    encoder: EncoderConfig = field(
        default_factory=lambda: EncoderRegistry.get("swinv2_small_window8_256")()
    )
    decoder: DecoderConfig = field(
        default_factory=lambda: DecoderRegistry.get("segformer_head")()
    )
    
    def __post_init__(self):
        # Ensure decoder input_dim matches encoder output_dim
        self.decoder.in_channels = self.encoder.output_dim
        self.decoder.input_dim = self.encoder.output_dim
        self.decoder.encoder_name = self.encoder.name

@ModelRegistry.register("lora_swinv2_small_window8_256-segformer_head")
@dataclass
class LoRASwinV2SmallWindow8SegFormerHeadConfig(ModelConfig):
    name: str = 'lora_swinv2_small_window8_256-segformer_head'
    input_size: Tuple[int, int] = (256, 256)
    encoder: EncoderConfig = field(
        default_factory=lambda: EncoderRegistry.get("lora_swinv2_small_window8_256")()
    )
    decoder: DecoderConfig = field(
        default_factory=lambda: DecoderRegistry.get("segformer_head")()
    )
    
    def __post_init__(self):
        # Ensure decoder input_dim matches encoder output_dim
        self.decoder.in_channels = self.encoder.output_dim
        self.decoder.input_dim = self.encoder.output_dim
        self.decoder.encoder_name = self.encoder.name

@ModelRegistry.register("swinv2_base_window8_256-segformer_head")
@dataclass
class SwinV2BaseWindow8SegFormerHeadConfig(ModelConfig):
    name: str = 'swinv2_base_window8_256-segformer_head'
    input_size: Tuple[int, int] = (256, 256)
    encoder: EncoderConfig = field(
        default_factory=lambda: EncoderRegistry.get("swinv2_base_window8_256")()
    )
    decoder: DecoderConfig = field(
        default_factory=lambda: DecoderRegistry.get("segformer_head")()
    )
    
    def __post_init__(self):
        # Ensure decoder input_dim matches encoder output_dim
        self.decoder.in_channels = self.encoder.output_dim
        self.decoder.input_dim = self.encoder.output_dim
        self.decoder.encoder_name = self.encoder.name

@ModelRegistry.register("lora_swinv2_base_window8_256-segformer_head")
@dataclass
class LoRASwinV2BaseWindow8SegFormerHeadConfig(ModelConfig):
    name: str = 'lora_swinv2_base_window8_256-segformer_head'
    input_size: Tuple[int, int] = (256, 256)
    encoder: EncoderConfig = field(
        default_factory=lambda: EncoderRegistry.get("lora_swinv2_base_window8_256")()
    )
    decoder: DecoderConfig = field(
        default_factory=lambda: DecoderRegistry.get("segformer_head")()
    )
    
    def __post_init__(self):
        # Ensure decoder input_dim matches encoder output_dim
        self.decoder.in_channels = self.encoder.output_dim
        self.decoder.input_dim = self.encoder.output_dim
        self.decoder.encoder_name = self.encoder.name

@ModelRegistry.register("swinv2_base_window8_256-mask2former_head")
@dataclass
class SwinV2BaseWindow8Mask2FormerHeadConfig(ModelConfig):
    name: str = 'swinv2_base_window8_256-mask2former_head'
    input_size: Tuple[int, int] = (256, 256)
    encoder: EncoderConfig = field(
        default_factory=lambda: EncoderRegistry.get("swinv2_base_window8_256")()
    )
    decoder: DecoderConfig = field(
        default_factory=lambda: DecoderRegistry.get("mask2former_head")()
    )
    
    def __post_init__(self):
        # Ensure decoder input_dim matches encoder output_dim
        # self.decoder.in_channels = self.encoder.output_dim
        # self.decoder.input_dim = self.encoder.output_dim
        # self.decoder.encoder_name = self.encoder.name

        channels_list = self.encoder.output_dim

        self.decoder.input_shape = {
            "res2": ShapeSpec(channels=channels_list[0], stride=4, height=64, width=64),   # 256/4
            "res3": ShapeSpec(channels=channels_list[1], stride=8, height=32, width=32),   # 256/8
            "res4": ShapeSpec(channels=channels_list[2], stride=16, height=16, width=16),  # 256/16
            "res5": ShapeSpec(channels=channels_list[3], stride=32, height=8, width=8),    # 256/32
        }
        self.decoder.transformer_in_features = ["res2", "res3", "res4", "res5"]

@ModelRegistry.register("lora_swinv2_base_window8_256-mask2former_head")
@dataclass
class LoRASwinV2BaseWindow8Mask2FormerHeadConfig(ModelConfig):
    name: str = 'lora_swinv2_base_window8_256-mask2former_head'
    input_size: Tuple[int, int] = (256, 256)
    encoder: EncoderConfig = field(
        default_factory=lambda: EncoderRegistry.get("lora_swinv2_base_window8_256")()
    )
    decoder: DecoderConfig = field(
        default_factory=lambda: DecoderRegistry.get("mask2former_head")()
    )
    
    def __post_init__(self):
        # Ensure decoder input_dim matches encoder output_dim
        # self.decoder.in_channels = self.encoder.output_dim
        # self.decoder.input_dim = self.encoder.output_dim
        # self.decoder.encoder_name = self.encoder.name

        channels_list = self.encoder.output_dim

        self.decoder.input_shape = {
            "res2": ShapeSpec(channels=channels_list[0], stride=4, height=64, width=64),   # 256/4
            "res3": ShapeSpec(channels=channels_list[1], stride=8, height=32, width=32),   # 256/8
            "res4": ShapeSpec(channels=channels_list[2], stride=16, height=16, width=16),  # 256/16
            "res5": ShapeSpec(channels=channels_list[3], stride=32, height=8, width=8),    # 256/32
        }
        self.decoder.transformer_in_features = ["res2", "res3", "res4", "res5"]

@ModelRegistry.register("vit_adapter_dinov2_vits14-mask2former_head")
@dataclass
class ViTAdapterDinov2ViTS14Mask2FormerHeadConfig(ModelConfig):
    name: str = 'vit_adapter_dinov2_vits14-mask2former_head'
    input_size: Tuple[int, int] = (224, 224)
    encoder: EncoderConfig = field(
        default_factory=lambda: EncoderRegistry.get("vit_adapter_dinov2_vits14")()
    )
    decoder: DecoderConfig = field(
        default_factory=lambda: DecoderRegistry.get("mask2former_head")()
    )
    
    def __post_init__(self):
        channels_list = self.encoder.output_dim

        self.decoder.input_shape = {
            "res2": ShapeSpec(channels=channels_list[0], stride=4, height=self.input_size[0] // 4, width=self.input_size[1] // 4), 
            "res3": ShapeSpec(channels=channels_list[1], stride=8, height=self.input_size[0] // 8, width=self.input_size[1] // 8),   
            "res4": ShapeSpec(channels=channels_list[2], stride=16, height=self.input_size[0] // 16, width=self.input_size[1] // 16),  
            "res5": ShapeSpec(channels=channels_list[3], stride=32, height=self.input_size[0] // 32, width=self.input_size[1] // 32),    
        }
        self.decoder.transformer_in_features = ["res2", "res3", "res4", "res5"]

@ModelRegistry.register("vit_adapter_dinov2_vitb14-mask2former_head")
@dataclass
class ViTAdapterDinov2ViTB14Mask2FormerHeadConfig(ModelConfig):
    name: str = 'vit_adapter_dinov2_vitb14-mask2former_head'
    input_size: Tuple[int, int] = (224, 224)
    encoder: EncoderConfig = field(
        default_factory=lambda: EncoderRegistry.get("vit_adapter_dinov2_vitb14")()
    )
    decoder: DecoderConfig = field(
        default_factory=lambda: DecoderRegistry.get("mask2former_head")()
    )
    
    def __post_init__(self):
        channels_list = self.encoder.output_dim

        self.decoder.input_shape = {
            "res2": ShapeSpec(channels=channels_list[0], stride=4, height=self.input_size[0] // 4, width=self.input_size[1] // 4), 
            "res3": ShapeSpec(channels=channels_list[1], stride=8, height=self.input_size[0] // 8, width=self.input_size[1] // 8),   
            "res4": ShapeSpec(channels=channels_list[2], stride=16, height=self.input_size[0] // 16, width=self.input_size[1] // 16),  
            "res5": ShapeSpec(channels=channels_list[3], stride=32, height=self.input_size[0] // 32, width=self.input_size[1] // 32),    
        }
        self.decoder.transformer_in_features = ["res2", "res3", "res4", "res5"]

@ModelRegistry.register("vit_adapter_lora_dinov2_vits14-mask2former_head")
@dataclass
class ViTAdapterLoRADinov2ViTS14Mask2FormerHeadConfig(ModelConfig):
    name: str = 'vit_adapter_lora_dinov2_vits14-mask2former_head'
    input_size: Tuple[int, int] = (224, 224)
    encoder: EncoderConfig = field(
        default_factory=lambda: EncoderRegistry.get("vit_adapter_lora_dinov2_vits14")()
    )
    decoder: DecoderConfig = field(
        default_factory=lambda: DecoderRegistry.get("mask2former_head")()
    )
    
    def __post_init__(self):
        channels_list = self.encoder.output_dim

        self.decoder.input_shape = {
            "res2": ShapeSpec(channels=channels_list[0], stride=4, height=self.input_size[0] // 4, width=self.input_size[1] // 4), 
            "res3": ShapeSpec(channels=channels_list[1], stride=8, height=self.input_size[0] // 8, width=self.input_size[1] // 8),   
            "res4": ShapeSpec(channels=channels_list[2], stride=16, height=self.input_size[0] // 16, width=self.input_size[1] // 16),  
            "res5": ShapeSpec(channels=channels_list[3], stride=32, height=self.input_size[0] // 32, width=self.input_size[1] // 32),    
        }
        self.decoder.transformer_in_features = ["res2", "res3", "res4", "res5"]

@ModelRegistry.register("vit_adapter_lora_dinov2_vitb14-mask2former_head")
@dataclass
class ViTAdapterLoRADinov2ViTB14Mask2FormerHeadConfig(ModelConfig):
    name: str = 'vit_adapter_lora_dinov2_vitb14-mask2former_head'
    input_size: Tuple[int, int] = (224, 224)
    encoder: EncoderConfig = field(
        default_factory=lambda: EncoderRegistry.get("vit_adapter_lora_dinov2_vitb14")()
    )
    decoder: DecoderConfig = field(
        default_factory=lambda: DecoderRegistry.get("mask2former_head")()
    )
    
    def __post_init__(self):
        channels_list = self.encoder.output_dim

        self.decoder.input_shape = {
            "res2": ShapeSpec(channels=channels_list[0], stride=4, height=self.input_size[0] // 4, width=self.input_size[1] // 4), 
            "res3": ShapeSpec(channels=channels_list[1], stride=8, height=self.input_size[0] // 8, width=self.input_size[1] // 8),   
            "res4": ShapeSpec(channels=channels_list[2], stride=16, height=self.input_size[0] // 16, width=self.input_size[1] // 16),  
            "res5": ShapeSpec(channels=channels_list[3], stride=32, height=self.input_size[0] // 32, width=self.input_size[1] // 32),    
        }
        self.decoder.transformer_in_features = ["res2", "res3", "res4", "res5"]

@ModelRegistry.register("vit_adapter_dinov2_vitb14-open_mask2former_head")
@dataclass
class ViTAdapterDinov2ViTB14OpenMask2FormerHeadConfig(ModelConfig):
    name: str = 'vit_adapter_dinov2_vitb14-open_mask2former_head'
    input_size: Tuple[int, int] = (224, 224)
    encoder: EncoderConfig = field(
        default_factory=lambda: EncoderRegistry.get("vit_adapter_dinov2_vitb14")()
    )
    decoder: DecoderConfig = field(
        default_factory=lambda: DecoderRegistry.get("open_mask2former_head")()
    )
    
    def __post_init__(self):
        channels_list = self.encoder.output_dim

        self.decoder.input_shape = {
            "res2": ShapeSpec(channels=channels_list[0], stride=4, height=self.input_size[0] // 4, width=self.input_size[1] // 4), 
            "res3": ShapeSpec(channels=channels_list[1], stride=8, height=self.input_size[0] // 8, width=self.input_size[1] // 8),   
            "res4": ShapeSpec(channels=channels_list[2], stride=16, height=self.input_size[0] // 16, width=self.input_size[1] // 16),  
            "res5": ShapeSpec(channels=channels_list[3], stride=32, height=self.input_size[0] // 32, width=self.input_size[1] // 32),    
        }
        self.decoder.transformer_in_features = ["res2", "res3", "res4", "res5"]


@ModelRegistry.register("vit_adapter_dinov2_vitb14-multitask_mask2former_head")
@dataclass
class ViTAdapterDinov2ViTB14MultitaskMask2FormerHeadConfig(ModelConfig):
    name: str = 'vit_adapter_dinov2_vitb14-multitask_mask2former_head'
    input_size: Tuple[int, int] = (224, 224)
    encoder: EncoderConfig = field(
        default_factory=lambda: EncoderRegistry.get("vit_adapter_dinov2_vitb14")()
    )
    decoder: DecoderConfig = field(
        default_factory=lambda: DecoderRegistry.get("multitask_mask2former_head")()
    )
    
    def __post_init__(self):
        channels_list = self.encoder.output_dim

        self.decoder.input_shape = {
            "res2": ShapeSpec(channels=channels_list[0], stride=4, height=self.input_size[0] // 4, width=self.input_size[1] // 4), 
            "res3": ShapeSpec(channels=channels_list[1], stride=8, height=self.input_size[0] // 8, width=self.input_size[1] // 8),   
            "res4": ShapeSpec(channels=channels_list[2], stride=16, height=self.input_size[0] // 16, width=self.input_size[1] // 16),  
            "res5": ShapeSpec(channels=channels_list[3], stride=32, height=self.input_size[0] // 32, width=self.input_size[1] // 32),    
        }
        self.decoder.transformer_in_features = ["res2", "res3", "res4", "res5"]