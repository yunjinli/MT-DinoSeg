# src/config/schema.py
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import yaml
from pathlib import Path

@dataclass
class EncoderConfig:
    """Base configuration for encoders"""
    name: str
    freeze: bool = True
    output_dim: int = -1 ## will be set later

@dataclass
class DecoderConfig:
    """Base configuration for decoders"""
    name: str
    input_dim: int = -1 ## Will be set later
    # num_classes: int = -1 ## Will be set later

@dataclass
class ModelConfig:
    """Base configuration for models."""
    encoder: EncoderConfig
    decoder: DecoderConfig
    input_size: Tuple[int, int]

@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    name: str = "adam"
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    betas: tuple = (0.9, 0.999)  # For Adam-based optimizers
    momentum: float = 0.9  # For SGD

@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    name: str = "cosine"
    warmup_epochs: int = 5
    min_lr: float = 1e-6

@dataclass
class TrainingConfig:
    """Base configuration for training."""
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: Optional[SchedulerConfig] = field(default_factory=SchedulerConfig)
    batch_size: int = -1
    epochs: int = -1

    use_amp: bool = False
    grad_clip_val: Optional[float] = 1.0 
    loss_scale: float = 2**16 

@dataclass
class DataConfig:
    """Base configuration for data."""
    dataset_name: str
    dataset_path: str
    # image_size: int = 224
    # batch_size: int = 32
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    num_workers: int = 4
    pin_memory: bool = True

@dataclass
class ShapeSpec:
    """Simple dataclass to represent feature map shapes, replacing detectron2's ShapeSpec"""
    channels: int
    height: int = None
    width: int = None
    stride: int = None
    
@dataclass
class Config:
    """Base configuration class."""
    experiment_name: str
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    seed: int = 0
        
    @classmethod
    def from_yaml(cls, config_path: Path) -> 'Config':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        # Construct nested dataclass objects
        encoder_config = EncoderConfig(**config_dict.get('model', {}).get('encoder', {}))
        decoder_config = DecoderConfig(**config_dict.get('model', {}).get('decoder', {}))
        
        model_config = ModelConfig(
            encoder=encoder_config,
            decoder=decoder_config,
        )
        optimizer_config = OptimizerConfig(**config_dict.get('training', {}).get('optimizer', {}))
        
        scheduler_dict = config_dict.get('training', {}).get('scheduler', {})
        scheduler_config = SchedulerConfig(**scheduler_dict) if scheduler_dict else None
        
        if 'optimizer' in training_dict:
            del training_dict['optimizer']
        if 'scheduler' in training_dict:
            del training_dict['scheduler']
        
        training_config = TrainingConfig(
            **training_dict,
            optimizer=optimizer_config,
            scheduler=scheduler_config
        )

        data_config = DataConfig(**config_dict.get('data', {}))

        return cls(
            experiment_name=config_dict.get('experiment_name', 'default'),
            model=model_config,
            training=training_config,
            data=data_config,
            seed=config_dict.get('seed')
        )