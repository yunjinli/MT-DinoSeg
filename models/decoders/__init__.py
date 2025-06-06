from typing import Dict, Type

import torch.nn as nn

from config.base import DecoderConfig
from .base import BaseDecoder
from .linear_probing import LinearProbing
from .segformer import SegFormerHead
# Registry of decoder implementations
DECODER_REGISTRY: Dict[str, Type[BaseDecoder]] = {
    'linear_probing': LinearProbing,
    'segformer_head': SegFormerHead
}

def create_decoder(config: DecoderConfig) -> BaseDecoder:
    """
    Create a decoder model from configuration.
    
    Args:
        config: Decoder configuration
        
    Returns:
        Initialized decoder model
    """
    decoder_class = DECODER_REGISTRY.get(config.name)
    if decoder_class is None:
        raise ValueError(f"Unknown decoder: {config.name}")
    
    return decoder_class(config)

__all__ = ['BaseDecoder', 'LinearProbing', 'SegFormerHead', 'create_decoder']