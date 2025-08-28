from typing import Dict, Type

import torch.nn as nn

from config.base import DecoderConfig
from .base import BaseDecoder
from .linear_probing import LinearProbing
from .segformer import SegFormerHead
from .mask2former import Mask2FormerHead, OpenMask2FormerHead, MultitaskMask2FormerHead
# Registry of decoder implementations
DECODER_REGISTRY: Dict[str, Type[BaseDecoder]] = {
    'linear_probing': LinearProbing,
    'segformer_head': SegFormerHead,
    'mask2former_head': Mask2FormerHead,
    'open_mask2former_head': OpenMask2FormerHead,
    'multitask_mask2former_head': MultitaskMask2FormerHead,

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

__all__ = ['BaseDecoder', 'LinearProbing', 'SegFormerHead', 'Mask2FormerHead', 'OpenMask2FormerHead', 'MultitaskMask2FormerHead', 'create_decoder']