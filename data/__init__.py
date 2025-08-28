"""
Data loading and processing functionality.

This module provides dataset implementations, data loaders, and related utilities
for training and evaluating models on various datasets including R2S100K.
"""

# Import and expose key functions
from .dataset import get_dataset
from .loader import get_data_loaders

# Re-export R2S100K dataset class directly
from .r2s100k.dataset import R2S100k
from .bdd100k.dataset import SemsegBDD100k
from .combined.dataset import BDD100kR2S100k
# Add other datasets as they're implemented
# from .other_dataset.dataset import OtherDataset

# Optional: Import and expose transforms if you have any
# from .transforms import get_transforms

# Define __all__ to control what's imported with "from data import *"
__all__ = [
    # Core functionality
    'get_dataset',
    'get_data_loaders',
    
    # Dataset classes
    'R2S100k',
    'SemsegBDD100k',
    'BDD100kR2S100k',
    # Add other exports as needed
]

# Optional: You can add some module-level information