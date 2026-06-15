"""
TAGTorch Transforms Module

This module provides PyTorch-style transforms for topological and symmetry-based
data augmentation, following the design patterns of torchvision.transforms.

The module is organized into:
- Topological transforms (ECT, persistent homology)
- Symmetry transforms (group actions as augmentations)
- Composition utilities (Compose, RandomApply, etc.)

Both class-based (nn.Module) and functional APIs are provided.

Note: All transforms are nn.Module subclasses, so you can also use torch.nn.Sequential
for simple composition instead of our Compose class:

    # Using tagtorch.transforms.Compose
    from tagtorch.transforms import Compose, ToECT
    transform = Compose([ToECT()])
    
    # Or using torch.nn.Sequential (equivalent)
    import torch.nn as nn
    transform = nn.Sequential(ToECT())
"""

from .topological import (
    GrayscaleToECT,
    ColorToECT,
    ToECT,
)

from .symmetry import (
    RandomGroupAction,
    GroupActionAugmentation,
    EquivariantAugmentation,
)

from .compose import (
    Compose,
    RandomApply,
    RandomChoice,
    RandomOrder,
)

# Import functional API
from . import functional as F

__all__ = [
    # Topological transforms
    'GrayscaleToECT',
    'ColorToECT',
    'ToECT',
    
    # Symmetry transforms
    'RandomGroupAction',
    'GroupActionAugmentation',
    
    # Composition
    'Compose',
    'RandomApply',
    'RandomChoice',
    
    # Functional API
    'F',
]
