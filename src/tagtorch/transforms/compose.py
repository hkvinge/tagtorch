"""
Transform Composition Utilities

This module provides utilities for composing and chaining transforms,
following the torchvision.transforms pattern.
"""

import torch
import torch.nn as nn
from typing import List, Callable
import random


class Compose(nn.Module):
    """
    Compose multiple transforms together.
    
    Transforms are applied in the order they are provided.
    
    Args:
        transforms (list): List of transform objects (nn.Module or callables)
    
    Example:
        >>> from tagtorch.transforms import Compose, RandomGroupAction, ToECT
        >>> from tagtorch.symmetries.groups.group_actions import DiscreteImageRotation
        >>> 
        >>> rotation = DiscreteImageRotation(order=4)
        >>> transform = Compose([
        ...     RandomGroupAction(rotation, p=0.5),
        ...     ToECT(num_dirs=32, num_thresh=128)
        ... ])
        >>> 
        >>> img = torch.randn(1, 28, 28)
        >>> output = transform(img)
    """
    
    def __init__(self, transforms: List[Callable]):
        super().__init__()
        self.transforms = nn.ModuleList([
            t if isinstance(t, nn.Module) else _CallableWrapper(t)
            for t in transforms
        ])
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply all transforms sequentially.
        
        Args:
            img (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Transformed tensor
        """
        for t in self.transforms:
            img = t(img)
        return img
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string


class RandomApply(nn.Module):
    """
    Apply a list of transforms with a given probability.
    
    When applied, all transforms in the list are executed sequentially.
    
    Args:
        transforms (list): List of transform objects
        p (float): Probability of applying the transforms. Default: 0.5
    
    Example:
        >>> from tagtorch.transforms import RandomApply, ToECT
        >>> 
        >>> transform = RandomApply([
        ...     ToECT(num_dirs=32, num_thresh=128)
        ... ], p=0.3)
        >>> 
        >>> img = torch.randn(1, 28, 28)
        >>> output = transform(img)  # 30% chance of applying ECT
    """
    
    def __init__(self, transforms: List[Callable], p: float = 0.5):
        super().__init__()
        self.transforms = nn.ModuleList([
            t if isinstance(t, nn.Module) else _CallableWrapper(t)
            for t in transforms
        ])
        self.p = p
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply transforms with probability p.
        
        Args:
            img (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Transformed or original tensor
        """
        if torch.rand(1).item() < self.p:
            for t in self.transforms:
                img = t(img)
        return img
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += f'\n    p={self.p}'
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string


class RandomChoice(nn.Module):
    """
    Apply one transform randomly chosen from a list.
    
    Args:
        transforms (list): List of transform objects
        p (list, optional): Probabilities for each transform. If None, uniform distribution.
    
    Example:
        >>> from tagtorch.transforms import RandomChoice, GrayscaleToECT, ColorToECT
        >>> 
        >>> transform = RandomChoice([
        ...     GrayscaleToECT(num_dirs=32),
        ...     GrayscaleToECT(num_dirs=64),
        ...     GrayscaleToECT(num_dirs=128),
        ... ])
        >>> 
        >>> img = torch.randn(1, 28, 28)
        >>> output = transform(img)  # Randomly uses 32, 64, or 128 directions
    """
    
    def __init__(self, transforms: List[Callable], p: List[float] = None):
        super().__init__()
        self.transforms = nn.ModuleList([
            t if isinstance(t, nn.Module) else _CallableWrapper(t)
            for t in transforms
        ])
        
        if p is not None:
            if len(p) != len(transforms):
                raise ValueError(f"Length of p ({len(p)}) must match number of transforms ({len(transforms)})")
            if not abs(sum(p) - 1.0) < 1e-6:
                raise ValueError(f"Probabilities must sum to 1.0, got {sum(p)}")
        self.p = p
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply one randomly chosen transform.
        
        Args:
            img (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Transformed tensor
        """
        if self.p is None:
            t = random.choice(self.transforms)
        else:
            t = random.choices(self.transforms, weights=self.p, k=1)[0]
        return t(img)
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        if self.p is not None:
            format_string += f'\n    p={self.p}'
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string


class RandomOrder(nn.Module):
    """
    Apply transforms in a random order.
    
    Args:
        transforms (list): List of transform objects
    
    Example:
        >>> from tagtorch.transforms import RandomOrder, RandomGroupAction
        >>> from tagtorch.symmetries.groups.group_actions import DiscreteImageRotation, ImageTranslation
        >>> 
        >>> rotation = DiscreteImageRotation(order=4)
        >>> translation = ImageTranslation(28, 28)
        >>> 
        >>> # Apply rotation and translation in random order
        >>> transform = RandomOrder([
        ...     RandomGroupAction(rotation),
        ...     RandomGroupAction(translation),
        ... ])
    """
    
    def __init__(self, transforms: List[Callable]):
        super().__init__()
        self.transforms = nn.ModuleList([
            t if isinstance(t, nn.Module) else _CallableWrapper(t)
            for t in transforms
        ])
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply transforms in random order.
        
        Args:
            img (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Transformed tensor
        """
        order = list(range(len(self.transforms)))
        random.shuffle(order)
        for i in order:
            img = self.transforms[i](img)
        return img
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string


class _CallableWrapper(nn.Module):
    """Internal wrapper for non-Module callables."""
    
    def __init__(self, func: Callable):
        super().__init__()
        self.func = func
    
    def forward(self, x):
        return self.func(x)
    
    def __repr__(self):
        return repr(self.func)
