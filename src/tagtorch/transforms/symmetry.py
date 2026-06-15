"""
Symmetry Transform Classes

This module provides nn.Module-based transforms that wrap group actions
for use as data augmentation in PyTorch pipelines.

These transforms integrate tagtorch's symmetry group actions into the
standard torchvision-style transform API.
"""

import torch
import torch.nn as nn
from typing import Optional

from . import functional as F


class RandomGroupAction(nn.Module):
    """
    Randomly apply a symmetry group action to data.
    
    This transform samples a random element from the specified group
    and applies its action to the input tensor. Can be used for
    data augmentation based on mathematical symmetries.
    
    Args:
        group_action: A tagtorch GroupAction instance (from tagtorch.symmetries)
        p (float): Probability of applying the transform. Default: 1.0
    
    Example:
        >>> from tagtorch.symmetries.groups.group_actions import DiscreteImageRotation
        >>> from tagtorch.transforms import RandomGroupAction
        >>> 
        >>> rotation = DiscreteImageRotation(order=4)
        >>> transform = RandomGroupAction(rotation, p=0.5)
        >>> 
        >>> img = torch.randn(1, 28, 28)
        >>> augmented = transform(img)
    """
    
    def __init__(self, group_action, p: float = 1.0):
        super().__init__()
        self.group_action = group_action
        self.p = p
        
        # Store group info for repr
        self._group_name = group_action.__class__.__name__
        self._group_order = getattr(group_action, 'order', 'infinite')
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply group action with probability p.
        
        Args:
            img (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Transformed tensor
        """
        if torch.rand(1).item() < self.p:
            return F.apply_group_action(img, self.group_action)
        return img
    
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"group_action={self._group_name}, "
                f"order={self._group_order}, "
                f"p={self.p})")


class GroupActionAugmentation(nn.Module):
    """
    Apply a group action augmentation with more control options.
    
    This is a more sophisticated version of RandomGroupAction that allows:
    - Applying specific group elements instead of random sampling
    - Batched application with different elements per item
    - Enumeration over all group elements (for finite groups)
    
    Args:
        group_action: A tagtorch GroupAction instance
        mode (str): How to apply the action:
            - 'random': Sample random group element (default)
            - 'identity': Apply identity (no-op, useful for testing)
            - 'all_same': Sample one element, apply to all batch items
        p (float): Probability of applying the transform. Default: 1.0
    
    Example:
        >>> from tagtorch.symmetries.groups.group_actions import ContinuousImageRotation
        >>> 
        >>> rotation = ContinuousImageRotation()
        >>> transform = GroupActionAugmentation(rotation, mode='random')
        >>> 
        >>> batch = torch.randn(8, 1, 28, 28)
        >>> augmented = transform(batch)  # Different rotation per image
    """
    
    def __init__(
        self,
        group_action,
        mode: str = 'random',
        p: float = 1.0
    ):
        super().__init__()
        self.group_action = group_action
        self.mode = mode
        self.p = p
        
        # Validate mode
        valid_modes = ['random', 'identity', 'all_same']
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got '{mode}'")
        
        # Store group info for repr
        self._group_name = group_action.__class__.__name__
        self._group_order = getattr(group_action, 'order', 'infinite')
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply group action based on mode and probability.
        
        Args:
            img (torch.Tensor): Input tensor (can be batched)
        
        Returns:
            torch.Tensor: Transformed tensor
        """
        if torch.rand(1).item() >= self.p:
            return img
        
        if self.mode == 'identity':
            # Apply identity element (no transformation)
            g = self.group_action.group.identity()
            return self.group_action.action(g, img)
        
        elif self.mode == 'all_same':
            # Sample one element, apply to all items
            g = self.group_action.sample()
            return self.group_action.action(g, img)
        
        else:  # mode == 'random'
            # For batched inputs, apply different transformations
            if img.dim() >= 3:
                # Batched input - apply different action to each item
                batch_size = img.shape[0]
                g_list = [self.group_action.sample() for _ in range(batch_size)]
                return self.group_action.action(g_list, img)
            else:
                # Single input
                return F.apply_group_action(img, self.group_action)
    
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"group_action={self._group_name}, "
                f"order={self._group_order}, "
                f"mode='{self.mode}', "
                f"p={self.p})")


class EquivariantAugmentation(nn.Module):
    """
    Apply augmentation while maintaining equivariance properties.
    
    This transform applies the same group element to both input and target,
    which is useful when training equivariant models where the target
    should transform consistently with the input.
    
    Args:
        group_action: A tagtorch GroupAction instance
        p (float): Probability of applying the transform. Default: 1.0
    
    Example:
        >>> from tagtorch.symmetries.groups.group_actions import DiscreteImageRotation
        >>> 
        >>> rotation = DiscreteImageRotation(order=4)
        >>> transform = EquivariantAugmentation(rotation)
        >>> 
        >>> # For tasks where both input and target should rotate together
        >>> img, target = torch.randn(1, 28, 28), torch.randn(1, 28, 28)
        >>> aug_img, aug_target = transform(img, target)
    """
    
    def __init__(self, group_action, p: float = 1.0):
        super().__init__()
        self.group_action = group_action
        self.p = p
        self._group_name = group_action.__class__.__name__
    
    def forward(self, img: torch.Tensor, target: Optional[torch.Tensor] = None):
        """
        Apply same group element to both input and target.
        
        Args:
            img (torch.Tensor): Input tensor
            target (torch.Tensor, optional): Target tensor
        
        Returns:
            tuple or torch.Tensor: (transformed_img, transformed_target) if target provided,
                                   otherwise just transformed_img
        """
        if torch.rand(1).item() >= self.p:
            return (img, target) if target is not None else img
        
        # Sample one group element
        g = self.group_action.sample()
        
        # Apply to both input and target
        aug_img = self.group_action.action(g, img)
        
        if target is not None:
            aug_target = self.group_action.action(g, target)
            return aug_img, aug_target
        
        return aug_img
    
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"group_action={self._group_name}, "
                f"p={self.p})")
