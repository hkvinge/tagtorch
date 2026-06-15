"""
Functional API for TAGTorch transforms.

This module provides pure functional implementations of transforms.
These are stateless functions that can be composed in custom ways.

Functions here serve as the core implementation that the class-based
transforms (nn.Module wrappers) delegate to.
"""

import torch
import numpy as np
from typing import Literal, Tuple, Optional, Union

from demeter import euler
from demeter import directions as dirs


def grayscale_to_ect(
    img: torch.Tensor,
    num_dirs: int = 32,
    num_thresh: int = 128,
    foreground: Literal['B', 'W'] = 'B',
    epsilon: float = 0.01,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Convert grayscale image to Euler Characteristic Transform.
    
    Args:
        img (torch.Tensor): Grayscale image as 2D or 3D tensor (C, H, W) where C=1
        num_dirs (int): Number of directions for ECT computation
        num_thresh (int): Number of threshold bins
        foreground (str): 'B' for black, 'W' for white foreground
        epsilon (float): Threshold for foreground detection
        device (torch.device, optional): Device for computation
    
    Returns:
        torch.Tensor: ECT representation of shape (1, num_thresh, num_dirs)
    
    Example:
        >>> img = torch.randn(28, 28)
        >>> ect = grayscale_to_ect(img, num_dirs=32, num_thresh=128)
        >>> ect.shape
        torch.Size([1, 128, 32])
    """
    # Handle different input shapes
    if img.dim() == 3 and img.shape[0] == 1:
        this_img = img[0]
    elif img.dim() == 3 and img.shape[0] > 1:
        raise ValueError(f"Expected grayscale image with 1 channel, got {img.shape[0]} channels")
    elif img.dim() == 2:
        this_img = img
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got shape {img.shape}")
    
    # Threshold image to binary
    if foreground == 'W':
        # White = 1.0, pick pixels within epsilon of 1.0
        binary_img = torch.where(this_img > 1.0 - epsilon, 1.0, 0.0)
    elif foreground == 'B':
        # Black = 0.0, pick pixels within epsilon of 0.0
        binary_img = torch.where(this_img < 0.0 + epsilon, 1.0, 0.0)
    else:
        raise ValueError(f"foreground must be 'B' or 'W', got '{foreground}'")
    
    # This fixes the issue where if there are no foreground pixels, the ECT will be trivial and demeter may throw an error.
    # This does not fix the issue in demeter, but at least allows the code to run and return a trivial ECT instead of crashing.
    # TODO: fix the error in demeter itself so that it can handle empty foregrounds without crashing and contribute that to the demeter open source project.
    if min(binary_img.flatten()) == 0.0 and max(binary_img.flatten()) == 0.0:
        print("Warning: No foreground pixels detected in image. ECT will be trivial.")
        ect_tensor = torch.zeros((num_thresh, num_dirs), device=device)
        return ect_tensor.unsqueeze(0).float()
    
    # Convert to NumPy for demeter (TODO: make fully PyTorch in future)
    binary_np = binary_img.cpu().numpy()
    
    # Build cubical complex
    img_complex = euler.CubicalComplex(binary_np).complexify()
    
    # Compute ECT
    circle_dirs = dirs.regular_directions(num_dirs, dims=img_complex.ndim)
    ect_result = img_complex.ECT(circle_dirs, T=num_thresh)
    
    # Reshape and convert to tensor if needed
    if isinstance(ect_result, np.ndarray):
        ect_result = torch.from_numpy(ect_result)
    
    ect_tensor = ect_result.reshape(num_dirs, num_thresh).T
    
    # Add channel dimension and convert to float
    return ect_tensor.unsqueeze(0).float()


def color_to_ect(
    img: torch.Tensor,
    num_dirs: int = 32,
    num_thresh: int = 128,
    mode: Literal['grayscale', 'channels'] = 'grayscale',
    foreground: Literal['B', 'W'] = 'B',
    epsilon: float = 0.01,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Convert color image to Euler Characteristic Transform.
    
    Args:
        img (torch.Tensor): Color image as 3D tensor (3, H, W)
        num_dirs (int): Number of directions for ECT computation
        num_thresh (int): Number of threshold bins
        mode (str): 'grayscale' to convert to grayscale first, 'channels' to process each channel
        foreground (str): 'B' for black, 'W' for white foreground
        epsilon (float): Threshold for foreground detection
        device (torch.device, optional): Device for computation
    
    Returns:
        torch.Tensor: ECT representation
            - If mode='grayscale': shape (1, num_thresh, num_dirs)
            - If mode='channels': shape (3, num_thresh, num_dirs)
    
    Example:
        >>> img = torch.randn(3, 32, 32)
        >>> ect = color_to_ect(img, mode='channels')
        >>> ect.shape
        torch.Size([3, 128, 32])
    """
    if img.dim() != 3 or img.shape[0] != 3:
        raise ValueError(f"Expected color image with shape (3, H, W), got {img.shape}")
    
    if mode == 'grayscale':
        # Convert to grayscale using luminosity formula
        weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(3, 1, 1).to(img.device)
        gray_img = (img * weights).sum(dim=0, keepdim=False)
        return grayscale_to_ect(gray_img, num_dirs, num_thresh, foreground, epsilon, device)
    
    elif mode == 'channels':
        # Process each channel separately
        r_ect = grayscale_to_ect(img[0], num_dirs, num_thresh, foreground, epsilon, device)
        g_ect = grayscale_to_ect(img[1], num_dirs, num_thresh, foreground, epsilon, device)
        b_ect = grayscale_to_ect(img[2], num_dirs, num_thresh, foreground, epsilon, device)
        return torch.cat([r_ect, g_ect, b_ect], dim=0)
    
    else:
        raise ValueError(f"mode must be 'grayscale' or 'channels', got '{mode}'")


def to_ect(
    img: torch.Tensor,
    num_dirs: int = 32,
    num_thresh: int = 128,
    mode: Optional[Literal['grayscale', 'channels']] = None,
    foreground: Literal['B', 'W'] = 'B',
    epsilon: float = 0.01,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Convert image to Euler Characteristic Transform (auto-detects color vs grayscale).
    
    Args:
        img (torch.Tensor): Image tensor (C, H, W) where C is 1 or 3
        num_dirs (int): Number of directions for ECT computation
        num_thresh (int): Number of threshold bins
        mode (str, optional): For color images, 'grayscale' or 'channels'
        foreground (str): 'B' for black, 'W' for white foreground
        epsilon (float): Threshold for foreground detection
        device (torch.device, optional): Device for computation
    
    Returns:
        torch.Tensor: ECT representation
    
    Example:
        >>> gray_img = torch.randn(1, 28, 28)
        >>> ect = to_ect(gray_img)
        >>> ect.shape
        torch.Size([1, 128, 32])
        
        >>> color_img = torch.randn(3, 32, 32)
        >>> ect = to_ect(color_img, mode='channels')
        >>> ect.shape
        torch.Size([3, 128, 32])
    """
    if img.dim() == 2:
        # 2D tensor, treat as grayscale
        return grayscale_to_ect(img, num_dirs, num_thresh, foreground, epsilon, device)
    elif img.dim() == 3:
        if img.shape[0] == 1:
            # Grayscale with channel dimension
            return grayscale_to_ect(img, num_dirs, num_thresh, foreground, epsilon, device)
        elif img.shape[0] == 3:
            # Color image
            if mode is None:
                mode = 'grayscale'  # Default for color images
            return color_to_ect(img, num_dirs, num_thresh, mode, foreground, epsilon, device)
        else:
            raise ValueError(f"Expected 1 or 3 channels, got {img.shape[0]}")
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got {img.dim()}D")


def apply_group_action(
    img: torch.Tensor,
    group_action,
    g=None
) -> torch.Tensor:
    """
    Apply a symmetry group action to data.
    
    Args:
        img (torch.Tensor): Input data tensor
        group_action: A tagtorch GroupAction instance
        g (optional): Specific group element to apply. If None, samples randomly
    
    Returns:
        torch.Tensor: Transformed data
    
    Example:
        >>> from tagtorch.symmetries.groups.group_actions import DiscreteImageRotation
        >>> img = torch.randn(1, 28, 28)
        >>> rotation = DiscreteImageRotation(order=4)
        >>> rotated = apply_group_action(img, rotation)
    """
    if g is None:
        g = group_action.sample()
    return group_action.action(g, img)


def compose_transforms(img: torch.Tensor, transforms: list) -> torch.Tensor:
    """
    Apply a sequence of transforms to an image.
    
    Args:
        img (torch.Tensor): Input image
        transforms (list): List of callable transforms
    
    Returns:
        torch.Tensor: Transformed image
    """
    for transform in transforms:
        img = transform(img)
    return img
