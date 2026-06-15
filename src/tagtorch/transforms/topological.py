"""
Topological Transform Classes

This module provides nn.Module-based transforms for topological data processing,
primarily focused on Euler Characteristic Transform (ECT) computations.

These classes wrap the functional API in tagtorch.transforms.functional
"""

import torch
import torch.nn as nn
from typing import Literal, Optional

from . import functional as F


class GrayscaleToECT(nn.Module):
    """
    Convert grayscale image to Euler Characteristic Transform.
    
    This transform computes the ECT, a topological descriptor that captures
    shape information in images. The ECT is computed by building a cubical
    complex from the binary image and computing Euler characteristics along
    multiple directions.
    
    Args:
        num_dirs (int): Number of directions for ECT computation. Default: 32
        num_thresh (int): Number of threshold bins. Default: 128
        foreground (str): 'B' for black or 'W' for white foreground. Default: 'B'
        epsilon (float): Threshold for foreground detection. Default: 0.01
        device (torch.device, optional): Device for computation
    
    Shape:
        - Input: (H, W) or (1, H, W)
        - Output: (1, num_thresh, num_dirs)
    
    Example:
        >>> transform = GrayscaleToECT(num_dirs=32, num_thresh=128)
        >>> img = torch.randn(28, 28)
        >>> ect = transform(img)
        >>> ect.shape
        torch.Size([1, 128, 32])
    """
    
    def __init__(
        self,
        num_dirs: int = 32,
        num_thresh: int = 128,
        foreground: Literal['B', 'W'] = 'B',
        epsilon: float = 0.01,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.num_dirs = num_dirs
        self.num_thresh = num_thresh
        self.foreground = foreground
        self.epsilon = epsilon
        self.device = device
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply ECT transform to grayscale image.
        
        Args:
            img (torch.Tensor): Grayscale image
        
        Returns:
            torch.Tensor: ECT representation
        """
        return F.grayscale_to_ect(
            img,
            num_dirs=self.num_dirs,
            num_thresh=self.num_thresh,
            foreground=self.foreground,
            epsilon=self.epsilon,
            device=self.device
        )
    
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"num_dirs={self.num_dirs}, "
                f"num_thresh={self.num_thresh}, "
                f"foreground='{self.foreground}', "
                f"epsilon={self.epsilon})")


class ColorToECT(nn.Module):
    """
    Convert color image to Euler Characteristic Transform.
    
    This transform can process color images in two ways:
    - 'grayscale': Convert to grayscale first, then compute ECT
    - 'channels': Compute ECT for each RGB channel separately
    
    Args:
        num_dirs (int): Number of directions for ECT computation. Default: 32
        num_thresh (int): Number of threshold bins. Default: 128
        mode (str): 'grayscale' or 'channels'. Default: 'grayscale'
        foreground (str): 'B' for black or 'W' for white foreground. Default: 'B'
        epsilon (float): Threshold for foreground detection. Default: 0.01
        device (torch.device, optional): Device for computation
    
    Shape:
        - Input: (3, H, W)
        - Output (grayscale mode): (1, num_thresh, num_dirs)
        - Output (channels mode): (3, num_thresh, num_dirs)
    
    Example:
        >>> transform = ColorToECT(mode='channels')
        >>> img = torch.randn(3, 32, 32)
        >>> ect = transform(img)
        >>> ect.shape
        torch.Size([3, 128, 32])
    """
    
    def __init__(
        self,
        num_dirs: int = 32,
        num_thresh: int = 128,
        mode: Literal['grayscale', 'channels'] = 'grayscale',
        foreground: Literal['B', 'W'] = 'B',
        epsilon: float = 0.01,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.num_dirs = num_dirs
        self.num_thresh = num_thresh
        self.mode = mode
        self.foreground = foreground
        self.epsilon = epsilon
        self.device = device
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply ECT transform to color image.
        
        Args:
            img (torch.Tensor): Color image with 3 channels
        
        Returns:
            torch.Tensor: ECT representation
        """
        return F.color_to_ect(
            img,
            num_dirs=self.num_dirs,
            num_thresh=self.num_thresh,
            mode=self.mode,
            foreground=self.foreground,
            epsilon=self.epsilon,
            device=self.device
        )
    
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"num_dirs={self.num_dirs}, "
                f"num_thresh={self.num_thresh}, "
                f"mode='{self.mode}', "
                f"foreground='{self.foreground}', "
                f"epsilon={self.epsilon})")


class ToECT(nn.Module):
    """
    Convert image to Euler Characteristic Transform (auto-detects color vs grayscale).
    
    This is a convenience transform that automatically handles both grayscale
    and color images based on the number of channels.
    
    Args:
        num_dirs (int): Number of directions for ECT computation. Default: 32
        num_thresh (int): Number of threshold bins. Default: 128
        mode (str, optional): For color images, 'grayscale' or 'channels'.
            If None, defaults to 'grayscale' for color images.
        foreground (str): 'B' for black or 'W' for white foreground. Default: 'B'
        epsilon (float): Threshold for foreground detection. Default: 0.01
        device (torch.device, optional): Device for computation
    
    Shape:
        - Input: (H, W), (1, H, W), or (3, H, W)
        - Output: Depends on input and mode
    
    Example:
        >>> transform = ToECT(num_dirs=64, num_thresh=256)
        >>> 
        >>> # Grayscale image
        >>> gray_img = torch.randn(1, 28, 28)
        >>> ect = transform(gray_img)
        >>> ect.shape
        torch.Size([1, 256, 64])
        >>> 
        >>> # Color image
        >>> color_img = torch.randn(3, 32, 32)
        >>> ect = transform(color_img)
        >>> ect.shape
        torch.Size([1, 256, 64])  # Converted to grayscale by default
    """
    
    def __init__(
        self,
        num_dirs: int = 32,
        num_thresh: int = 128,
        mode: Optional[Literal['grayscale', 'channels']] = None,
        foreground: Literal['B', 'W'] = 'B',
        epsilon: float = 0.01,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.num_dirs = num_dirs
        self.num_thresh = num_thresh
        self.mode = mode
        self.foreground = foreground
        self.epsilon = epsilon
        self.device = device
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply ECT transform to image (auto-detects type).
        
        Args:
            img (torch.Tensor): Input image
        
        Returns:
            torch.Tensor: ECT representation
        """
        return F.to_ect(
            img,
            num_dirs=self.num_dirs,
            num_thresh=self.num_thresh,
            mode=self.mode,
            foreground=self.foreground,
            epsilon=self.epsilon,
            device=self.device
        )
    
    def __repr__(self):
        mode_str = f"mode='{self.mode}'" if self.mode else "mode=None"
        return (f"{self.__class__.__name__}("
                f"num_dirs={self.num_dirs}, "
                f"num_thresh={self.num_thresh}, "
                f"{mode_str}, "
                f"foreground='{self.foreground}', "
                f"epsilon={self.epsilon})")
