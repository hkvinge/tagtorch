import torch
import random
import math
import itertools
from torchvision import transforms
from typing import List, Union, Tuple
import warnings
import sys
import os

# Get the parent directory (groups/) and add to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from cyclic_groups import CyclicGroup
from so_n import CircleGroup
from group_actions.base import GroupAction


############### Image Symmetries #######################

class ProductGroup:
    """
    Direct product of two groups G × H.

    Elements are pairs (g, h) where g ∈ G and h ∈ H.
    """
    def __init__(self, group1, group2):
        self.group1 = group1
        self.group2 = group2
        if isinstance(group1.order, int) and isinstance(group2.order, int):
            self.order = group1.order * group2.order
        else:
            self.order = float('inf')

    def sample(self):
        """Sample a random element from G × H."""
        return (self.group1.sample(), self.group2.sample())

    def enumerate_elements(self):
        """Enumerate all elements of G × H (finite groups only)."""
        if not isinstance(self.order, int):
            raise NotImplementedError("Cannot enumerate infinite group")
        return [(g1, g2) for g1 in self.group1.enumerate_elements()
                for g2 in self.group2.enumerate_elements()]


class ImageTranslation(GroupAction):
    """
    Discrete translations of an image with wraparound.

    Applies vertical and horizontal translations where pixels wrap around
    the image boundaries (cyclic boundary conditions).

    The underlying group is Z/h × Z/w (product of cyclic groups), where
    h is the image height and w is the image width.

    Parameters
    ----------
    image_height : int
        Height of the image.
    image_width : int
        Width of the image.
    """

    def __init__(self, image_height: int, image_width: int):
        """
        Initialize image translation action.

        Parameters
        ----------
        image_height : int
            Height of the image.
        image_width : int
            Width of the image.
        """
        # Create product group Z/h × Z/w
        vertical_group = CyclicGroup(image_height)
        horizontal_group = CyclicGroup(image_width)
        super().__init__(ProductGroup(vertical_group, horizontal_group))

        self.image_height = image_height
        self.image_width = image_width

    def action(self, g: Union[Tuple[int, int], List[Tuple[int, int]]],
               image: torch.Tensor) -> torch.Tensor:
        """
        Apply translation to an image with wraparound.

        Parameters
        ----------
        g : Tuple[int, int] or List[Tuple[int, int]]
            Translation vector(s) as (vertical_shift, horizontal_shift).
        image : torch.Tensor
            Image tensor of shape (batch_size, ..., height, width) or (..., height, width).

        Returns
        -------
        torch.Tensor
            Translated image(s) with same shape as input.

        Raises
        ------
        ValueError
            If dimensions don't match or invalid group elements.
        """
        if image is None:
            raise ValueError("Image tensor cannot be None.")

        # Validate image dimensions match constructor
        if image.shape[-2] != self.image_height or image.shape[-1] != self.image_width:
            raise ValueError(f"Image shape {image.shape[-2:]}, expected ({self.image_height}, {self.image_width})")

        if isinstance(g, list):
            if len(g) != image.shape[0]:
                raise ValueError(f"Expected {image.shape[0]} group elements, got {len(g)}")
            result = torch.stack([torch.roll(image[i], shifts=(dy, dx), dims=(-2, -1))
                                 for i, (dy, dx) in enumerate(g)])
        else:
            if not (isinstance(g, tuple) and len(g) == 2 and all(isinstance(x, int) for x in g)):
                raise ValueError(f"Expected tuple of two integers, got {g}")
            result = torch.roll(image, shifts=g, dims=(-2, -1))

        return result


class ContinuousImageRotation(GroupAction):
    """
    Continuous rotation group SO(2) acting on images.

    Rotates images by random angles sampled uniformly from [0, 2π) radians.
    The underlying group is CircleGroup (circle group S¹ ≅ SO(2)).
    """

    def __init__(self):
        """Initialize continuous rotation action with CircleGroup."""
        super().__init__(CircleGroup())

    def action(self, g: Union[float, List[float]], image: torch.Tensor) -> torch.Tensor:
        """
        Apply rotation(s) to image(s).

        Parameters
        ----------
        g : float or List[float]
            Rotation angle(s) in radians from CircleGroup.
        image : torch.Tensor
            Input image tensor.

        Returns
        -------
        torch.Tensor
            Rotated image tensor with same shape as input.

        Raises
        ------
        ValueError
            If input validation fails.
        """
        if image is None:
            raise ValueError("Image tensor cannot be None.")

        rotate = transforms.functional.rotate

        if isinstance(g, list):
            if len(g) != image.shape[0]:
                raise ValueError(f"Expected {image.shape[0]} group elements, got {len(g)}")
            # Convert radians to degrees for torchvision
            angles_deg = [angle * 180.0 / math.pi for angle in g]
            result = torch.stack([rotate(image[k], angle=angle) for k, angle in enumerate(angles_deg)])
        else:
            if not isinstance(g, (float, int)):
                raise ValueError(f"Expected g to be a number, got {type(g)}")
            # Convert radians to degrees for torchvision
            angle_deg = float(g) * 180.0 / math.pi
            result = rotate(image, angle=angle_deg)

        return result


class DiscreteImageRotation(GroupAction):
    """
    Discrete rotation group C_n acting on images.

    Rotates images by multiples of 360/n degrees, where n is the order.
    Default is C_4 (90-degree rotations).

    The underlying group is CyclicGroup(n), where element k corresponds
    to a rotation by k * (360/n) degrees.

    Parameters
    ----------
    order : int, optional
        The order of the cyclic group. Default is 4 (90-degree rotations).
    """

    def __init__(self, order: int = 4):
        """
        Initialize discrete rotation action.

        Parameters
        ----------
        order : int, optional
            The order of the cyclic group (number of distinct rotations).
            Default is 4.
        """
        if order <= 0:
            raise ValueError(f"Order must be positive, got {order}")
        super().__init__(CyclicGroup(order))

    def action(self, g: Union[int, List[int]], image: torch.Tensor) -> torch.Tensor:
        """
        Apply discrete rotation(s) to image(s).

        Parameters
        ----------
        g : int or List[int]
            Rotation index(es) in range [0, order).
        image : torch.Tensor
            Input image tensor.

        Returns
        -------
        torch.Tensor
            Rotated image tensor with same shape as input.

        Raises
        ------
        ValueError
            If group elements are invalid or out of range.
        """
        if image is None:
            raise ValueError("Image tensor cannot be None.")

        rotate = transforms.functional.rotate
        order = self.group.order

        if isinstance(g, list):
            if len(g) != image.shape[0]:
                raise ValueError(f"Expected {image.shape[0]} group elements, got {len(g)}")
            # Validate all elements are in range
            for i, elem in enumerate(g):
                if not isinstance(elem, int) or not (0 <= elem < order):
                    raise ValueError(f"Group element {elem} at index {i} out of range [0, {order})")
            angle_lst = [360.0 * (i / order) for i in g]
            result = torch.stack([rotate(image[k], angle=angle) for k, angle in enumerate(angle_lst)])
        else:
            if not isinstance(g, (float, int)):
                raise ValueError(f"Expected g to be an int, got {type(g)}")
            if not (0 <= g < order):
                raise ValueError(f"Group element {g} out of range [0, {order})")
            angle = 360.0 * (g / order)
            result = rotate(image, angle=angle)

        return result