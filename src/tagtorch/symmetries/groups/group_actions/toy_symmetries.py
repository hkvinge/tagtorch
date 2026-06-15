import torch
import random
from typing import List, Union
import numbers
import sys
import os

# Get the parent directory (groups/) and add to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from cyclic_groups import CyclicGroup
from group_actions.base import GroupAction

############### Toy Symmetries #######################

class VectorReversal(GroupAction):
    """
    The Z/2Z action that multiplies vectors by -1 or +1.

    This represents the cyclic group of order 2 acting on vectors by
    sign multiplication. One element is the identity (multiplication by +1)
    and the other is negation (multiplication by -1).

    The underlying group is CyclicGroup(2) = {0, 1}, which maps to
    the signs {+1, -1} for the action (0 → +1, 1 → -1).
    """

    def __init__(self):
        """Initialize the vector reversal action with Z/2Z."""
        super().__init__(CyclicGroup(2))

    def action(self, g: Union[int, List[int]], data: torch.Tensor) -> torch.Tensor:
        """
        Apply sign multiplication to data vectors.

        Parameters
        ----------
        g : int or List[int]
            Single element or list of elements from {0, 1} (CyclicGroup(2)).
            0 maps to +1 (identity), 1 maps to -1 (negation).
            If list, must have length equal to batch size.
        data : torch.Tensor
            Data tensor with batch dimension as first axis.

        Returns
        -------
        torch.Tensor
            Transformed data tensor with same shape as input.

        Raises
        ------
        ValueError
            If g contains invalid elements or data is None.
        """
        if data is None:
            raise ValueError("Data tensor cannot be None.")

        if isinstance(g, list):
            if len(g) != data.shape[0]:
                raise ValueError(f"Expected {data.shape[0]} group elements, got {len(g)}")
            # Validate all elements are in {0, 1} and convert to {+1, -1}
            signs = []
            for i, elem in enumerate(g):
                if elem not in (0, 1):
                    raise ValueError(f"Element {elem} at index {i} not in {{0, 1}}")
                signs.append(1 if elem == 0 else -1)
            g_tensor = torch.tensor(signs, dtype=data.dtype, device=data.device)
            result = data * g_tensor.view(-1, *[1] * (data.ndim - 1))
        else:
            if g not in (0, 1):
                raise ValueError(f"Expected g in {{0, 1}}, got {g}")
            sign = 1 if g == 0 else -1
            result = data * sign

        return result

    def orbit(self, data: torch.Tensor, num_samples: int = 2) -> List[torch.Tensor]:
        """
        Generate orbit under sign multiplication.

        For Z/2Z, the complete orbit has exactly 2 elements: [data, -data].

        Parameters
        ----------
        data : torch.Tensor
            Input data tensor.
        num_samples : int, optional
            Number of orbit elements to return. For Z/2Z, the complete
            orbit has 2 elements. If num_samples >= 2, returns complete orbit.

        Returns
        -------
        List[torch.Tensor]
            List of orbit elements. Complete orbit is [data, -data].
        """
        if num_samples <= 0:
            return []
        elif num_samples == 1:
            return [self.action(0, data)]  # Just identity (0 → +1)
        else:
            # Return complete orbit for Z/2Z: identity and negation
            return [self.action(0, data), self.action(1, data)]


############### Random scalar multiplication of vectors #######################

class ScalarMultiplication:
    """
    R* (non-zero reals) action via scalar multiplication.

    Multiplies vectors by random scalars sampled uniformly from [lambda1, lambda2].
    Note: This is not a group action unless lambda1 and lambda2 have the same sign.
    """
    finite = False
    trivial = False

    def __init__(self, lambda1: float = -1.0, lambda2: float = 1.0):
        if lambda1 >= lambda2:
            raise ValueError(f"Expected lambda1 < lambda2, got lambda1={lambda1}, lambda2={lambda2}")
        if lambda1 == 0 and lambda2 == 0:
            raise ValueError("Cannot sample from interval containing only 0")
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    @property
    def order(self):
        """Number of elements in R* (infinite)."""
        return float('inf')

    def sample(self) -> float:
        """Sample a random scalar from [lambda1, lambda2].

        Returns
        -------
        float
            A random float uniformly sampled from [lambda1, lambda2].
        """
        return random.uniform(self.lambda1, self.lambda2)

    def action(self, g: Union[float, List[float]], data: torch.Tensor) -> torch.Tensor:
        """Apply scalar multiplication to data vectors.

        Parameters
        ----------
        g : float or List[float]
            Scalar multiplier(s). If list, must have length equal to batch size.
        data : torch.Tensor
            Data tensor with batch dimension as first axis.

        Returns
        -------
        torch.Tensor
            Scaled data tensor with same shape as input.

        Raises
        ------
        ValueError
            If g contains invalid elements or data is None.
        """
        if data is None:
            raise ValueError("Data tensor cannot be None.")

        if isinstance(g, list):
            if len(g) != data.shape[0]:
                raise ValueError(f"Expected {data.shape[0]} group elements, got {len(g)}")
            g_tensor = torch.tensor(g, dtype=data.dtype, device=data.device)
            result = data * g_tensor.view(-1, *[1] * (data.ndim - 1))
        else:
            if not isinstance(g, numbers.Number):
                raise ValueError(f"Expected g to be a number, got {type(g)}")
            result = data * g

        return result

    def sample_and_act(self, data: torch.Tensor, all_same: bool = False) -> torch.Tensor:
        """Randomly sample scalar(s) and apply multiplication.

        Parameters
        ----------
        data : torch.Tensor
            Data tensor with batch dimension as first axis.
        all_same : bool, optional
            If True, apply same scalar to all batch items.
            If False, sample different scalars for each batch item.

        Returns
        -------
        torch.Tensor
            Scaled data tensor with same shape as input.
        """
        if all_same:
            group_elements = self.sample()
        else:
            group_elements = [self.sample() for _ in range(data.shape[0])]

        return self.action(group_elements, data)

    def orbit(self, data: torch.Tensor, num_samples: int = 100, all_same: bool = True) -> List[torch.Tensor]:
        """Generate orbit approximation via random scalar multiplication.

        Parameters
        ----------
        data : torch.Tensor
            Input data tensor.
        num_samples : int, optional
            Number of scaled versions to generate.
        all_same : bool, optional
            If True, apply same scalar to all batch items in each sample.
            If False, use different scalars for each batch item.

        Returns
        -------
        List[torch.Tensor]
            List of scaled data tensors approximating the orbit.
        """
        orbit_lst = []

        for _ in range(num_samples):
            orbit_lst.append(self.sample_and_act(data, all_same=all_same))

        return orbit_lst