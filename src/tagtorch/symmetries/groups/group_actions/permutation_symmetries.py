import torch
from typing import List, Union, Tuple
import sys
import os

# Get the parent directory (groups/) and add to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from symmetric_groups import SymmetricGroup
from group_actions.base import GroupAction

############### Permutation action on vectors #######################

class ColumnPermutation(GroupAction):
    """
    Symmetric group S_n acting by permuting vector elements.

    This action permutes the elements along the second dimension (columns)
    of a data tensor. Useful for testing permutation equivariance.

    Parameters
    ----------
    n : int
        The number of elements to permute (degree of the symmetric group).

    Examples
    --------
    >>> perm_action = ColumnPermutation(5)
    >>> data = torch.randn(10, 5, 3)  # 10 samples, 5 features, 3 additional dims
    >>> transformed = perm_action.sample_and_act(data)  # Permutes the 5 features
    """

    def __init__(self, n: int):
        """
        Initialize the column permutation action.

        Parameters
        ----------
        n : int
            The number of elements to permute.
        """
        super().__init__(SymmetricGroup(n))
        self.n = n

    def action(self, perm: Union[Tuple[int, ...], List[Tuple[int, ...]]],
               data: torch.Tensor) -> torch.Tensor:
        """
        Apply permutation(s) to data columns.

        Permutes elements along dimension 1 (the column/feature dimension).

        Parameters
        ----------
        perm : Tuple[int, ...] or List[Tuple[int, ...]]
            Single permutation as tuple, or list of permutations (one per batch item).
            Each permutation is a tuple of length n representing a bijection.
        data : torch.Tensor
            Data tensor with shape (batch_size, n, ...) where n is the number
            of elements to permute.

        Returns
        -------
        torch.Tensor
            Permuted data tensor with same shape as input.

        Raises
        ------
        ValueError
            If dimensions don't match or data is None.
        """
        if data is None:
            raise ValueError("Data tensor cannot be None.")

        if data.shape[1] != self.n:
            raise ValueError(f"Data has {data.shape[1]} columns, expected {self.n}")

        if isinstance(perm, list):
            # Batch case: list of permutations, one per batch item
            if len(perm) != data.shape[0]:
                raise ValueError(f"Expected {data.shape[0]} permutations, got {len(perm)}")

            result = []
            for i, p in enumerate(perm):
                indices = torch.tensor(p, dtype=torch.long, device=data.device)
                result.append(data[i, indices, ...])
            return torch.stack(result)
        else:
            # Single permutation applied to all batch items
            indices = torch.tensor(perm, dtype=torch.long, device=data.device)
            return data[:, indices, ...]

    def orbit(self, data: torch.Tensor, num_samples: int = 100) -> List[torch.Tensor]:
        """
        Calculate the orbit of data under column permutations.

        For small symmetric groups (n <= 5), you can set num_samples = None
        to get the complete orbit. For larger groups, sampling is recommended
        as the orbit size is n!.

        Parameters
        ----------
        data : torch.Tensor
            Input data tensor with shape (..., n, ...).
        num_samples : int or None, optional
            Number of orbit elements to generate. If None and the group is
            finite, returns the complete orbit. Default is 100.

        Returns
        -------
        List[torch.Tensor]
            List of permuted data tensors.
        """
        if num_samples is None:
            # Generate complete orbit
            if self.finite:
                return [self.action(g, data) for g in self.group.enumerate_elements()]
            else:
                raise ValueError("Cannot enumerate complete orbit of infinite group")
        else:
            # Sample orbit
            return [self.action(self.sample(), data) for _ in range(num_samples)]


# Backwards compatibility alias
column_permutation = ColumnPermutation