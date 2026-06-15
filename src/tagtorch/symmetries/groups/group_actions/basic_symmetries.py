import torch
from typing import List, Union
import sys
import os

# Get the parent directory (groups/) and add to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from trivial_group import TrivialGroup
from group_actions.base import GroupAction

############### Trivial action #######################

class TrivialSymmetry(GroupAction):
    """
    The trivial group action where every group element acts as the identity.

    This implements the trivial group acting on data, where all group elements
    leave data unchanged. Useful as a baseline for equivariance testing.

    The underlying group is TrivialGroup with a single identity element.
    """

    def __init__(self):
        """Initialize the trivial action with the trivial group."""
        super().__init__(TrivialGroup())

    def action(self, g: Union[int, List[int]], data: torch.Tensor) -> torch.Tensor:
        """Apply the trivial action to data.

        The trivial action always returns data unchanged, regardless of the
        group element.

        Parameters
        ----------
        g : int or List[int]
            Group element(s). For trivial group, should be 1 (or list of 1's).
        data : torch.Tensor
            Input data tensor to transform.

        Returns
        -------
        torch.Tensor
            Unchanged data tensor (identity transformation).

        Raises
        ------
        ValueError
            If data is None.
        """
        if data is None:
            raise ValueError("Data tensor cannot be None.")

        # Trivial action: always return data unchanged
        return data

    def orbit(self, data: torch.Tensor, num_samples: int = 1) -> List[torch.Tensor]:
        """Calculate the orbit of data under the trivial group action.

        For the trivial action, the orbit always contains only the original data,
        since no transformation is ever applied.

        Parameters
        ----------
        data : torch.Tensor
            Input data tensor.
        num_samples : int, optional
            Number of orbit elements to return. For trivial action,
            this is ignored since the orbit always contains only the original data.

        Returns
        -------
        List[torch.Tensor]
            List containing only the original data tensor.
        """
        # For trivial symmetry, orbit is always just [data] regardless of num_samples
        return [data]
        
