import torch
from typing import List, Union

############### Base Group Action Class #######################

class GroupAction:
    """
    Base class for group actions on data.

    This class provides a generic implementation for group actions by separating
    the group structure (sampling, composition, inverse) from the action itself
    (how group elements transform data).

    Subclasses only need to:
    1. Initialize with an appropriate group (TrivialGroup, CyclicGroup, etc.)
    2. Implement the action() method to define how group elements transform data

    All sampling, orbit generation, and batching logic is handled by this base class.

    Parameters
    ----------
    group : Group
        The group that acts on the data. Should have methods:
        - sample(): return a random group element
        - enumerate_elements(): return all elements (for finite groups)
        - order: property giving the number of elements

    Attributes
    ----------
    group : Group
        The underlying group structure.
    """

    def __init__(self, group):
        """
        Initialize the group action.

        Parameters
        ----------
        group : Group
            The group that acts on the data (TrivialGroup, CyclicGroup, SymmetricGroup, etc.)
        """
        self.group = group

    @property
    def order(self):
        """
        Number of elements in the group.

        Returns
        -------
        int or float
            The order of the group. Returns float('inf') for infinite groups.
        """
        return self.group.order

    @property
    def finite(self):
        """
        Whether the group is finite.

        Returns
        -------
        bool
            True if the group has finitely many elements.
        """
        return isinstance(self.group.order, int)

    @property
    def trivial(self):
        """
        Whether this is the trivial action.

        Returns
        -------
        bool
            True if the group has only one element (the identity).
        """
        return self.group.order == 1

    def sample(self):
        """
        Sample a random group element.

        Returns
        -------
        group element
            A randomly sampled element from the group.
        """
        return self.group.sample()

    def action(self, g, data: torch.Tensor) -> torch.Tensor:
        """
        Apply group element(s) to data.

        This method MUST be implemented by subclasses to define the specific
        transformation that group elements apply to data.

        Parameters
        ----------
        g : group element or List[group element]
            Single group element or list of group elements (one per batch item).
        data : torch.Tensor
            Input data tensor to transform.

        Returns
        -------
        torch.Tensor
            Transformed data tensor with same shape as input.

        Raises
        ------
        NotImplementedError
            If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement action() method")

    def sample_and_act(self, data: torch.Tensor, all_same: bool = False) -> torch.Tensor:
        """
        Randomly sample group element(s) and apply to data.

        This generic implementation works for any group action.

        Parameters
        ----------
        data : torch.Tensor
            Input data tensor with batch dimension as first axis.
        all_same : bool, optional
            If True, apply same group element to all batch items.
            If False, sample different group elements for each batch item.
            Default is False.

        Returns
        -------
        torch.Tensor
            Transformed data tensor with same shape as input.
        """
        if all_same:
            g = self.sample()
            return self.action(g, data)
        else:
            g_list = [self.sample() for _ in range(data.shape[0])]
            return self.action(g_list, data)

    def orbit(self, data: torch.Tensor, num_samples: int = 100) -> List[torch.Tensor]:
        """
        Generate orbit of data under the group action.

        For finite groups with num_samples >= order, returns the complete orbit
        deterministically. Otherwise, samples random group elements.

        Parameters
        ----------
        data : torch.Tensor
            Input data tensor.
        num_samples : int, optional
            Number of orbit elements to generate. Default is 100.

        Returns
        -------
        List[torch.Tensor]
            List of transformed data tensors under the group action.
        """
        if self.finite and num_samples >= self.order:
            # Return complete orbit for finite groups
            return [self.action(g, data) for g in self.group.enumerate_elements()]
        else:
            # Sample orbit for infinite groups or when num_samples < order
            return [self.action(self.sample(), data) for _ in range(num_samples)]
