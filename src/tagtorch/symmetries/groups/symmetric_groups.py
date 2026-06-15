import random
import itertools
from typing import List, Tuple

############### Symmetric group #######################

class SymmetricGroup:
    """
    The symmetric group S_n of all permutations of n elements.

    The symmetric group S_n consists of all bijections from {0, 1, ..., n-1}
    to itself. Elements are represented as tuples of length n, where the i-th
    entry gives the image of i under the permutation.

    For example, in S_3:
    - (0, 1, 2) is the identity
    - (1, 0, 2) swaps 0 and 1
    - (1, 2, 0) is a 3-cycle

    Parameters
    ----------
    n : int
        The number of elements being permuted (n >= 1).

    Attributes
    ----------
    n : int
        The degree of the symmetric group.
    order : int
        The number of elements in the group (n!).
    abelian : bool
        True if n <= 2, False otherwise.
    """

    def __init__(self, n: int):
        if not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer")
        self.n = n
        self.order = self._factorial(n)
        self.abelian = (n <= 2)

    @staticmethod
    def _factorial(n: int) -> int:
        """Compute n! efficiently."""
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    def enumerate_elements(self) -> List[Tuple[int, ...]]:
        """Enumerate all permutations in S_n.

        Returns
        -------
        List[Tuple[int, ...]]
            List of all permutations as tuples.

        Warning
        -------
        This generates n! permutations, which becomes very large quickly.
        For n=10, this is over 3.6 million permutations.
        """
        if self.n > 9:
            raise NotImplementedError(f"To return a full enumeration of S(n), n "
                                      "should be less than 10, got {self.n}")

        return [tuple(p) for p in itertools.permutations(range(self.n))]

    def sample(self) -> Tuple[int, ...]:
        """Sample a uniformly random permutation from S_n.

        Returns
        -------
        Tuple[int, ...]
            A random permutation of (0, 1, ..., n-1).
        """
        perm = list(range(self.n))
        random.shuffle(perm)
        return tuple(perm)

    def compose(self, g1: Tuple[int, ...], g2: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compose two permutations (apply g1 then g2).

        The composition is defined so that compose(g1, g2)(x) = g2(g1(x)).

        Parameters
        ----------
        g1 : Tuple[int, ...]
            First permutation.
        g2 : Tuple[int, ...]
            Second permutation.

        Returns
        -------
        Tuple[int, ...]
            The composition g2 ∘ g1.
        """
        if len(g1) != self.n or len(g2) != self.n:
            raise ValueError(f"Permutations must have length {self.n}")
        return tuple(g2[g1[i]] for i in range(self.n))

    def inverse(self, g: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute the inverse of a permutation.

        Parameters
        ----------
        g : Tuple[int, ...]
            A permutation.

        Returns
        -------
        Tuple[int, ...]
            The inverse permutation such that compose(g, inverse(g)) = identity().
        """
        if len(g) != self.n:
            raise ValueError(f"Permutation must have length {self.n}")
        inv = [0] * self.n
        for i, val in enumerate(g):
            inv[val] = i
        return tuple(inv)

    def identity(self) -> Tuple[int, ...]:
        """Return the identity permutation.

        Returns
        -------
        Tuple[int, ...]
            The identity permutation (0, 1, ..., n-1).
        """
        return tuple(range(self.n))

    def cycle_decomposition(self, g: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """Compute the cycle decomposition of a permutation.

        Parameters
        ----------
        g : Tuple[int, ...]
            A permutation.

        Returns
        -------
        List[Tuple[int, ...]]
            List of disjoint cycles. Fixed points are omitted.

        Examples
        --------
        >>> S3 = SymmetricGroup(3)
        >>> S3.cycle_decomposition((1, 2, 0))
        [(0, 1, 2)]
        >>> S3.cycle_decomposition((1, 0, 2))
        [(0, 1)]
        """
        if len(g) != self.n:
            raise ValueError(f"Permutation must have length {self.n}")

        visited = [False] * self.n
        cycles = []

        for start in range(self.n):
            if visited[start]:
                continue

            cycle = []
            current = start
            while not visited[current]:
                visited[current] = True
                cycle.append(current)
                current = g[current]

            # Only include non-trivial cycles (length > 1)
            if len(cycle) > 1:
                cycles.append(tuple(cycle))

        return cycles

