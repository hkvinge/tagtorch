import torch
import random
import math
from typing import List

############### O(n) - Orthogonal Groups #######################

class O:
    """
    The orthogonal group O(n) of rotations and reflections in n dimensions.

    O(n) consists of all n x n orthogonal matrices (both rotations and reflections).
    Elements satisfy R @ R.T = I. They are also characterized by the fact that
    det(R) = ±1.

    Properties:
    - Continuous Lie group 
    - Non-abelian for n ≥ 2
    - Contains SO(n) as an index-2 subgroup (det = +1 elements)
    - O(n) is the union of SO(n) and all elements with det = -1

    Elements are represented as n x n torch.Tensor matrices satisfying:
    - R @ R.T = I (orthogonality)
    - det(R) = ±1 (orthogonal, includes reflections)

    Parameters
    ----------
    n : int
        Dimension of the orthogonal group (n >= 2).

    Examples
    --------
    >>> # 3D orthogonal group (rotations and reflections)
    >>> O3 = O(3)
    >>> R = O3.sample()
    >>> R.shape
    torch.Size([3, 3])
    >>> torch.allclose(R @ R.T, torch.eye(3))
    True
    >>> det = torch.det(R)
    >>> det in [1.0, -1.0]  # Can be rotation or reflection
    True

    Notes
    -----
    For pure rotations (det = +1 only), use SO(n) instead.

    The relationship between O(n) and SO(n):
    - O(n) contains all orthogonal transformations
    - SO(n) ⊂ O(n) consists of only rotations (det = +1)

    See Also
    --------
    SO : Special orthogonal group SO(n) (rotations only)
    """

    def __init__(self, n: int):
        """
        Initialize O(n) group.

        Parameters
        ----------
        n : int
            Dimension (must be at least 2).

        Raises
        ------
        ValueError
            If n < 2.
        """
        if not isinstance(n, int) or n < 2:
            raise ValueError(f"Dimension must be an integer >= 2, got {n}")

        self.n = n
        self.order = float('inf')
        self.abelian = False  # O(n) is non-abelian for all n ≥ 2

    def enumerate_elements(self) -> List[torch.Tensor]:
        """
        Enumerate all elements in the group.

        Raises
        ------
        NotImplementedError
            O(n) is a continuous group and cannot be enumerated.
        """
        raise NotImplementedError(f"Cannot enumerate elements of continuous group O({self.n}).")

    def sample(self) -> torch.Tensor:
        """
        Sample from O(n) according to Haar measure.

        Uses QR decomposition

        Returns
        -------
        torch.Tensor
            An n x n orthogonal matrix.

        Examples
        --------
        >>> group = O(3)
        >>> R = group.sample()  
        """
        # Start with QR decomposition to get orthogonal matrix
        random_matrix = torch.randn(self.n, self.n)
        Q, R = torch.linalg.qr(random_matrix)

        # Ensure positive diagonal of R for uniform distribution
        d = torch.diagonal(R)
        Q = Q @ torch.diag(d / torch.abs(d))

        return Q

    def compose(self, g1: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
        """
        Compose two orthogonal matrices via matrix multiplication.

        Parameters
        ----------
        g1 : torch.Tensor
            First orthogonal matrix (n x n).
        g2 : torch.Tensor
            Second orthogonal matrix (n x n).

        Returns
        -------
        torch.Tensor
            The composed matrix g2 @ g1.

        Examples
        --------
        >>> group = O(3)
        >>> R1, R2 = group.sample(), group.sample()
        >>> R3 = group.compose(R1, R2)
        >>> torch.allclose(R3, R2 @ R1)
        True
        """
        if g1.shape != (self.n, self.n) or g2.shape != (self.n, self.n):
            raise ValueError(f"Expected {self.n}×{self.n} matrices")
        return g2 @ g1

    def inverse(self, g: torch.Tensor) -> torch.Tensor:
        """
        Compute the inverse of an orthogonal matrix.

        For orthogonal matrices, the inverse is simply the transpose.

        Parameters
        ----------
        g : torch.Tensor
            Orthogonal matrix (n x n).

        Returns
        -------
        torch.Tensor
            The inverse matrix g^(-1) = g.T.

        Examples
        --------
        >>> group = O(3)
        >>> R = group.sample()
        >>> R_inv = group.inverse(R)
        >>> torch.allclose(group.compose(R, R_inv), group.identity())
        True
        """
        if g.shape != (self.n, self.n):
            raise ValueError(f"Expected {self.n} x {self.n} matrix")
        return g.T

    def identity(self) -> torch.Tensor:
        """
        Return the identity matrix.

        Returns
        -------
        torch.Tensor
            The n x n identity matrix.

        Examples
        --------
        >>> group = O(3)
        >>> I = group.identity()
        >>> torch.allclose(I, torch.eye(3))
        True
        """
        return torch.eye(self.n)

    def is_valid(self, R: torch.Tensor, atol: float = 1e-6) -> bool:
        """
        Check if a matrix R is a valid element of O(n) by both 
        checking that R @ R.T = I and det(R) = +/-1

        Parameters
        ----------
        R : torch.Tensor
            Matrix to check (n x n).
        atol : float, optional
            Absolute tolerance for checks. Default is 1e-6.

        Returns
        -------
        bool
            True if R is orthogonal (det(R) = ±1).

        Examples
        --------
        >>> group = O(3)
        >>> R = group.sample()
        >>> group.is_valid(R)
        True
        >>> group.is_valid(torch.randn(3, 3))
        False
        """
        if R.shape != (self.n, self.n):
            return False

        # Check orthogonality: R @ R.T ≈ I
        is_orthogonal = torch.allclose(R @ R.T, torch.eye(self.n), atol=atol)

        # Check determinant ≈ ±1
        det = torch.det(R)
        det_is_valid = torch.allclose(det.abs(), torch.tensor(1.0), atol=atol)

        return is_orthogonal and det_is_valid

    def is_rotation(self, R: torch.Tensor, atol: float = 1e-6) -> bool:
        """
        Check if an orthogonal matrix is a rotation (det = +1).

        Parameters
        ----------
        R : torch.Tensor
            Orthogonal matrix (n x n).
        atol : float, optional
            Absolute tolerance. Default is 1e-6.

        Returns
        -------
        bool
            True if R is in SO(n) (det = +1), False if reflection (det = -1).

        Examples
        --------
        >>> reflection = torch.eye(3)
        >>> reflection[0, 0] = -1  # Reflect through yz-plane
        >>> group.is_rotation(reflection)
        False
        """
        if not self.is_valid(R, atol=atol):
            raise ValueError("Matrix is not a valid orthogonal matrix")

        det = torch.det(R)
        return torch.allclose(det, torch.tensor(1.0), atol=atol)

    def reflection(self, axis: int = 0) -> torch.Tensor:
        """
        Generate a reflection matrix along a coordinate axis.

        Parameters
        ----------
        axis : int, optional
            Index of axis to reflect (0 to n-1). Default is 0.

        Returns
        -------
        torch.Tensor
            Reflection matrix with det = -1.

        Examples
        --------
        >>> group = O(3)
        >>> R = group.reflection(0)  # Reflect through yz-plane
        >>> torch.det(R)
        tensor(-1.0)
        >>> point = torch.tensor([1., 2., 3.])
        >>> reflected = R @ point  # x → -x, y and z unchanged
        >>> torch.allclose(reflected, torch.tensor([-1., 2., 3.]))
        True
        """
        if not (0 <= axis < self.n):
            raise ValueError(f"Axis must be in range [0, {self.n})")

        R = torch.eye(self.n)
        R[axis, axis] = -1
        return R

    def to_so(self, R: torch.Tensor) -> torch.Tensor:
        """
        Project an orthogonal matrix to SO(n) if needed.

        If det(R) = -1, compose with a reflection to get det = +1.

        Parameters
        ----------
        R : torch.Tensor
            Orthogonal matrix (n x n).

        Returns
        -------
        torch.Tensor
            Matrix in SO(n) with det = +1.

        Examples
        --------
        >>> group = O(3)
        >>> reflection = group.reflection(0)
        >>> torch.det(reflection)
        tensor(-1.0)
        >>> rotation = group.to_so(reflection)
        >>> torch.det(rotation)
        tensor(1.0)
        """
        if not self.is_valid(R):
            raise ValueError("Matrix is not a valid orthogonal matrix")

        if self.is_rotation(R):
            return R
        else:
            # Compose with a reflection to flip determinant
            return R @ self.reflection(0)


class O2:
    """
    The orthogonal group O(2) with angle parameterization.

    O(2) consists of 2D rotations and reflections. Elements can be
    parameterized as:
    - Rotations: angle θ ∈ [0, 2π), det = +1
    - Reflections: angle θ ∈ [0, 2π) with reflection, det = -1

    This provides a convenient angle-based interface for 2D orthogonal
    transformations, analogous to CircleGroup for SO(2).

    Examples
    --------
    >>> o2 = O2()
    >>> theta, is_reflection = o2.sample()
    >>> R = o2.to_matrix(theta, is_reflection)
    >>> torch.det(R)  # Can be 1 or -1
    """

    order = float('inf')
    abelian = False
    n = 2

    def sample(self) -> tuple[float, bool]:
        """
        Sample a random element of O(2).

        Returns
        -------
        theta : float
            Angle in radians [0, 2π).
        is_reflection : bool
            True if this is a reflection, False if rotation.
        """
        import math
        theta = random.uniform(0, 2 * math.pi)
        is_reflection = random.random() < 0.5
        return theta, is_reflection

    def to_matrix(self, theta: float, is_reflection: bool = False) -> torch.Tensor:
        """
        Convert angle (and reflection flag) to O(2) matrix.

        Parameters
        ----------
        theta : float
            Angle in radians.
        is_reflection : bool, optional
            If True, include reflection. Default False.

        Returns
        -------
        torch.Tensor
            2 x 2 orthogonal matrix.

        Examples
        --------
        >>> o2 = O2()
        >>> R = o2.to_matrix(0, is_reflection=True)  # Reflection through x-axis
        >>> torch.allclose(R, torch.tensor([[1., 0.], [0., -1.]]))
        True
        """
        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta, dtype=torch.float32)

        c = torch.cos(theta)
        s = torch.sin(theta)

        if is_reflection:
            # Reflection followed by rotation
            return torch.tensor([[c, s], [s, -c]], dtype=torch.float32)
        else:
            # Pure rotation
            return torch.tensor([[c, -s], [s, c]], dtype=torch.float32)

    def identity(self) -> torch.Tensor:
        """Return the identity element."""
        return torch.eye(2)

    def compose(self, g1: tuple[float, bool], g2: tuple[float, bool]) -> tuple[float, bool]:
        """
        Compose two O(2) elements.

        Parameters
        ----------
        g1, g2 : tuple[float, bool]
            Elements as (angle, is_reflection).

        Returns
        -------
        tuple[float, bool]
            Composed element.
        """
        import math
        theta1, ref1 = g1
        theta2, ref2 = g2

        # Composition rules in O(2)
        if not ref1 and not ref2:
            # Both rotations: angles add
            return (theta1 + theta2) % (2 * math.pi), False
        elif not ref1 and ref2:
            # Rotation then reflection
            return (theta2 - theta1) % (2 * math.pi), True
        elif ref1 and not ref2:
            # Reflection then rotation
            return (theta1 + theta2) % (2 * math.pi), True
        else:
            # Both reflections: compose to rotation
            return (theta2 - theta1) % (2 * math.pi), False

    def inverse(self, g: tuple[float, bool]) -> tuple[float, bool]:
        """
        Compute inverse of O(2) element.

        Parameters
        ----------
        g : tuple[float, bool]
            Element as (angle, is_reflection).

        Returns
        -------
        tuple[float, bool]
            Inverse element.
        """
        theta, is_reflection = g

        if is_reflection:
            # Reflections are self-inverse
            return (theta, True)
        else:
            # Rotation inverse is negative angle
            return ((2 * math.pi - theta) % (2 * math.pi), False)
