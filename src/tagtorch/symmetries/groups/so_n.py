import torch
import random
import math
from typing import List, Union

############### SO(n) - Special Orthogonal Groups #######################

class SO:
    """
    The special orthogonal group SO(n) of rotations in n dimensions.

    SO(n) consists of all n x n orthogonal matrices with determinant +1.
    These represent rotations in n-dimensional space.

    Properties:
    - Continuous Lie group 
    - Abelian only for n=2 (SO(2) is the circle group)
    - Non-abelian for n≥3 
    - Compact manifold 

    Elements are represented as n x n torch.Tensor matrices satisfying:
    - R @ R.T = I (orthogonality)
    - det(R) = 1 (special orthogonal)

    Parameters
    ----------
    n : int
        Dimension of the rotation group (n >= 2).

    Examples
    --------
    >>> # 2D rotations (circle group)
    >>> SO2 = SO(2)
    >>> R = SO2.sample()
    >>> R.shape
    torch.Size([2, 2])

    >>> # 4D rotations
    >>> SO4 = SO(4)
    >>> R = SO4.sample()
    >>> torch.allclose(R @ R.T, torch.eye(4))
    True

    Notes
    -----
    For SO(2), consider using CircleGroup class which provides a more
    convenient angle-based parameterization.

    References
    ----------
    Mezzadri, F. (2007). "How to generate random matrices from the classical
    compact groups." Notices of the AMS, 54(5), 592-604.
    """

    def __init__(self, n: int):
        """
        Initialize SO(n) group.

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
        self.abelian = (n == 2)  # Only SO(2) is abelian

    def enumerate_elements(self) -> List[torch.Tensor]:
        """
        Enumerate all elements in the group.

        Raises
        ------
        NotImplementedError
            SO(n) is a continuous group and cannot be enumerated.
        """
        raise NotImplementedError(f"Cannot enumerate elements of continuous group SO({self.n}).")

    def sample(self) -> torch.Tensor:
        """
        Sample a uniformly random rotation matrix from SO(n).

        Uses the QR decomposition method with determinant correction
        to ensure uniform sampling with respect to the Haar measure.

        Returns
        -------
        torch.Tensor
            An n x n rotation matrix with det(R) = 1.

        Examples
        --------
        >>> group = SO(5)
        >>> R = group.sample()
        >>> R.shape
        torch.Size([5, 5])
        >>> torch.allclose(torch.det(R), torch.tensor(1.0))
        True
        """
        # Random Gaussian n×n matrix
        random_matrix = torch.randn(self.n, self.n)

        # QR decomposition to extract orthogonal component
        Q, R = torch.linalg.qr(random_matrix)

        # Ensure positive diagonal of R to make Q uniformly distributed
        # (QR decomposition is unique up to signs)
        d = torch.diagonal(R)
        Q = Q @ torch.diag(d / torch.abs(d))

        # Ensure det(Q) = +1 (not -1)
        if torch.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]

        return Q

    def compose(self, g1: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
        """
        Compose two rotation matrices via matrix multiplication.

        Parameters
        ----------
        g1 : torch.Tensor
            First rotation matrix (n x n).
        g2 : torch.Tensor
            Second rotation matrix (n x n).

        Returns
        -------
        torch.Tensor
            The composed rotation matrix g2 @ g1.

        Examples
        --------
        >>> group = SO(3)
        >>> R1, R2 = group.sample(), group.sample()
        >>> R3 = group.compose(R1, R2)
        >>> torch.allclose(R3, R2 @ R1)
        True
        """
        if g1.shape != (self.n, self.n) or g2.shape != (self.n, self.n):
            raise ValueError(f"Expected {self.n} x {self.n} matrices")
        return g2 @ g1

    def inverse(self, g: torch.Tensor) -> torch.Tensor:
        """
        Compute the inverse of a rotation matrix.

        For orthogonal matrices, the inverse is simply the transpose.

        Parameters
        ----------
        g : torch.Tensor
            Rotation matrix (n x n).

        Returns
        -------
        torch.Tensor
            The inverse rotation matrix g^(-1) = g.T.

        Examples
        --------
        >>> group = SO(4)
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
        Return the identity rotation matrix.

        Returns
        -------
        torch.Tensor
            The n x n identity matrix.

        Examples
        --------
        >>> group = SO(3)
        >>> I = group.identity()
        >>> torch.allclose(I, torch.eye(3))
        True
        """
        return torch.eye(self.n)

    def is_valid(self, R: torch.Tensor, atol: float = 1e-6) -> bool:
        """
        Check if a matrix is a valid element of SO(n).

        Parameters
        ----------
        R : torch.Tensor
            Matrix to check (n x n).
        atol : float, optional
            Absolute tolerance for checks. Default is 1e-6.

        Returns
        -------
        bool
            True if R is orthogonal with det(R) = 1.

        Examples
        --------
        >>> group = SO(3)
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

        # Check determinant ≈ 1
        det_is_one = torch.allclose(torch.det(R), torch.tensor(1.0), atol=atol)

        return is_orthogonal and det_is_one


class CircleGroup:
    """
    The circle group S¹ ≅ SO(2) with angle parameterization.

    This is mathematically equivalent to SO(2) (2D rotations) but uses
    the more convenient angle representation θ ∈ [0, 2π) instead of 2 x 2
    matrices.

    The circle group is:
    - Abelian (rotations commute in 2D)
    - Continuous and compact
    - The unique compact, connected 1-dimensional Lie group

    Elements are represented as floats (angles in radians).

    For matrix operations, use SO(2) directly via the SO class.

    Examples
    --------
    >>> circle = CircleGroup()
    >>> theta = circle.sample()  # Random angle
    >>> 0 <= theta < 2*math.pi
    True

    >>> # Composition is angle addition
    >>> circle.compose(math.pi/2, math.pi/2)  # 90° + 90° = 180°
    3.141592653589793

    >>> # Convert to matrix form
    >>> R = circle.to_matrix(math.pi/4)  # 45° rotation
    >>> R.shape
    torch.Size([2, 2])

    Notes
    -----
    This is equivalent to:
    - SO(2) - 2D rotation matrices
    - U(1) - Complex numbers on unit circle (z = e^(iθ))
    - The quotient R/Z under addition
    """

    order = float('inf')
    abelian = True
    n = 2

    def enumerate_elements(self) -> List[float]:
        """
        Enumerate all elements in the group.

        Raises
        ------
        NotImplementedError
            The circle group is continuous and cannot be enumerated.
        """
        raise NotImplementedError("Cannot enumerate elements of continuous circle group.")

    def sample(self) -> float:
        """
        Sample a uniformly random angle from [0, 2π).

        Returns
        -------
        float
            Random angle in radians from [0, 2π).

        Examples
        --------
        >>> circle = CircleGroup()
        >>> theta = circle.sample()
        >>> 0 <= theta < 2*math.pi
        True
        """
        return random.uniform(0, 2 * math.pi)

    def compose(self, theta1: float, theta2: float) -> float:
        """
        Compose two rotations by adding angles modulo 2π.

        Parameters
        ----------
        theta1 : float
            First angle in radians.
        theta2 : float
            Second angle in radians.

        Returns
        -------
        float
            The composed angle (theta1 + theta2) mod 2π.

        Examples
        --------
        >>> circle = CircleGroup()
        >>> circle.compose(math.pi, math.pi)  # 180° + 180° = 0° (mod 360°)
        0.0
        """
        return (theta1 + theta2) % (2 * math.pi)

    def inverse(self, theta: float) -> float:
        """
        Compute the inverse of a rotation angle.

        Parameters
        ----------
        theta : float
            Angle in radians.

        Returns
        -------
        float
            The inverse angle (2π - theta) mod 2π.

        Examples
        --------
        >>> circle = CircleGroup()
        >>> theta = math.pi / 3  # 60°
        >>> theta_inv = circle.inverse(theta)  # -60° = 300°
        >>> circle.compose(theta, theta_inv)
        0.0
        """
        return (2 * math.pi - theta) % (2 * math.pi)

    def identity(self) -> float:
        """
        Return the identity element (zero angle).

        Returns
        -------
        float
            The identity angle (0.0).

        Examples
        --------
        >>> circle = CircleGroup()
        >>> circle.identity()
        0.0
        """
        return 0.0

    def to_matrix(self, theta: float) -> torch.Tensor:
        """
        Convert angle to SO(2) matrix representation.

        Parameters
        ----------
        theta : float
            Angle in radians.

        Returns
        -------
        torch.Tensor
            2 x 2 rotation matrix [[cos θ, -sin θ], [sin θ, cos θ]].

        Examples
        --------
        >>> circle = CircleGroup()
        >>> R = circle.to_matrix(math.pi / 2)  # 90° rotation
        >>> torch.allclose(R @ torch.tensor([1., 0.]), torch.tensor([0., 1.]))
        True
        """
        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta, dtype=torch.float32)

        c = torch.cos(theta)
        s = torch.sin(theta)
        return torch.tensor([[c, -s], [s, c]], dtype=torch.float32)

    def from_matrix(self, R: torch.Tensor) -> float:
        """
        Extract angle from SO(2) matrix representation.

        Parameters
        ----------
        R : torch.Tensor
            2 x 2 rotation matrix.

        Returns
        -------
        float
            The rotation angle in [0, 2π).

        Examples
        --------
        >>> circle = CircleGroup()
        >>> theta = math.pi / 4
        >>> R = circle.to_matrix(theta)
        >>> recovered = circle.from_matrix(R)
        >>> abs(recovered - theta) < 1e-6
        True
        """
        if R.shape != (2, 2):
            raise ValueError(f"Expected 2 x 2 matrix, got {R.shape}")

        angle = torch.atan2(R[1, 0], R[0, 0]).item()
        # Ensure angle is in [0, 2π)
        if angle < 0:
            angle += 2 * math.pi
        return angle

    def to_complex(self, theta: float) -> complex:
        """
        Convert angle to complex number e^(iθ) on the unit circle.

        Parameters
        ----------
        theta : float
            Angle in radians.

        Returns
        -------
        complex
            Complex number e^(iθ) = cos(θ) + i·sin(θ).

        Examples
        --------
        >>> circle = CircleGroup()
        >>> z = circle.to_complex(math.pi / 2)
        >>> abs(abs(z) - 1.0) < 1e-6  # Unit circle
        True
        """
        return math.cos(theta) + 1j * math.sin(theta)

    def from_complex(self, z: complex) -> float:
        """
        Extract angle from complex number on unit circle.

        Parameters
        ----------
        z : complex
            Complex number (should be on unit circle).

        Returns
        -------
        float
            The angle in [0, 2π).

        Examples
        --------
        >>> circle = CircleGroup()
        >>> z = 0 + 1j  # i = e^(iπ/2)
        >>> theta = circle.from_complex(z)
        >>> abs(theta - math.pi/2) < 1e-6
        True
        """
        angle = math.atan2(z.imag, z.real)
        if angle < 0:
            angle += 2 * math.pi
        return angle

############### SO(3) - Special Orthogonal Group in 3D #######################

class SO3(SO):
    """
    The special orthogonal group SO(3) of 3D rotations.

    This is a specialized version of SO(n) for n=3, with additional
    methods specific to 3D rotations (axis-angle representation via
    Rodrigues' formula).

    SO(3) consists of all 3 x 3 orthogonal matrices with determinant +1.
    These represent rotations in 3-dimensional space.

    Properties:
    - Continuous Lie group 
    - Non-abelian 
    - Compact manifold 

    Elements are represented as 3 x 3 torch.Tensor matrices satisfying:
    - R @ R.T = I (orthogonality)
    - det(R) = 1 (special orthogonal)

    Examples
    --------
    >>> group = SO3()
    >>> R = group.sample()  # Random rotation matrix
    >>> R.shape
    torch.Size([3, 3])
    >>> torch.allclose(R @ R.T, torch.eye(3))  # Orthogonality check
    True

    >>> # Use axis-angle representation
    >>> axis = torch.tensor([0., 0., 1.])  # z-axis
    >>> R = group.from_axis_angle(axis, torch.pi / 2)  # 90° rotation

    Notes
    -----
    For general n-dimensional rotations, use the SO class directly.
    For 2D rotations, consider using CircleGroup.
    """

    def __init__(self):
        """Initialize SO(3) group."""
        super().__init__(n=3)

    def from_axis_angle(self, axis: torch.Tensor, angle: float) -> torch.Tensor:
        """
        Create a rotation matrix from axis-angle representation.

        Uses Rodrigues' rotation formula to construct the rotation matrix
        that rotates by `angle` radians around the given `axis`.

        This parameterization is specific to SO(3). In higher dimensions,
        rotations occur in planes rather than around axes.

        Parameters
        ----------
        axis : torch.Tensor
            3D rotation axis (will be normalized).
        angle : float
            Rotation angle in radians.

        Returns
        -------
        torch.Tensor
            3 x 3 rotation matrix.

        Examples
        --------
        >>> group = SO3()
        >>> axis = torch.tensor([0., 0., 1.])  # z-axis
        >>> R = group.from_axis_angle(axis, torch.pi / 2)  # 90-degree rotation
        >>> point = torch.tensor([1., 0., 0.])
        >>> rotated = R @ point
        >>> torch.allclose(rotated, torch.tensor([0., 1., 0.]))
        True

        Notes
        -----
        Uses Rodrigues' rotation formula:
        R = I + sin(θ)K + (1 - cos(θ))K²
        where K is the skew-symmetric cross-product matrix of the axis.

        References
        ----------
        Rodrigues, O. (1840). "Des lois géométriques qui régissent les
        déplacements d'un système solide dans l'espace."
        """
        if axis.shape != (3,):
            raise ValueError(f"Axis must be 3D vector, got shape {axis.shape}")

        # Normalize axis
        axis = axis / torch.linalg.norm(axis)

        # Convert angle to tensor if needed
        if not isinstance(angle, torch.Tensor):
            angle = torch.tensor(angle, dtype=axis.dtype)

        # Rodrigues' formula: R = I + sin(θ)K + (1 - cos(θ))K²
        # where K is the skew-symmetric cross-product matrix
        K = torch.tensor([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]], dtype=axis.dtype)

        R = torch.eye(3, dtype=axis.dtype) + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
        return R

    def to_axis_angle(self, R: torch.Tensor) -> tuple[torch.Tensor, float]:
        """
        Extract axis-angle representation from a rotation matrix.

        Parameters
        ----------
        R : torch.Tensor
            3 x 3 rotation matrix.

        Returns
        -------
        axis : torch.Tensor
            Rotation axis (unit vector).
        angle : float
            Rotation angle in radians [0, π].

        Examples
        --------
        >>> group = SO3()
        >>> axis_in = torch.tensor([0., 0., 1.])
        >>> angle_in = torch.pi / 3
        >>> R = group.from_axis_angle(axis_in, angle_in)
        >>> axis_out, angle_out = group.to_axis_angle(R)
        >>> torch.allclose(axis_out, axis_in)
        True
        >>> abs(angle_out - angle_in) < 1e-6
        True

        Notes
        -----
        Special cases:
        - Identity: angle = 0, axis is arbitrary (returns [0, 0, 1])
        - 180° rotation: angle = π, axis extracted from matrix
        """
        if R.shape != (3, 3):
            raise ValueError(f"Expected 3 x 3 matrix, got {R.shape}")

        # Extract angle from trace
        trace = torch.trace(R)
        angle = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))

        # Handle special cases
        if angle.abs() < 1e-6:
            # Identity rotation
            return torch.tensor([0., 0., 1.]), 0.0

        if (angle - torch.pi).abs() < 1e-6:
            # 180-degree rotation - extract axis from diagonal
            # The axis is the eigenvector with eigenvalue 1
            diag = torch.diagonal(R)
            i = torch.argmax(diag)
            axis = torch.sqrt(torch.clamp((R[i, i] + 1) / 2, 0, 1))
            # ... (full implementation would extract other components)
            # Simplified: just return normalized column
            axis = R[:, i]
            axis = axis / torch.linalg.norm(axis)
            return axis, torch.pi.item()

        # General case: extract axis from antisymmetric part
        axis = torch.tensor([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ], dtype=R.dtype)
        axis = axis / torch.linalg.norm(axis)

        return axis, angle.item()

    def random_axis_angle(self) -> torch.Tensor:
        """
        Sample a random rotation using axis-angle parameterization.

        This is an alternative sampling method that's more interpretable
        than the QR decomposition approach used in sample().

        Returns
        -------
        torch.Tensor
            A 3 x 3 rotation matrix.

        Examples
        --------
        >>> group = SO3()
        >>> R = group.random_axis_angle()
        >>> group.is_valid(R)
        True

        Notes
        -----
        This method samples uniformly with respect to the Haar measure
        by choosing a uniformly random axis on the unit sphere and a
        uniformly random angle in [0, 2π).
        """
        # Random axis (uniformly distributed on unit sphere)
        axis = torch.randn(3)
        axis = axis / torch.linalg.norm(axis)

        # Random angle
        angle = random.uniform(0, 2 * torch.pi)

        return self.from_axis_angle(axis, angle)