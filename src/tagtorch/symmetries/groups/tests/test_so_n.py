"""
Test suite for SO(n) and CircleGroup implementations.
"""

import torch
import math
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from so_n import SO, CircleGroup


def test_so_n_basic():
    """Test SO(n) for various dimensions."""
    print("Testing SO(n) for n=2,3,4,5...")

    for n in [2, 3, 4, 5]:
        group = SO(n)

        # Test properties
        assert group.n == n
        assert group.order == float('inf')
        assert group.abelian == (n == 2), f"SO({n}) abelian property incorrect"

        # Test identity
        I = group.identity()
        assert I.shape == (n, n)
        assert torch.allclose(I, torch.eye(n))

        print(f"  ✓ SO({n}) basic properties correct!")


def test_so_n_sampling():
    """Test that sampling produces valid rotation matrices."""
    print("\nTesting SO(n) sampling...")

    for n in [2, 3, 4, 6]:
        group = SO(n)

        for _ in range(5):
            R = group.sample()

            # Check shape
            assert R.shape == (n, n)

            # Check orthogonality: R @ R.T = I
            assert torch.allclose(R @ R.T, torch.eye(n), atol=1e-6), \
                f"SO({n}) not orthogonal"

            # Check determinant = +1
            det = torch.det(R)
            assert torch.allclose(det, torch.tensor(1.0), atol=1e-6), \
                f"SO({n}) determinant is {det}, not 1"

        print(f"  ✓ SO({n}) sampling produces valid rotations!")


def test_so_n_group_axioms():
    """Test group axioms for SO(n)."""
    print("\nTesting SO(n) group axioms...")

    for n in [2, 3, 4]:
        group = SO(n)

        # Test associativity: (R1 ∘ R2) ∘ R3 = R1 ∘ (R2 ∘ R3)
        R1, R2, R3 = group.sample(), group.sample(), group.sample()
        left_assoc = group.compose(group.compose(R1, R2), R3)
        right_assoc = group.compose(R1, group.compose(R2, R3))
        assert torch.allclose(left_assoc, right_assoc, atol=1e-5), \
            f"SO({n}) not associative"

        # Test identity: R ∘ I = I ∘ R = R
        R = group.sample()
        I = group.identity()
        assert torch.allclose(group.compose(R, I), R, atol=1e-6)
        assert torch.allclose(group.compose(I, R), R, atol=1e-6)

        # Test inverse: R ∘ R^(-1) = I
        R_inv = group.inverse(R)
        assert torch.allclose(group.compose(R, R_inv), I, atol=1e-6)
        assert torch.allclose(group.compose(R_inv, R), I, atol=1e-6)

        print(f"  ✓ SO({n}) satisfies group axioms!")


def test_so2_abelian():
    """Test that SO(2) is abelian."""
    print("\nTesting SO(2) commutativity...")

    group = SO(2)

    for _ in range(10):
        R1, R2 = group.sample(), group.sample()

        # R1 ∘ R2 should equal R2 ∘ R1
        assert torch.allclose(
            group.compose(R1, R2),
            group.compose(R2, R1),
            atol=1e-5
        )

    print("  ✓ SO(2) is abelian!")


def test_so_n_higher_non_abelian():
    """Test that SO(n) for n≥3 is non-abelian."""
    print("\nTesting SO(n) non-commutativity for n≥3...")

    for n in [3, 4]:
        group = SO(n)

        # Try to find non-commuting elements
        found_non_commuting = False
        for _ in range(20):
            R1, R2 = group.sample(), group.sample()

            if not torch.allclose(
                group.compose(R1, R2),
                group.compose(R2, R1),
                atol=1e-5
            ):
                found_non_commuting = True
                break

        assert found_non_commuting, f"Failed to find non-commuting elements in SO({n})"
        print(f"  ✓ SO({n}) is non-abelian!")


def test_so_is_valid():
    """Test the is_valid method."""
    print("\nTesting SO(n).is_valid()...")

    group = SO(3)

    # Valid rotation
    R = group.sample()
    assert group.is_valid(R)

    # Invalid: random matrix (not orthogonal)
    assert not group.is_valid(torch.randn(3, 3))

    # Invalid: reflection (det = -1)
    reflection = torch.eye(3)
    reflection[0, 0] = -1
    assert not group.is_valid(reflection)

    print("  ✓ is_valid() works correctly!")


def test_circle_group_basic():
    """Test CircleGroup basic properties."""
    print("\nTesting CircleGroup basic properties...")

    circle = CircleGroup()

    assert circle.order == float('inf')
    assert circle.abelian == True
    assert circle.n == 2

    # Test identity
    assert circle.identity() == 0.0

    print("  ✓ CircleGroup basic properties correct!")


def test_circle_group_sampling():
    """Test CircleGroup sampling."""
    print("\nTesting CircleGroup sampling...")

    circle = CircleGroup()

    for _ in range(20):
        theta = circle.sample()
        assert 0 <= theta < 2 * math.pi, f"Angle {theta} out of range"

    print("  ✓ CircleGroup sampling works!")


def test_circle_group_operations():
    """Test CircleGroup group operations."""
    print("\nTesting CircleGroup operations...")

    circle = CircleGroup()

    # Test composition (angle addition)
    theta1 = math.pi / 2
    theta2 = math.pi / 2
    result = circle.compose(theta1, theta2)
    assert abs(result - math.pi) < 1e-10

    # Test wrapping
    theta1 = 3 * math.pi / 2
    theta2 = math.pi
    result = circle.compose(theta1, theta2)
    assert abs(result - math.pi / 2) < 1e-10

    # Test inverse
    theta = math.pi / 3
    theta_inv = circle.inverse(theta)
    result = circle.compose(theta, theta_inv)
    assert abs(result) < 1e-10 or abs(result - 2 * math.pi) < 1e-10

    # Test identity
    I = circle.identity()
    assert circle.compose(theta, I) == theta

    print("  ✓ CircleGroup operations work correctly!")


def test_circle_to_matrix():
    """Test CircleGroup to/from matrix conversion."""
    print("\nTesting CircleGroup ↔ SO(2) matrix conversion...")

    circle = CircleGroup()

    # Test conversion to matrix
    theta = math.pi / 4  # 45 degrees
    R = circle.to_matrix(theta)

    assert R.shape == (2, 2)

    # Check it's a valid rotation
    assert torch.allclose(R @ R.T, torch.eye(2), atol=1e-6)
    assert torch.allclose(torch.det(R), torch.tensor(1.0), atol=1e-6)

    # Test specific rotation
    point = torch.tensor([1., 0.])
    rotated = R @ point
    expected = torch.tensor([math.cos(theta), math.sin(theta)])
    assert torch.allclose(rotated, expected, atol=1e-6)

    # Test round-trip conversion
    theta_recovered = circle.from_matrix(R)
    assert abs(theta_recovered - theta) < 1e-6

    print("  ✓ CircleGroup ↔ matrix conversion works!")


def test_circle_to_complex():
    """Test CircleGroup to/from complex conversion."""
    print("\nTesting CircleGroup ↔ complex conversion...")

    circle = CircleGroup()

    # Test conversion to complex
    theta = math.pi / 3
    z = circle.to_complex(theta)

    # Check it's on unit circle
    assert abs(abs(z) - 1.0) < 1e-10

    # Check value
    expected = math.cos(theta) + 1j * math.sin(theta)
    assert abs(z - expected) < 1e-10

    # Test round-trip
    theta_recovered = circle.from_complex(z)
    assert abs(theta_recovered - theta) < 1e-10

    # Test special values
    assert abs(circle.to_complex(0) - 1.0) < 1e-10
    assert abs(circle.to_complex(math.pi / 2) - 1j) < 1e-10
    assert abs(circle.to_complex(math.pi) - (-1.0)) < 1e-10

    print("  ✓ CircleGroup ↔ complex conversion works!")


def test_circle_group_is_so2():
    """Test that CircleGroup is equivalent to SO(2)."""
    print("\nTesting CircleGroup ≅ SO(2)...")

    circle = CircleGroup()
    so2 = SO(2)

    # Sample angles and convert to matrices
    for _ in range(10):
        theta1 = circle.sample()
        theta2 = circle.sample()

        # Convert to matrices
        R1 = circle.to_matrix(theta1)
        R2 = circle.to_matrix(theta2)

        # Compose in angle space
        theta_composed = circle.compose(theta1, theta2)
        R_composed_angle = circle.to_matrix(theta_composed)

        # Compose in matrix space
        R_composed_matrix = so2.compose(R1, R2)

        # Should be equivalent
        assert torch.allclose(R_composed_angle, R_composed_matrix, atol=1e-5)

    print("  ✓ CircleGroup ≅ SO(2) verified!")


if __name__ == "__main__":
    test_so_n_basic()
    test_so_n_sampling()
    test_so_n_group_axioms()
    test_so2_abelian()
    test_so_n_higher_non_abelian()
    test_so_is_valid()
    test_circle_group_basic()
    test_circle_group_sampling()
    test_circle_group_operations()
    test_circle_to_matrix()
    test_circle_to_complex()
    test_circle_group_is_so2()

    print("\n" + "="*50)
    print("All SO(n) and CircleGroup tests passed! ✓")
    print("="*50)
    print("\nKey results:")
    print("  • SO(n) works for arbitrary dimensions n≥2")
    print("  • SO(2) is correctly identified as abelian")
    print("  • SO(n) for n≥3 is correctly non-abelian")
    print("  • CircleGroup provides angle-based SO(2)")
    print("  • CircleGroup ≅ SO(2) verified")
