"""
Test suite for SO(3) implementation.
"""

import torch
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import with the filename directly
import importlib.util
spec = importlib.util.spec_from_file_location("so3_module", "so(3).py")
so3_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(so3_module)
SO3 = so3_module.SO3


def test_basic_properties():
    """Test basic group properties."""
    print("Testing SO(3) basic properties...")

    group = SO3()

    # Test properties
    assert group.order == float('inf')
    assert group.abelian == False

    # Test identity
    I = group.identity()
    assert I.shape == (3, 3)
    assert torch.allclose(I, torch.eye(3))

    print("  ✓ Basic properties correct!")


def test_sampling():
    """Test random sampling produces valid rotation matrices."""
    print("\nTesting SO(3) sampling...")

    group = SO3()

    for _ in range(10):
        R = group.sample()

        # Check shape
        assert R.shape == (3, 3)

        # Check orthogonality: R @ R.T = I
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-6), "Not orthogonal"

        # Check determinant = +1
        det = torch.det(R)
        assert torch.allclose(det, torch.tensor(1.0), atol=1e-6), f"Determinant is {det}, not 1"

    print("  ✓ Sampling produces valid rotations!")


def test_composition():
    """Test group composition."""
    print("\nTesting SO(3) composition...")

    group = SO3()

    R1 = group.sample()
    R2 = group.sample()
    R3 = group.compose(R1, R2)

    # Check composition is correct matrix multiplication
    assert torch.allclose(R3, R2 @ R1)

    # Check result is still a valid rotation
    assert torch.allclose(R3 @ R3.T, torch.eye(3), atol=1e-6)
    assert torch.allclose(torch.det(R3), torch.tensor(1.0), atol=1e-6)

    # Test associativity: (R1 ∘ R2) ∘ R3 = R1 ∘ (R2 ∘ R3)
    R4 = group.sample()
    left_assoc = group.compose(group.compose(R1, R2), R4)
    right_assoc = group.compose(R1, group.compose(R2, R4))
    assert torch.allclose(left_assoc, right_assoc, atol=1e-5)

    print("  ✓ Composition works correctly!")


def test_inverse():
    """Test group inverse."""
    print("\nTesting SO(3) inverse...")

    group = SO3()

    R = group.sample()
    R_inv = group.inverse(R)

    # Check inverse is the transpose
    assert torch.allclose(R_inv, R.T)

    # Check R @ R^(-1) = I
    assert torch.allclose(group.compose(R, R_inv), group.identity(), atol=1e-6)

    # Check R^(-1) @ R = I
    assert torch.allclose(group.compose(R_inv, R), group.identity(), atol=1e-6)

    # Check inverse is also a valid rotation
    assert torch.allclose(R_inv @ R_inv.T, torch.eye(3), atol=1e-6)
    assert torch.allclose(torch.det(R_inv), torch.tensor(1.0), atol=1e-6)

    print("  ✓ Inverse works correctly!")


def test_identity():
    """Test identity element."""
    print("\nTesting SO(3) identity...")

    group = SO3()
    I = group.identity()

    # Test identity with random rotations
    for _ in range(5):
        R = group.sample()

        # R ∘ I = R
        assert torch.allclose(group.compose(R, I), R, atol=1e-6)

        # I ∘ R = R
        assert torch.allclose(group.compose(I, R), R, atol=1e-6)

    print("  ✓ Identity element works correctly!")


def test_non_abelian():
    """Test that SO(3) is non-abelian (rotations don't commute)."""
    print("\nTesting SO(3) non-commutativity...")

    group = SO3()

    # Try multiple pairs to find non-commuting elements
    found_non_commuting = False
    for _ in range(20):
        R1 = group.sample()
        R2 = group.sample()

        R1_R2 = group.compose(R1, R2)
        R2_R1 = group.compose(R2, R1)

        if not torch.allclose(R1_R2, R2_R1, atol=1e-5):
            found_non_commuting = True
            break

    assert found_non_commuting, "Failed to find non-commuting rotations"

    print("  ✓ SO(3) is correctly non-abelian!")


def test_axis_angle():
    """Test axis-angle representation."""
    print("\nTesting SO(3) axis-angle representation...")

    group = SO3()

    # Test rotation around z-axis by 90 degrees
    axis = torch.tensor([0., 0., 1.])
    angle = torch.pi / 2
    R = group.from_axis_angle(axis, angle)

    # Check it's a valid rotation
    assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-6)
    assert torch.allclose(torch.det(R), torch.tensor(1.0), atol=1e-6)

    # Test point rotation
    point = torch.tensor([1., 0., 0.])
    rotated = R @ point
    expected = torch.tensor([0., 1., 0.])
    assert torch.allclose(rotated, expected, atol=1e-6), f"Expected {expected}, got {rotated}"

    # Test 180-degree rotation
    R_180 = group.from_axis_angle(axis, torch.pi)
    point2 = torch.tensor([1., 0., 0.])
    rotated2 = R_180 @ point2
    expected2 = torch.tensor([-1., 0., 0.])
    assert torch.allclose(rotated2, expected2, atol=1e-6)

    print("  ✓ Axis-angle representation works correctly!")


def test_random_axis_angle():
    """Test alternative sampling method."""
    print("\nTesting SO(3) random axis-angle sampling...")

    group = SO3()

    for _ in range(10):
        R = group.random_axis_angle()

        # Check it's a valid rotation
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-6)
        assert torch.allclose(torch.det(R), torch.tensor(1.0), atol=1e-6)

    print("  ✓ Random axis-angle sampling works!")


if __name__ == "__main__":
    test_basic_properties()
    test_sampling()
    test_composition()
    test_inverse()
    test_identity()
    test_non_abelian()
    test_axis_angle()
    test_random_axis_angle()

    print("\n" + "="*50)
    print("All SO(3) tests passed! ✓")
    print("="*50)
