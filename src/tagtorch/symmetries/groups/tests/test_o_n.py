"""
Test suite for O(n) (orthogonal groups) implementation.
"""

import torch
import math
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from o_n import O, O2


def test_o_n_basic():
    """Test O(n) basic properties."""
    print("Testing O(n) basic properties...")

    for n in [2, 3, 4]:
        group = O(n)

        # Test properties
        assert group.n == n
        assert group.order == float('inf')
        assert group.abelian == False  # O(n) non-abelian for all n ≥ 2

        # Test identity
        I = group.identity()
        assert I.shape == (n, n)
        assert torch.allclose(I, torch.eye(n))

        print(f"  ✓ O({n}) basic properties correct!")


def test_o_n_sampling():
    """Test sampling from O(n)."""
    print("\nTesting O(n) sampling...")

    group = O(3)

    # Test sampling with reflections
    det_plus = 0
    det_minus = 0

    for _ in range(20):
        R = group.sample(include_reflections=True)

        # Check orthogonality
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-6)

        # Check determinant
        det = torch.det(R)
        assert torch.allclose(det.abs(), torch.tensor(1.0), atol=1e-6)

        if det > 0:
            det_plus += 1
        else:
            det_minus += 1

    # Should get both +1 and -1 determinants
    assert det_plus > 0 and det_minus > 0, "Should sample both rotations and reflections"

    print("  ✓ O(3) sampling produces valid orthogonal matrices!")

    # Test sampling without reflections (should be SO(n))
    print("\nTesting O(n) sampling without reflections...")
    for _ in range(10):
        R = group.sample(include_reflections=False)
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-6)
        assert torch.allclose(torch.det(R), torch.tensor(1.0), atol=1e-6)

    print("  ✓ O(3) sampling without reflections produces SO(3) elements!")


def test_o_n_group_operations():
    """Test group operations."""
    print("\nTesting O(n) group operations...")

    group = O(3)

    # Test composition
    R1, R2 = group.sample(), group.sample()
    R3 = group.compose(R1, R2)

    assert torch.allclose(R3, R2 @ R1)
    assert group.is_valid(R3)

    # Test inverse
    R = group.sample()
    R_inv = group.inverse(R)

    assert torch.allclose(group.compose(R, R_inv), group.identity(), atol=1e-6)
    assert torch.allclose(group.compose(R_inv, R), group.identity(), atol=1e-6)

    # Test associativity
    R1, R2, R3 = group.sample(), group.sample(), group.sample()
    left = group.compose(group.compose(R1, R2), R3)
    right = group.compose(R1, group.compose(R2, R3))
    assert torch.allclose(left, right, atol=1e-5)

    print("  ✓ O(3) group operations work correctly!")


def test_o_n_is_valid():
    """Test is_valid method."""
    print("\nTesting O(n).is_valid()...")

    group = O(3)

    # Valid: sample from group
    R = group.sample()
    assert group.is_valid(R)

    # Invalid: random matrix
    assert not group.is_valid(torch.randn(3, 3))

    # Valid: reflection
    reflection = torch.eye(3)
    reflection[0, 0] = -1
    assert group.is_valid(reflection)

    print("  ✓ is_valid() works correctly!")


def test_o_n_is_rotation():
    """Test is_rotation method."""
    print("\nTesting O(n).is_rotation()...")

    group = O(3)

    # Rotation (det = +1)
    rotation = group.sample(include_reflections=False)
    assert group.is_rotation(rotation)

    # Reflection (det = -1)
    reflection = torch.eye(3)
    reflection[0, 0] = -1
    assert not group.is_rotation(reflection)

    # Sample and check
    R = group.sample(include_reflections=True)
    det = torch.det(R)
    if det > 0:
        assert group.is_rotation(R)
    else:
        assert not group.is_rotation(R)

    print("  ✓ is_rotation() distinguishes rotations from reflections!")


def test_o_n_reflection():
    """Test reflection generation."""
    print("\nTesting O(n).reflection()...")

    group = O(3)

    # Reflection through yz-plane (x → -x)
    R = group.reflection(0)
    assert torch.allclose(torch.det(R), torch.tensor(-1.0))
    assert group.is_valid(R)
    assert not group.is_rotation(R)

    # Test action
    point = torch.tensor([1., 2., 3.])
    reflected = R @ point
    assert torch.allclose(reflected, torch.tensor([-1., 2., 3.]))

    # Test other axes
    R1 = group.reflection(1)  # Reflect through xz-plane
    point1 = torch.tensor([1., 2., 3.])
    reflected1 = R1 @ point1
    assert torch.allclose(reflected1, torch.tensor([1., -2., 3.]))

    print("  ✓ reflection() generates correct reflections!")


def test_o_n_to_so():
    """Test to_so projection."""
    print("\nTesting O(n).to_so()...")

    group = O(3)

    # Test with rotation (should be unchanged)
    rotation = group.sample(include_reflections=False)
    projected = group.to_so(rotation)
    assert torch.allclose(projected, rotation)
    assert torch.allclose(torch.det(projected), torch.tensor(1.0))

    # Test with reflection (should flip to rotation)
    reflection = group.reflection(0)
    assert torch.allclose(torch.det(reflection), torch.tensor(-1.0))

    projected = group.to_so(reflection)
    assert torch.allclose(torch.det(projected), torch.tensor(1.0), atol=1e-6)
    assert group.is_rotation(projected)

    print("  ✓ to_so() correctly projects to SO(n)!")


def test_o_n_non_abelian():
    """Test that O(n) is non-abelian."""
    print("\nTesting O(n) non-commutativity...")

    group = O(3)

    # O(n) should be non-abelian for all n ≥ 2
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

    assert found_non_commuting, "O(3) should be non-abelian"

    print("  ✓ O(3) is correctly non-abelian!")


def test_o2_basic():
    """Test O2 angle parameterization."""
    print("\nTesting O2 (angle parameterization)...")

    o2 = O2()

    # Test properties
    assert o2.order == float('inf')
    assert o2.abelian == False
    assert o2.n == 2

    # Test identity
    I = o2.identity()
    assert torch.allclose(I, torch.eye(2))

    print("  ✓ O2 basic properties correct!")


def test_o2_sampling():
    """Test O2 sampling."""
    print("\nTesting O2 sampling...")

    o2 = O2()

    reflections = 0
    rotations = 0

    for _ in range(20):
        theta, is_reflection = o2.sample()

        # Check angle range
        assert 0 <= theta < 2 * math.pi

        # Check type
        assert isinstance(is_reflection, bool)

        if is_reflection:
            reflections += 1
        else:
            rotations += 1

    # Should get both
    assert reflections > 0 and rotations > 0

    print("  ✓ O2 sampling works!")


def test_o2_to_matrix():
    """Test O2 to_matrix conversion."""
    print("\nTesting O2 matrix conversion...")

    o2 = O2()

    # Pure rotation
    R_rot = o2.to_matrix(math.pi / 2, is_reflection=False)
    assert torch.allclose(torch.det(R_rot), torch.tensor(1.0))
    assert torch.allclose(R_rot @ R_rot.T, torch.eye(2), atol=1e-6)

    # With reflection
    R_ref = o2.to_matrix(0, is_reflection=True)
    assert torch.allclose(torch.det(R_ref), torch.tensor(-1.0), atol=1e-6)
    assert torch.allclose(R_ref @ R_ref.T, torch.eye(2), atol=1e-6)

    # Test reflection through x-axis
    assert torch.allclose(R_ref, torch.tensor([[1., 0.], [0., -1.]]))

    print("  ✓ O2 matrix conversion works!")


def test_o2_operations():
    """Test O2 group operations."""
    print("\nTesting O2 group operations...")

    o2 = O2()

    # Test composition
    g1 = (math.pi / 4, False)  # 45° rotation
    g2 = (math.pi / 4, False)  # 45° rotation
    g3 = o2.compose(g1, g2)

    theta3, ref3 = g3
    assert abs(theta3 - math.pi / 2) < 1e-10  # 90° rotation
    assert ref3 == False

    # Test rotation × reflection = reflection
    g1 = (math.pi / 4, False)
    g2 = (0, True)
    g3 = o2.compose(g1, g2)
    assert g3[1] == True  # Should be reflection

    # Test reflection × reflection = rotation
    g1 = (0, True)
    g2 = (math.pi / 4, True)
    g3 = o2.compose(g1, g2)
    assert g3[1] == False  # Should be rotation

    # Test inverse
    g = (math.pi / 3, False)
    g_inv = o2.inverse(g)
    composed = o2.compose(g, g_inv)
    assert abs(composed[0]) < 1e-10 or abs(composed[0] - 2*math.pi) < 1e-10
    assert composed[1] == False

    print("  ✓ O2 operations work correctly!")


if __name__ == "__main__":
    test_o_n_basic()
    test_o_n_sampling()
    test_o_n_group_operations()
    test_o_n_is_valid()
    test_o_n_is_rotation()
    test_o_n_reflection()
    test_o_n_to_so()
    test_o_n_non_abelian()
    test_o2_basic()
    test_o2_sampling()
    test_o2_to_matrix()
    test_o2_operations()

    print("\n" + "="*50)
    print("All O(n) tests passed! ✓")
    print("="*50)
    print("\nKey results:")
    print("  • O(n) includes both rotations and reflections")
    print("  • O(n) sampling produces det = ±1 matrices")
    print("  • is_rotation() distinguishes SO(n) ⊂ O(n)")
    print("  • O2 provides angle parameterization for 2D")
    print("  • All group axioms verified")
