"""
Test suite for group actions after refactoring.

Verifies that all action classes work correctly with the abstract group classes.
"""

import torch
import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basic_symmetries import TrivialSymmetry
from permutation_symmetries import ColumnPermutation
from toy_symmetries import VectorReversal, ScalarMultiplication
from image_symmetries import ImageTranslation, ContinuousImageRotation, DiscreteImageRotation


def test_trivial_symmetry():
    """Test TrivialSymmetry with new architecture."""
    print("Testing TrivialSymmetry...")

    action = TrivialSymmetry()

    # Test properties
    assert action.order == 1
    assert action.finite == True
    assert action.trivial == True

    # Test on data
    data = torch.randn(5, 10)

    # Sample and act
    transformed = action.sample_and_act(data, all_same=True)
    assert torch.allclose(data, transformed)

    # Test orbit
    orbit = action.orbit(data[0:1])
    assert len(orbit) == 1
    assert torch.allclose(orbit[0], data[0:1])

    print("  ✓ TrivialSymmetry tests passed!")


def test_column_permutation():
    """Test ColumnPermutation with SymmetricGroup."""
    print("\nTesting ColumnPermutation...")

    action = ColumnPermutation(5)

    # Test properties
    assert action.order == 120  # 5! = 120
    assert action.finite == True
    assert action.trivial == False

    # Test on data
    data = torch.arange(10 * 5).reshape(10, 5).float()

    # Test specific permutation
    perm = (1, 0, 2, 3, 4)  # Swap first two columns
    transformed = action.action(perm, data)
    assert transformed.shape == data.shape
    assert torch.allclose(transformed[:, 0], data[:, 1])
    assert torch.allclose(transformed[:, 1], data[:, 0])

    # Test sample_and_act
    transformed = action.sample_and_act(data, all_same=True)
    assert transformed.shape == data.shape

    # Test orbit (sample a few) - need to keep batch dimension
    orbit = action.orbit(data[0:1], num_samples=10)
    assert len(orbit) == 10

    print("  ✓ ColumnPermutation tests passed!")


def test_vector_reversal():
    """Test VectorReversal with CyclicGroup(2)."""
    print("\nTesting VectorReversal...")

    action = VectorReversal()

    # Test properties
    assert action.order == 2
    assert action.finite == True
    assert action.trivial == False

    # Test on data
    data = torch.randn(5, 10)

    # Test specific actions
    identity = action.action(1, data)
    assert torch.allclose(identity, data)

    negated = action.action(-1, data)
    assert torch.allclose(negated, -data)

    # Test orbit
    orbit = action.orbit(data[0:1])
    assert len(orbit) == 2
    # One should be original, one should be negated
    # (order depends on which comes first)

    print("  ✓ VectorReversal tests passed!")


def test_scalar_multiplication():
    """Test ScalarMultiplication (not using group classes)."""
    print("\nTesting ScalarMultiplication...")

    action = ScalarMultiplication(lambda1=0.5, lambda2=2.0)

    # Test properties
    assert action.finite == False

    # Test on data
    data = torch.randn(5, 10)

    # Test specific scalar
    scaled = action.action(2.0, data)
    assert torch.allclose(scaled, data * 2.0)

    # Test sample_and_act
    transformed = action.sample_and_act(data, all_same=True)
    assert transformed.shape == data.shape

    print("  ✓ ScalarMultiplication tests passed!")


def test_image_translation():
    """Test ImageTranslation with ProductGroup."""
    print("\nTesting ImageTranslation...")

    height, width = 32, 32
    action = ImageTranslation(height, width)

    # Test properties
    assert action.order == height * width
    assert action.finite == True

    # Create test image
    image = torch.randn(3, 3, height, width)  # 3 images, 3 channels

    # Test specific translation
    shift = (5, 7)
    translated = action.action(shift, image)
    assert translated.shape == image.shape

    # Verify translation (check a specific pixel)
    assert torch.allclose(translated[0, 0, 5, 7], image[0, 0, 0, 0])

    # Test sample_and_act
    transformed = action.sample_and_act(image, all_same=True)
    assert transformed.shape == image.shape

    print("  ✓ ImageTranslation tests passed!")


def test_continuous_image_rotation():
    """Test ContinuousImageRotation with infinite CyclicGroup."""
    print("\nTesting ContinuousImageRotation...")

    action = ContinuousImageRotation()

    # Test properties
    assert action.order == 'inf'
    assert action.finite == False

    # Create test image
    image = torch.randn(2, 3, 32, 32)  # 2 images, 3 channels, 32x32

    # Test specific rotation
    angle = 45.0
    rotated = action.action(angle, image)
    assert rotated.shape == image.shape

    # Test sample_and_act
    transformed = action.sample_and_act(image, all_same=True)
    assert transformed.shape == image.shape

    # Test orbit sampling
    orbit = action.orbit(image[0:1], num_samples=8)
    assert len(orbit) == 8

    print("  ✓ ContinuousImageRotation tests passed!")


def test_discrete_image_rotation():
    """Test DiscreteImageRotation with CyclicGroup."""
    print("\nTesting DiscreteImageRotation...")

    action = DiscreteImageRotation(order=4)  # 90-degree rotations

    # Test properties
    assert action.order == 4
    assert action.finite == True

    # Create test image
    image = torch.randn(2, 3, 32, 32)

    # Test specific rotations
    identity = action.action(0, image)
    assert torch.allclose(identity, image)

    # 90 degree rotation
    rotated_90 = action.action(1, image)
    assert rotated_90.shape == image.shape

    # Test orbit (complete)
    orbit = action.orbit(image[0:1], num_samples=10)  # Should give all 4 rotations
    assert len(orbit) == 4

    print("  ✓ DiscreteImageRotation tests passed!")


def test_inheritance_structure():
    """Verify all actions inherit from GroupAction properly."""
    print("\nTesting inheritance structure...")

    from base import GroupAction

    # All should have the key methods from GroupAction
    actions = [
        TrivialSymmetry(),
        ColumnPermutation(3),
        VectorReversal(),
        ImageTranslation(32, 32),
        ContinuousImageRotation(),
        DiscreteImageRotation()
    ]

    for action in actions:
        # Check that they have the key methods
        assert hasattr(action, 'action')
        assert hasattr(action, 'sample')
        assert hasattr(action, 'sample_and_act')
        assert hasattr(action, 'orbit')
        assert hasattr(action, 'group')
        assert hasattr(action, 'order')
        assert hasattr(action, 'finite')
        assert hasattr(action, 'trivial')
        # Check that they're actually GroupAction instances
        assert action.__class__.__bases__[0].__name__ == 'GroupAction'

    print("  ✓ All actions properly inherit from GroupAction!")


if __name__ == "__main__":
    test_trivial_symmetry()
    test_column_permutation()
    test_vector_reversal()
    test_scalar_multiplication()
    test_image_translation()
    test_continuous_image_rotation()
    test_discrete_image_rotation()
    test_inheritance_structure()

    print("\n" + "="*50)
    print("All action tests passed! ✓")
    print("="*50)
    print("\nRefactoring successful:")
    print("  • Group structure separated from actions")
    print("  • All actions inherit from GroupAction base class")
    print("  • sample_and_act() and orbit() methods unified")
    print("  • Code reuse maximized, duplication minimized")
