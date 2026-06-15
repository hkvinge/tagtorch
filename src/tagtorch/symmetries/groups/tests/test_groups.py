"""
Simple tests to verify group implementations.
"""

from trivial_group import TrivialGroup
from cyclic_groups import CyclicGroup
from symmetric_groups import SymmetricGroup


def test_trivial_group():
    """Test TrivialGroup implementation."""
    print("Testing TrivialGroup...")
    G = TrivialGroup()

    # Test basic properties
    assert G.order == 1
    assert G.abelian == True
    assert G.identity() == 1

    # Test operations
    g = G.sample()
    assert g == 1
    assert G.inverse(g) == 1
    assert G.compose(g, g) == 1
    assert G.enumerate_elements() == [1]

    print("  ✓ All TrivialGroup tests passed!")


def test_cyclic_group():
    """Test CyclicGroup implementation."""
    print("\nTesting CyclicGroup (finite)...")
    G = CyclicGroup(5)

    # Test basic properties
    assert G.order == 5
    assert G.abelian == True
    assert G.identity() == 0

    # Test composition
    assert G.compose(2, 3) == 0  # 2 + 3 = 5 ≡ 0 (mod 5)
    assert G.compose(1, 4) == 0
    assert G.compose(2, 2) == 4

    # Test inverse
    assert G.inverse(0) == 0
    assert G.inverse(1) == 4
    assert G.inverse(2) == 3
    assert G.inverse(3) == 2
    assert G.inverse(4) == 1

    # Test that g + inverse(g) = identity
    for g in range(5):
        assert G.compose(g, G.inverse(g)) == G.identity()

    # Test enumeration
    assert G.enumerate_elements() == [0, 1, 2, 3, 4]

    # Test sampling
    g = G.sample()
    assert 0 <= g < 5

    print("  ✓ All CyclicGroup tests passed!")


def test_symmetric_group():
    """Test SymmetricGroup implementation."""
    print("\nTesting SymmetricGroup(3)...")
    S3 = SymmetricGroup(3)

    # Test basic properties
    assert S3.n == 3
    assert S3.order == 6  # 3! = 6
    assert S3.abelian == False  # S_n is non-abelian for n >= 3

    # Test identity
    e = S3.identity()
    assert e == (0, 1, 2)

    # Test composition
    g1 = (1, 0, 2)  # swap 0 and 1
    g2 = (0, 2, 1)  # swap 1 and 2
    g3 = S3.compose(g1, g2)  # g2(g1(·))
    assert g3 == (2, 0, 1)

    # Test that composition with identity gives same element
    assert S3.compose(g1, e) == g1
    assert S3.compose(e, g1) == g1

    # Test inverse
    inv_g1 = S3.inverse(g1)
    assert inv_g1 == (1, 0, 2)  # swaps are self-inverse
    assert S3.compose(g1, inv_g1) == e

    # Test that S3 is not abelian
    g1_g2 = S3.compose(g1, g2)
    g2_g1 = S3.compose(g2, g1)
    assert g1_g2 != g2_g1  # Non-commutative

    # Test enumeration
    elements = S3.enumerate_elements()
    assert len(elements) == 6
    assert e in elements

    # Test sampling
    g = S3.sample()
    assert len(g) == 3
    assert set(g) == {0, 1, 2}

    # Test cycle decomposition
    assert S3.cycle_decomposition((0, 1, 2)) == []  # identity has no cycles
    assert S3.cycle_decomposition((1, 0, 2)) == [(0, 1)]  # one 2-cycle
    assert S3.cycle_decomposition((1, 2, 0)) == [(0, 1, 2)]  # one 3-cycle

    print("  ✓ All SymmetricGroup tests passed!")

    # Test S2 (should be abelian)
    print("\nTesting SymmetricGroup(2)...")
    S2 = SymmetricGroup(2)
    assert S2.order == 2
    assert S2.abelian == True
    print("  ✓ S2 is correctly identified as abelian!")


if __name__ == "__main__":
    test_trivial_group()
    test_cyclic_group()
    test_symmetric_group()
    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50)
