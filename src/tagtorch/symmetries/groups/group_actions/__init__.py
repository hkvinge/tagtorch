"""
Group actions module for TAGTorch.

This module provides implementations of group actions on data, building on
the abstract group classes. Each action class combines a group with a specific
transformation on data (vectors, images, etc.).

Base class:
- GroupAction: Abstract base class for all group actions

Actions:
- TrivialSymmetry: Trivial group acting as identity
- ColumnPermutation: Symmetric group permuting vector elements
- VectorReversal: Z/2Z acting by sign multiplication
- ScalarMultiplication: R* acting by scalar multiplication (not a group action)
- ImageTranslation: Product of cyclic groups for image translations
- ContinuousImageRotation: SO(2) continuous rotations
- DiscreteImageRotation: C_n discrete rotations
"""

from .base import GroupAction
from .basic_symmetries import TrivialSymmetry
from .permutation_symmetries import ColumnPermutation, column_permutation
from .toy_symmetries import VectorReversal, ScalarMultiplication
from .image_symmetries import (
    ImageTranslation,
    ContinuousImageRotation,
    DiscreteImageRotation,
    ProductGroup
)

__all__ = [
    'GroupAction',
    'TrivialSymmetry',
    'ColumnPermutation',
    'column_permutation',
    'VectorReversal',
    'ScalarMultiplication',
    'ImageTranslation',
    'ContinuousImageRotation',
    'DiscreteImageRotation',
    'ProductGroup'
]
