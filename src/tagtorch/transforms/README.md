# TAGTorch Transforms

A PyTorch-style transforms API for topological and symmetry-based data processing and augmentation.

## Overview

The `tagtorch.transforms` module provides a familiar interface inspired by `torchvision.transforms`, but specialized for:
- **Topological transforms**: ECT (Euler Characteristic Transform) and other topological descriptors
- **Symmetry transforms**: Group action-based augmentations for equivariant learning
- **Composition utilities**: Standard transform composition patterns

## Architecture

The module follows the torchvision design pattern with two complementary APIs:

### 1. Functional API (`tagtorch.transforms.functional`)
Pure stateless functions for maximum flexibility:
```python
import tagtorch.transforms.functional as TF

ect = TF.grayscale_to_ect(img, num_dirs=32, num_thresh=128)
rotated = TF.apply_group_action(img, rotation_group)
```

### 2. Class API (nn.Module wrappers)
Composable, serializable transforms:
```python
from tagtorch.transforms import GrayscaleToECT, Compose

transform = Compose([
    GrayscaleToECT(num_dirs=32, num_thresh=128),
])
```

## Module Structure

```
tagtorch/transforms/
├── __init__.py           # Main exports
├── functional.py         # Functional API (core implementations)
├── topological.py        # ECT and topological transforms
├── symmetry.py           # Group action augmentations
└── compose.py            # Composition utilities
```

## Quick Start

### Basic ECT Transform

```python
import torch
from tagtorch.transforms import ToECT

# Create transform
transform = ToECT(num_dirs=32, num_thresh=128, foreground='B')

# Apply to image
img = torch.randn(1, 28, 28)  # Grayscale MNIST
ect = transform(img)
print(ect.shape)  # torch.Size([1, 128, 32])
```

### Symmetry-Based Augmentation

```python
from tagtorch.transforms import RandomGroupAction, Compose
from tagtorch.symmetries.groups.group_actions import DiscreteImageRotation

# Create rotation group
rotation = DiscreteImageRotation(order=4)  # 0°, 90°, 180°, 270°

# Create augmentation pipeline
transform = Compose([
    RandomGroupAction(rotation, p=0.5),  # 50% chance of rotation
    ToECT(num_dirs=64, num_thresh=256),
])

# Use in dataset
img = torch.randn(1, 28, 28)
output = transform(img)
```

### Complete Training Pipeline

```python
from torch.utils.data import DataLoader
from tagtorch.transforms import Compose, RandomGroupAction, ToECT
from tagtorch.symmetries.groups.group_actions import (
    DiscreteImageRotation, 
    ImageTranslation
)

# Define augmentations
rotation = DiscreteImageRotation(order=4)
translation = ImageTranslation(image_height=28, image_width=28)

# Build pipeline
train_transform = Compose([
    RandomGroupAction(rotation, p=0.5),
    RandomGroupAction(translation, p=0.3),
    ToECT(num_dirs=32, num_thresh=128, foreground='B'),
])

# Use with DataLoader
train_dataset = MyDataset(transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

## API Reference

### Topological Transforms

#### `GrayscaleToECT(num_dirs=32, num_thresh=128, foreground='B', epsilon=0.01)`
Convert grayscale image to Euler Characteristic Transform.

**Parameters:**
- `num_dirs` (int): Number of directions for ECT computation
- `num_thresh` (int): Number of threshold bins
- `foreground` (str): 'B' for black or 'W' for white foreground
- `epsilon` (float): Threshold for foreground detection

**Shape:**
- Input: `(H, W)` or `(1, H, W)`
- Output: `(1, num_thresh, num_dirs)`

#### `ColorToECT(num_dirs=32, num_thresh=128, mode='grayscale', foreground='B', epsilon=0.01)`
Convert color image to ECT.

**Parameters:**
- `mode` (str): 'grayscale' converts to grayscale first, 'channels' processes each RGB channel separately

**Shape:**
- Input: `(3, H, W)`
- Output (grayscale mode): `(1, num_thresh, num_dirs)`
- Output (channels mode): `(3, num_thresh, num_dirs)`

#### `ToECT(num_dirs=32, num_thresh=128, mode=None, foreground='B', epsilon=0.01)`
Auto-detects color vs grayscale and applies appropriate ECT.

**Shape:**
- Input: `(H, W)`, `(1, H, W)`, or `(3, H, W)`
- Output: Depends on input type and mode

### Symmetry Transforms

#### `RandomGroupAction(group_action, p=1.0)`
Apply random group action from specified group.

**Parameters:**
- `group_action`: A tagtorch GroupAction instance
- `p` (float): Probability of applying transform

**Example:**
```python
from tagtorch.symmetries.groups.group_actions import ContinuousImageRotation

rotation = ContinuousImageRotation()
transform = RandomGroupAction(rotation, p=0.5)
```

#### `GroupActionAugmentation(group_action, mode='random', p=1.0)`
More sophisticated group action application.

**Parameters:**
- `mode` (str): 
  - 'random': Different group element per batch item
  - 'identity': Apply identity (no-op)
  - 'all_same': Same group element for all batch items

**Example:**
```python
transform = GroupActionAugmentation(rotation, mode='all_same')
batch = torch.randn(8, 1, 28, 28)
augmented = transform(batch)  # Same rotation for all 8 images
```

#### `EquivariantAugmentation(group_action, p=1.0)`
Apply same group element to both input and target (for equivariant learning).

**Example:**
```python
transform = EquivariantAugmentation(rotation)
img, target = torch.randn(1, 28, 28), torch.randn(1, 28, 28)
aug_img, aug_target = transform(img, target)
```

### Composition Utilities

#### `Compose(transforms)`
Apply transforms sequentially.

```python
transform = Compose([
    RandomGroupAction(rotation),
    ToECT(num_dirs=32),
])
```

#### `RandomApply(transforms, p=0.5)`
Apply list of transforms with probability p.

```python
transform = RandomApply([
    ToECT(num_dirs=64, num_thresh=256)
], p=0.3)
```

#### `RandomChoice(transforms, p=None)`
Apply one randomly chosen transform.

```python
transform = RandomChoice([
    GrayscaleToECT(num_dirs=32),
    GrayscaleToECT(num_dirs=64),
    GrayscaleToECT(num_dirs=128),
])
```

#### `RandomOrder(transforms)`
Apply transforms in random order.

```python
transform = RandomOrder([
    RandomGroupAction(rotation),
    RandomGroupAction(translation),
])
```

## Functional API

All class transforms delegate to the functional API in `tagtorch.transforms.functional`:

```python
import tagtorch.transforms.functional as TF

# Topological transforms
ect = TF.grayscale_to_ect(img, num_dirs=32, num_thresh=128)
ect = TF.color_to_ect(img, num_dirs=32, mode='channels')
ect = TF.to_ect(img, num_dirs=32)  # Auto-detect

# Symmetry transforms
transformed = TF.apply_group_action(img, group_action, g=None)

# Composition
result = TF.compose_transforms(img, [transform1, transform2])
```

## Advanced Examples

### Multi-Scale ECT

```python
from tagtorch.transforms import Compose, RandomChoice, ToECT

# Randomly use different scales of ECT
transform = Compose([
    RandomChoice([
        ToECT(num_dirs=16, num_thresh=64),   # Coarse
        ToECT(num_dirs=32, num_thresh=128),  # Medium
        ToECT(num_dirs=64, num_thresh=256),  # Fine
    ]),
])
```

### Equivariant Training

```python
from tagtorch.transforms import EquivariantAugmentation
from tagtorch.symmetries.groups.group_actions import DiscreteImageRotation

rotation = DiscreteImageRotation(order=4)
transform = EquivariantAugmentation(rotation, p=0.5)

# In training loop
for img, target in dataloader:
    # Both img and target rotate together
    aug_img, aug_target = transform(img, target)
    output = model(aug_img)
    loss = criterion(output, aug_target)
```

### Custom Transform Pipeline

```python
from tagtorch.transforms import Compose, RandomApply, RandomOrder, RandomGroupAction, ToECT
from tagtorch.symmetries.groups.group_actions import (
    DiscreteImageRotation,
    ImageTranslation,
    ContinuousImageRotation,
)

# Complex augmentation pipeline
transform = Compose([
    # Randomly apply geometric augmentations in random order
    RandomApply([
        RandomOrder([
            RandomGroupAction(DiscreteImageRotation(order=8)),
            RandomGroupAction(ImageTranslation(28, 28)),
        ])
    ], p=0.7),
    
    # Always compute ECT at end
    ToECT(num_dirs=32, num_thresh=128, foreground='B'),
])
```

### GPU Acceleration

```python
import torch
from tagtorch.transforms import ToECT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Specify device for ECT computation
transform = ToECT(num_dirs=64, num_thresh=256, device=device)

# Or move data to GPU
img = torch.randn(1, 28, 28).to(device)
ect = transform(img)  # Computed on GPU
```

## Design Principles

1. **Familiar Interface**: Follows torchvision.transforms patterns
2. **Functional Core**: Pure functions for flexibility, classes for convenience
3. **Composability**: All transforms work with Compose and other utilities
4. **Type Safety**: Clear input/output shapes and types
5. **Documentation**: Comprehensive docstrings and examples
6. **Extensibility**: Easy to add new transforms following the pattern

## Integration with Existing Code

### Migrating from Old API

**Old (topology/transforms.py):**
```python
from tagtorch.topology.transforms import grayscale_image_to_ect

ect = grayscale_image_to_ect(img, num_dirs=32, num_thresh=128)
```

**New (transforms module):**
```python
# Functional API
from tagtorch.transforms import functional as TF
ect = TF.grayscale_to_ect(img, num_dirs=32, num_thresh=128)

# Or class API
from tagtorch.transforms import GrayscaleToECT
transform = GrayscaleToECT(num_dirs=32, num_thresh=128)
ect = transform(img)
```

## Performance Tips

1. **Reuse Transform Objects**: Create once, apply many times
2. **GPU Acceleration**: Specify `device='cuda'` for ECT computation on large images
3. **Batch Processing**: Group actions support batched inputs
4. **Profile First**: ECT computation can be expensive; profile before optimizing

## Future Enhancements

- Persistent homology transforms
- Differentiable ECT (for gradient-based optimization)
- Batched ECT computation
- More group actions (SO(3), product groups, etc.)
- Transform visualization utilities
- Integration with torchvision.transforms v2

## Contributing

To add a new transform:

1. **Implement functional version** in `functional.py`
2. **Create nn.Module wrapper** in appropriate file (topological.py, symmetry.py)
3. **Add to __init__.py** exports
4. **Document with examples** and type hints
5. **Test thoroughly** with different input shapes

## See Also

- [Group Actions Documentation](../symmetries/groups/group_actions/) - Available symmetry groups
- [Model Properties](../model_properties/) - Using transforms with equivariance metrics
