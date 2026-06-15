# `TAGTorch`: A PyTorch Library for Geometry, Topology, and Symmetry-Aware Machine Learning

TAGTorch provides a comprehensive suite of tools for analyzing and leveraging symmetry, equivariance, and topological properties in PyTorch deep learning models. The library offers:

- **Topological Transforms**: Euler Characteristic Transform (ECT) and persistent homology for robust geometric descriptors
- **Symmetry Analysis**: Group action-based data augmentation and equivariance metrics
- **PyTorch Integration**: Familiar transforms API inspired by torchvision, seamlessly integrating with PyTorch workflows

## Installation

TAGTorch requires Python 3.10 or later. Choose one of the following installation methods:

### Option 1: Using `uv` (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync

# Install with optional dependencies (topology)
uv sync --extra topology

# Install with all optional
uv sync --all-extras
```

### Option 2: Using `pip` with `virtualenv`

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install tagtorch
pip install -e .

# Install with optional dependencies
pip install -e ".[topology,timeout]"
```

### Option 3: Using `conda` with `pip`

```bash
# Create a conda environment
conda create -n tagtorch python=3.10
conda activate tagtorch

# Install tagtorch with pip
pip install -e .

# Install with optional dependencies
pip install -e ".[topology,timeout]"
```

## Quick Start

### Topological Transforms

Transform images into robust topological descriptors using the Euler Characteristic Transform:

```python
import torch
from tagtorch.transforms import ToECT

# Create ECT transform
transform = ToECT(num_dirs=32, num_thresh=128, foreground='B')

# Apply to image (e.g., MNIST)
img = torch.randn(1, 28, 28)  # Grayscale image
ect = transform(img)
print(ect.shape)  # torch.Size([1, 128, 32])
```

### Symmetry-Based Augmentation

Apply group action-based data augmentation for equivariant learning:

```python
from tagtorch.transforms import RandomGroupAction, Compose
from tagtorch.symmetries.groups.group_actions import DiscreteImageRotation

# Create rotation group (0°, 90°, 180°, 270°)
rotation = DiscreteImageRotation(order=4)

# Build augmentation pipeline
transform = Compose([
    RandomGroupAction(rotation, p=0.5),
    ToECT(num_dirs=32, num_thresh=128),
])

img = torch.randn(1, 28, 28)
augmented = transform(img)
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

# Build training pipeline
train_transform = Compose([
    RandomGroupAction(rotation, p=0.5),
    RandomGroupAction(translation, p=0.3),
    ToECT(num_dirs=32, num_thresh=128, foreground='B'),
])

# Use with DataLoader
train_dataset = MyDataset(transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

## Features

### 🔄 PyTorch-Style Transforms API

TAGTorch provides a familiar interface inspired by `torchvision.transforms`:

- **Functional API**: Pure stateless functions for maximum flexibility
- **Class API**: Composable, serializable `nn.Module` transforms
- **Composition utilities**: `Compose`, `RandomApply`, `RandomChoice`, `RandomOrder`

```python
import tagtorch.transforms.functional as TF

# Functional API
ect = TF.grayscale_to_ect(img, num_dirs=32, num_thresh=128)

# Class API
from tagtorch.transforms import GrayscaleToECT
transform = GrayscaleToECT(num_dirs=32, num_thresh=128)
ect = transform(img)
```

### 📐 Topological Transforms

- **Euler Characteristic Transform (ECT)**: Robust geometric descriptor for images
- **Grayscale and Color Support**: Automatic detection or explicit mode selection
- **Persistent Homology**: Tools for computing topological features

Available transforms:
- `GrayscaleToECT`: Convert grayscale images to ECT
- `ColorToECT`: Process RGB images (grayscale or per-channel)
- `ToECT`: Auto-detect image type and apply appropriate ECT

### 🔁 Symmetry & Group Actions

Leverage group theory for data augmentation and equivariance analysis:

- **Discrete & Continuous Rotations**: Image rotation groups
- **Translations**: Spatial translation groups
- **Reflections & Permutations**: Various symmetry operations
- **Custom Group Actions**: Extensible framework for defining new symmetries

Available group actions:
- `DiscreteImageRotation`: Discrete rotations (e.g., 90° increments)
- `ContinuousImageRotation`: Arbitrary angle rotations
- `ImageTranslation`: Spatial translations
- `ImageReflection`: Horizontal/vertical reflections

### 📊 Equivariance Metrics

Measure and analyze equivariance properties of neural networks:

```python
from tagtorch.model_properties import equivariance_metrics

# Measure how equivariant your model is to rotations
score = equivariance_metrics.compute_equivariance(
    model, 
    data, 
    group_action=rotation
)
```

### ⚙️ Hyperparameter Selection

Tools for selecting optimal hyperparameters for topological transforms:

- Runtime analysis for different parameter configurations
- Persistent homology-based quality metrics
- Automated parameter search with timeouts

## Project Structure

```
tagtorch/
├── transforms/          # PyTorch-style transforms API
│   ├── functional.py    # Functional transform implementations
│   ├── topological.py   # ECT and topological transforms
│   ├── symmetry.py      # Group action augmentations
│   └── compose.py       # Composition utilities
├── symmetries/          # Group theory and symmetry analysis
│   └── groups/          # Group implementations and actions
├── topology/            # Topological data analysis tools
│   ├── ect/             # Euler Characteristic Transform
│   ├── ph/              # Persistent homology utilities
│   └── hyperparameters/ # Parameter selection tools
├── model_properties/    # Neural network analysis
│   └── equivariance_metrics.py
```

## Documentation

For detailed documentation on specific modules:

- **[Transforms API](src/tagtorch/transforms/README.md)**: Comprehensive guide to the transforms module
- **[Group Actions](src/tagtorch/symmetries/groups/group_actions/)**: Available symmetry groups

### Contributing

We welcome contributions! Areas for enhancement include:

- Additional topological transforms (e.g., differentiable ECT)
- New group actions (e.g., SO(3), product groups)
- Performance optimizations for batch processing
- Additional examples and documentation

## Citation

If you use TAGTorch in your research, please cite:

```bibtex
@software{tagtorch2026,
  title={TAGTorch: Tools for Analyzing Geometry, Topology, and Symmetry in PyTorch},
  author={Kennedy, Brendan and Kvinge, Henry and Roek, Greg and Purvine, Emilie and Emerson, Tegan},
  year={2026},
  organization={Pacific Northwest National Laboratory}
}
```

## Authors

- **Brendan Kennedy** - brendan.kennedy@pnnl.gov
- **Henry Kvinge** - henry.kvinge@pnnl.gov
- **Greg Roek** - gregory.roek@pnnl.gov
- **Emilie Purvine** - emilie.purvine@pnnl.gov
- **Tegan Emerson** - tegan.emerson@pnnl.gov

## License (BSD-2)

Copyright Battelle Memorial Institute 2026
 
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
 
1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Disclaimer

This material was prepared as an account of work sponsored by an agency of the United States Government.  Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or
represents that its use would not infringe privately owned rights.
 
Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.
 
                 PACIFIC NORTHWEST NATIONAL LABORATORY
                              operated by
                                BATTELLE
                                for the
                   UNITED STATES DEPARTMENT OF ENERGY
                    under Contract DE-AC05-76RL01830