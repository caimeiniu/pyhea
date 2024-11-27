# PyHEA: High Performance High Entropy Alloys Model Builder

[![License](https://img.shields.io/badge/License-LGPL3.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macos-lightgrey)](https://github.com/yourusername/pyhea)

A high-performance implementation for building and optimizing High Entropy Alloys (HEA) models, featuring both CPU and GPU acceleration capabilities.

## Overview

PyHEA is designed for efficient simulation and optimization of High Entropy Alloys, providing:

- 🚀 High-performance parallel processing with MPI
- 💻 Hybrid CPU/GPU computation support
- ⚡ Incremental fitness calculation for improved efficiency
- 🔮 Support for both BCC and FCC lattice structures
- 🛠 Flexible YAML-based configuration system

## Installation

### Prerequisites

```bash
# Required
Python >= 3.6
C++ compiler with C++11 support
CMake >= 3.18
MPI implementation (OpenMPI/MPICH)

# Optional
CUDA Toolkit (for GPU acceleration)
```

### Installing from PyPI

```bash
pip install pyhea
```

### Installing from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/pyhea.git
cd pyhea

# Install in development mode
pip install -e .
```

### Installing from GitHub

```bash
# Latest version
pip install git+https://github.com/yourusername/pyhea.git

# Specific version
pip install git+https://github.com/yourusername/pyhea.git@v1.0.0
```

## Quick Start

1. Create your configuration file (input.yaml):

```yaml
type: 4
element:
  - 64000
  - 64000
  - 64000
  - 64000
cell_dim:
  - 40
  - 40
  - 40
device: cpu  # or 'gpu' for CUDA acceleration
solutions: 128
total_iter: 10
parallel_task: 256
converge_depth: 10
weight:
  - 4.0
  - 1.0
  - 0.5
structure: POSCAR_FCC
```

2. Run the optimization:

```bash
# Run with 4 MPI processes
mpirun -np 4 python -m hea.main --config input.yaml
```

## Configuration Guide

| Parameter | Description | Type |
|-----------|-------------|------|
| type | Number of element types | int |
| element | List of element counts | List[int] |
| cell_dim | Supercell dimensions | List[int] |
| device | Computation device ('cpu'/'gpu') | str |
| solutions | Number of parallel solutions | int |
| total_iter | Total optimization iterations | int |
| parallel_task | Number of parallel MC tasks | int |
| converge_depth | Convergence criterion depth | int |
| weight | Shell weights for fitness | List[float] |
| structure | Path to POSCAR file | str |

## Performance

PyHEA achieves high performance through:

1. MPI-based parallel processing
2. GPU acceleration (with CUDA)
3. Incremental fitness calculation
4. Optimized neighbor list management

Example performance comparison for a 40x40x40 supercell:
- CPU (single core): 3945.81s
- CPU (24 cores): 164.41s
- GPU (NVIDIA A100): 40.43s

## Development

```bash
# Clone repository
git clone https://github.com/yourusername/pyhea.git
cd pyhea

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
.\venv\Scripts\activate  # Windows

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```

## Contributing

1. Fork the repository
2. Create your feature branch
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Commit your changes
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. Push to the branch
   ```bash
   git push origin feature/AmazingFeature
   ```
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use PyHEA in your research, please cite:

```bibtex
@software{pyhea2024,
  author = {Niu, Charmy},
  title = {PyHEA: High Performance High Entropy Alloys Model Builder},
  year = {2024},
  url = {https://github.com/yourusername/pyhea}
}
```

## Troubleshooting

Common issues and solutions:

1. CUDA not detected
   ```bash
   # Check CUDA installation
   nvcc --version
   # Ensure CUDA_HOME is set
   echo $CUDA_HOME
   ```

2. MPI errors
   ```bash
   # Check MPI installation
   mpirun --version
   # Test MPI
   mpirun -np 2 hostname
   ```

3. Build failures
   ```bash
   # Check CMake version
   cmake --version
   # Required: >= 3.18
   ```

## Roadmap

- [ ] Add support for more lattice types
- [ ] Implement adaptive convergence criteria
- [ ] Add visualization tools
- [ ] Improve GPU memory management
- [ ] Add support for distributed GPU computing
