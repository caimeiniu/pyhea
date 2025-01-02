# PyHEA: A Short-Range Order Based High-Performance High-Entropy Alloys Modeling Toolkit

[![License](https://img.shields.io/badge/License-LGPL3.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macos-lightgrey)](https://github.com/caimeiniu/pyhea)

PyHEA is a Short-Range Order Based high-performance computational toolkit for modeling and optimizing High Entropy Alloys (HEAs). It provides a comprehensive suite of tools for atomic structure simulation, property prediction, and optimization of multi-component alloy systems.
 
## Key Features

- **High Performance Computing**
  - MPI-based parallel processing for distributed computing
  - Optimized neighbor list algorithms
  - Incremental fitness calculation for improved efficiency

- **Comprehensive Modeling**
  - Support for both BCC and FCC lattice structures
  - Warren-Cowley parameters calculation
  - Multi-component alloy system optimization
  - Customizable fitness functions

- **User-Friendly Design**
  - YAML-based configuration system
  - Flexible input/output formats
  - Extensive documentation and examples
  - Built-in visualization tools

## Requirements

### Core Dependencies
- Python >= 3.9
- NumPy >= 1.19.0
- SciPy >= 1.7.0
- PyYAML >= 5.1.0
- mpi4py >= 3.0.0
- pybind11 >= 2.6.0
- Matplotlib >= 3.7.0 (for visualization)
- dpdata >= 0.2.0 (for data processing)
- WarrenCowleyParameters >= 2.0.0 (for structure analysis)

### Optional Dependencies
- CUDA Toolkit (for GPU acceleration)

## Installation

### Build from Source

```bash
# Clone the repository
git clone https://github.com/caimeiniu/pyhea.git
cd pyhea

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install .
```

## Quick Start

1. **Basic Configuration**

Create a configuration file `config.yaml`:

```yaml
type: 4                    # Number of element types
element:                   # Number of atoms per element
  - 1000
  - 1000
  - 1000
  - 1000
cell_dim:                 # Supercell dimensions
  - 10
  - 10
  - 10
device: cpu               # Computation device (cpu/gpu)
solutions: 128            # Number of parallel solutions
total_iter: 10           # Total optimization iterations
parallel_task: 256       # Number of parallel MC tasks
converge_depth: 10       # Convergence criterion
weight:                  # Shell weights for fitness
  - 4.0
  - 1.0
  - 0.5
target_sro: TARGET_SRO   # Path to the target SRO file
structure: POSCAR_FCC    # Input structure file
```

2. **Running Simulations**

```bash
# Run simulation with configuration file
pyhea run config.yaml

# Run with MPI parallel processing
mpirun -np 4 pyhea run config.yaml

# Check version
pyhea --version
```

## Advanced Usage

### Structure Analysis

# Analyze structure 
Analyze the atomic distribution and generate a heatmap of the elements' short-range order (SRO) values.
```bash
pyhea analyze structure.lmp --format lmp --lattice-type FCC --elements Fe Ni Cr Co

```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the LGPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use PyHEA in your research, please cite:

```bibtex
@article{niu2024short,
  title={Short-Range Order Based Ultra Fast Large-Scale Modeling of High-Entropy Alloys},
  author={Niu, Caimei and Liu, Lifeng},
  journal={arXiv preprint arXiv:2411.18906},
  year={2024}
}
```

## Troubleshooting

### Common Issues

1. **CUDA Issues**
   ```bash
   # Verify CUDA installation
   nvcc --version
   # Check CUDA environment
   echo $CUDA_HOME
   ```

2. **MPI Errors**
   ```bash
   # Check MPI installation
   mpirun --version
   # Test MPI functionality
   mpirun -np 2 hostname
   ```

3. **Installation Problems**
   - Ensure all dependencies are installed
   - Check Python version compatibility
   - Verify compiler settings in setup.py

For more detailed troubleshooting, please visit our [documentation](docs/troubleshooting.md).

## Getting Help

If you need help with PyHEA, you can use the following resources:

- Issue Tracker: [GitHub Issues](https://github.com/caimeiniu/pyhea/issues)
- Community Forum: [PyHEA Discussions](https://github.com/caimeiniu/pyhea/discussions)
- Email Support: [support@pyhea.org](mailto:caimeiniu@stu.pku.edu.cn)

## Contact

- Issue Tracker: [GitHub Issues](https://github.com/caimeiniu/pyhea/issues)
