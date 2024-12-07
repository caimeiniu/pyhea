# PyHEA: A High Performance High-Entropy Alloys Modeling Toolkit

[![License](https://img.shields.io/badge/License-LGPL3.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macos-lightgrey)](https://github.com/caimeiniu/pyhea)

PyHEA is a high-performance computational toolkit for modeling and optimizing High Entropy Alloys (HEAs). It provides a comprehensive suite of tools for atomic structure simulation, property prediction, and optimization of multi-component alloy systems.

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
- Python >= 3.7
- NumPy >= 1.19.0
- SciPy >= 1.7.0
- PyYAML >= 5.1.0
- mpi4py >= 3.0.0
- pybind11 >= 2.6.0

### Optional Dependencies
- CUDA Toolkit (for GPU acceleration)
- Matplotlib >= 3.7.0 (for visualization)
- Seaborn >= 0.12.0 (for advanced plotting)
- dpdata >= 0.2.0 (for data processing)
- WarrenCowleyParameters >= 2.0.0 (for structure analysis)

## Installation

### Build from Source

```bash
# Clone the repository
git clone https://github.com/caimeiniu/pyhea.git
cd pyhea

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

1. **Basic Configuration**

Create a configuration file `config.yaml`:

```yaml
type: 4                    # Number of element types
element:                   # Number of atoms per element
  - 64000
  - 64000
  - 64000
  - 64000
cell_dim:                 # Supercell dimensions
  - 40
  - 40
  - 40
device: cpu               # Computation device (cpu/gpu)
solutions: 128            # Number of parallel solutions
total_iter: 10           # Total optimization iterations
parallel_task: 256       # Number of parallel MC tasks
converge_depth: 10       # Convergence criterion
weight:                  # Shell weights for fitness
  - 4.0
  - 1.0
  - 0.5
structure: POSCAR_FCC    # Input structure file
```

2. **Running Simulations**

```bash
# Run simulation with configuration file
python -m pyhea run config.yaml

# Run with MPI parallel processing
mpirun -np 4 python -m pyhea run config.yaml

# Analyze structure
python -m pyhea analyze structure.lmp --format lmp --lattice-type FCC --elements Fe Ni Cr
python -m pyhea analyze POSCAR --format poscar --lattice-type BCC --elements Al Ti V

# Check version
python -m pyhea --version
```

3. **Analyzing Results**

```python
from pyhea.utils.analyze import analyze_structure
from pyhea.io.input import read_structure

# Load and analyze structure
structure = read_structure("output_structure.xyz")
results = analyze_structure(structure)
```

## Performance Benchmarks

| System Size | CPU (1 core) | CPU (24 cores) | GPU (A100) |
|-------------|-------------|----------------|------------|
| 20x20x20    | 495.72s     | 20.55s         | 5.04s      |
| 40x40x40    | 3945.81s    | 164.41s        | 40.43s     |
| 60x60x60    | 13245.63s   | 551.90s        | 135.84s    |

## Advanced Usage

### Custom Fitness Functions

```python
from pyhea.fitness import BaseFitness

class CustomFitness(BaseFitness):
    def calculate(self, structure):
        # Implement your custom fitness calculation
        pass
```

### Structure Analysis

```python
from pyhea.utils.analyze import calculate_wcps

# Calculate Warren-Cowley parameters
wcps = calculate_wcps(structure, max_shell=3)
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
@software{pyhea2024,
  author = {Niu, Caimei},
  title = {PyHEA: A High Performance High-Entropy Alloys Model Builder},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/caimeiniu/pyhea},
  note = {Manuscript in preparation.}
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

## Contact

- Issue Tracker: [GitHub Issues](https://github.com/caimeiniu/pyhea/issues)
- Documentation: [Read the Docs](https://pyhea.readthedocs.io/)
