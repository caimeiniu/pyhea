""" @file
@brief Main entry point for the PyHEA (High Entropy Alloys Model) simulation.

This module serves as the main entry point for running the High Entropy Alloys Model simulation.
It handles command-line argument parsing, configuration loading, and initialization of the
simulation components.

@author Caimei Niu
@version 1.0
"""

import time
import argparse

from hea.lattice import lattice
from hea.io import input_config
from hea.model import opt_model
from hea.utils import logger
from hea.comm import comm
from hea.version import __version__

def main():
    """ Main function to run the PyHEA lattice simulation.
    
    This function performs the following operations:
    1. Parses command-line arguments
    2. Loads and validates configuration from YAML file
    3. Initializes the lattice simulation
    4. Displays simulation parameters and setup information
    
    Command-line Arguments:
        --config: Path to the configuration YAML file (default: 'config.yaml')
        --version: Show program's version number and exit
        --help: Show help message and exit
    
    @return None
    """
    # Initialize the argument parser
    parser = argparse.ArgumentParser(
        description="A script to run the HEAM lattice simulation."
    )
    # Add arguments
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration YAML file.')
    # Automatically adds --help functionality.
    # Adding --version option
    parser.add_argument('--version', action='version', version=f'PyHEA {__version__}')
    # Parse the arguments
    args = parser.parse_args()

    logger.info("PyHEA: A High performance implementation for building the High Entropy Alloys Model.")
    logger.info("""\n
             PyHEA  
        ╔═════════════╗
        ║   Fe   Ni   ║   A High performance implementation of building the High Entropy Alloys Models,
        ║      Mn     ║   
        ║   Al   Co   ║   Developed by Caimei Niu.
        ╚═════════════╝
            """
    )

    logger.info(f"Running the PyHEA lattice simulation with the configuration file: {args.config}")
    # Use the parsed argument
    config = input_config(args.config)
    logger.info(f"Configuration loaded successfully.")
    logger.info(f"{'Element types:':<30} {config.element}")
    logger.info(f"{'Element weights:':<30} {config.weight}")
    logger.info(f"{'Cell dimensions:':<30} {config.cell_dim}")
    logger.info(f"{'Number of solutions:':<30} {config.solutions}")
    logger.info(f"{'Number of shells:':<30} {config.max_shell_num}")
    logger.info(f"{'Total iterations:':<30} {config.total_iter}")
    logger.info(f"{'Convergence depth:':<30} {config.converge_depth}")
    logger.info(f"{'Parallel Monte Carlo tasks:':<30} {config.parallel_task}")
    logger.info(f"{'Running with MPI processes:':<30} {comm.Get_size()}")
    logger.info(f"{'Lattice structure:'} {config.structure}\n\n")

    # Continue with the rest of the code using the config
    # No atomic id is needed as the atomic ids of the positions will be shuffled by the code
    lattice_instance = lattice(
        config.cell_dim,
        config.latt_type,
        config.latt_const,
        config.latt_vectors,
        valid=True)

    start = time.time()
    model = opt_model(lattice_instance, config, comm)
    solutions, fitness = model.run_optimization()
    logger.info(f"Total time taken: {time.time() - start} seconds.")

if __name__ == "__main__":
    main()