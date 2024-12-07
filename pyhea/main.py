import os
import time

from pyhea.io import input_config
from pyhea.io.params import parse_args

from pyhea.comm import comm

from pyhea.utils import logger
from pyhea.utils.analyze import analyze_structure, analyze_result

from pyhea.model import opt_model
from pyhea.lattice import lattice

def main():
    """Main function to run the PyHEA lattice simulation."""
    # Parse command line arguments
    args = parse_args()
    
    # Print welcome message
    logger.info("PyHEA: A High performance implementation for building the High Entropy Alloys Models.")
    logger.info("""#nocutoff \n
              PyHEA      
        ╔═══════════════╗   
        ║               ║
        ║    Fe   Ni    ║   A High performance toolkit for High Entropy Alloys Modeling.
        ║       Mn      ║   
        ║    Al   Co    ║ 
        ║               ║   Developed by Caimei Niu.
        ╚═══════════════╝   
    """)

    # Handle different commands       
    if args.command == 'run':
        logger.info(f"Running the PyHEA lattice simulation with the configuration file: {args.config_file}")
        # Use the parsed argument
        config = input_config(args.config_file)
        logger.info(f"Configuration loaded successfully.")
        logger.info(f"{'Element types:':<30} {config.element}")
        logger.info(f"{'Sro shell weights:':<30} {config.weight}")
        logger.info(f"{'Cell dimensions:':<30} {config.cell_dim}")
        logger.info(f"{'Number of solutions:':<30} {config.solutions}")
        logger.info(f"{'Number of shells:':<30} {config.max_shell_num}")
        logger.info(f"{'Total iterations:':<30} {config.total_iter}")
        logger.info(f"{'Convergence depth:':<30} {config.converge_depth}")
        logger.info(f"{'Parallel Monte Carlo tasks:':<30} {config.parallel_task}")
        logger.info(f"{'Running with processes:':<30} {comm.Get_size()}")
        logger.info(f"{'Target SRO:'} {config.target_sro.tolist()}")
        logger.info(f"{'Lattice structure:'} {config.structure}\n\n")

        # Initialize and run the simulation
        lattice_instance = lattice(
            config.cell_dim,
            config.latt_type,
            config.latt_const,
            config.latt_vectors,
            valid=True)
        
        start = time.time()
        model = opt_model(lattice_instance, config, comm)
        solutions, fitness = model.run_optimization()
        logger.info(f"Total time taken: {time.time() - start} seconds.\n\n")

        # Analyze SRO parameters and compare with target values
        logger.info("Post-processing: Analyzing SRO parameters...")
        result_sro, mae, rmse = analyze_result(
            f'{config.output_name}.{config.output_format}',
            config.target_sro,
            config.element,
            config.latt_type
        )
    elif args.command == 'analyze':
        logger.info(f"Analyzing structure file: {args.structure_file}")
        sro_values, output_file = analyze_structure(
            args.structure_file,
            latt_type=args.lattice_type,
            element_types=args.elements,
            output_file=args.output
        )
        logger.info(f"SRO analysis complete. Heatmap saved to: {output_file}")
        logger.info(f"SRO values: {sro_values.tolist()}")
    else:
        raise ValueError(f"Unknown command: {args.command}")

if __name__ == "__main__":
    main()