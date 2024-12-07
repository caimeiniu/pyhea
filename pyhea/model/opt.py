from pyhea.utils.logger import logger
from pyhea.version import __version__  # Import version information
from pyhea.io.output import write_structure
try:
    from pyhea.cpp import accelerate as acc
except ImportError:
    logger.warning("Failed to import accelerate module: No module named 'pyhea.cpp.accelerate'\nThis might happen if the C++ extension was not properly built.")
    acc = None

def file_final_results(nest, latt, ntyp, elem, file, output_format='vasp/poscar'):
    """Write the final results to the specified output format.
    
    @param nest list List of atom types
    @param latt Lattice Lattice object containing vectors and coordinates
    @param ntyp int Number of atom types
    @param elem list List of elements (will be replaced with A, B, C, ...)
    @param file str Output file name
    @param output_format str Desired output format (default: 'vasp/poscar')
                           Supported formats: vasp/poscar, lammps/lmp
    @return None
    """
    write_structure(nest, latt, ntyp, elem, file, output_format)

class opt_model:
    def __init__(self, latt, conf, comm):
        """Initialize the optimization model.
        
        @param latt Lattice Lattice object containing structure information
        @param conf Config Configuration object containing simulation parameters
        @param comm Comm Communication object for parallel processing
        @return None
        """
        self.latt = latt
        self.comm = comm
        self.conf = conf  # Store conf for access to output format

        self.thr  = 0.001
        self.ntyp = conf.type
        self.nnet = conf.solutions
        self.nbor = latt.nbor_list
        self.step = conf.total_iter
        self.task = conf.parallel_task
        
        self.depth   = conf.converge_depth
        self.device  = conf.device
        self.element = conf.element
        self.target_sro = conf.target_sro

        if (len(conf.weight) > latt.shells):
            logger.error("The number of weight is larger than the maximum number(shell) of neighbors.")
            exit()
        self.weight = conf.weight
        
    def run_optimization(self):
        """Run the Local Parallel HCS optimization algorithm.
        
        Executes the High-throughput Computing Screening (HCS) optimization
        algorithm in parallel to find the optimal configuration for the high entropy alloy.
        
        @return tuple (list, list) Returns optimized lattice structures and their fitness values
        """
        logger.info("Running Local Parallel HCS optimization...")
        
        if (self.device == "cpu"):
            logger.info("Running on CPU devices")
            # Run the Local Parallel HCS algorithm
            if acc is not None:
                latts, fitss = acc.run_local_parallel_hcs(
                    self.nnet, self.step, self.task, self.depth, self.thr, self.nbor, self.element, self.weight)
            else:
                logger.error("Accelerate module is not available. Please ensure the package is properly installed.")
                exit()
            # Alltogther the results from all processes into rank 0 and reshape the results
            latts = self.comm.gather(latts, root=0)
            fitss = self.comm.gather(fitss, root=0)
            if (self.comm.Get_rank() == 0):
                latts = [latt for latts_rank in latts for latt in latts_rank]
                fitss = [fits for fits_rank in fitss for fits in fits_rank]
                latts = [latt for _, latt in sorted(zip(fitss, latts), key=lambda pair: pair[0])]
                fitss = sorted(fitss)
            else :
                latts = [None]
                fitss = [None]
        elif (self.device == "gpu"):
            if (self.comm.Get_size() > 1):
                logger.error("Cannot run on multiple GPUs. Please run on a single GPU.")
                exit()
            if acc is not None and acc.cuda_available() == True:
                logger.info("Running on GPU devices")
                # Run the Local Parallel HCS algorithm
                latts, fitss = acc.run_local_parallel_hcs_cuda(
                    self.nnet, self.step, self.task, self.depth, self.thr, self.nbor, self.element, self.weight, self.target_sro)
            else:
                logger.error("CUDA is not available on this machine or accelerate module is not available. Please ensure the package is properly installed and CUDA support is available.")
                exit()
        else:
            logger.error("Invalid device type. Please specify either 'cpu' or 'gpu'.")
            exit()

        # Print the final results
        logger.info("Optimization completed with final fitness values: \n{}\n".format(fitss))
        logger.info("Best fitness value: {}".format(fitss[0]))

        # Write the final lattice structure to a file
        if (self.comm.Get_rank() == 0):
            output_format = getattr(self.conf, 'output_format', 'vasp/poscar')
            output_name = getattr(self.conf, 'output_name', 'output')
            
            # Determine output filename based on format
            if output_format == 'lammps/lmp':
                output_file = f"{output_name}.lmp"
            else:  # vasp/poscar
                output_file = f"{output_name}.poscar"
            
            file_final_results(latts[0], self.latt, self.ntyp, self.element, output_file, output_format=output_format)
            logger.info(f"Final lattice structure saved to file {output_file}")
        
        # Return the optimized latts and fitness values
        return latts, fitss