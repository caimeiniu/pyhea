from hea.utils.logger import logger
try:
    from hea.cpp import accelerate as acc
except ImportError:
    logger.error("Failed to import accelerate module. Please ensure the package is properly installed.")
    raise

def file_final_results(nest, latt, ntyp, elem, file):
    natm = len(nest)
    coords = [""] * ntyp  # List to store coordinate strings for each type of atom
    
    with open(file, "w") as final_coords:  # Writes the final SCRAPS supercell to the specified output file
        final_coords.write("Generated-by-PyHEA\tv0.1\n")
        final_coords.write(f"{latt.latt_vec[0]}\t{latt.latt_vec[0]}\t{latt.latt_vec[0]}\n")
        final_coords.write(f"{latt.latt_vec[1]}\t{latt.latt_vec[1]}\t{latt.latt_vec[1]}\n")
        final_coords.write(f"{latt.latt_vec[2]}\t{latt.latt_vec[2]}\t{latt.latt_vec[2]}\n")

        final_coords.write("   ".join(map(str, elem)) + "\n")
        final_coords.write("Cartesian\n")
        
        for i in range(natm):
            coords[nest[i]] += f"{latt.coords[i][0]}\t{latt.coords[i][1]}\t{latt.coords[i][2]}\n"
        
        for i in range(ntyp):
            final_coords.write(coords[i])

class opt_model:
    def __init__(self, latt, conf, comm):
        self.latt = latt
        self.comm = comm

        self.thr  = 0.001
        self.ntyp = conf.type
        self.nnet = conf.solutions
        self.nbor = latt.nbor_list
        self.step = conf.total_iter
        self.task = conf.parallel_task
        
        self.depth   = conf.converge_depth
        self.device  = conf.device
        self.element = conf.element

        if (len(conf.weight) > latt.shells):
            logger.error("The number of weight is larger than the maximum number(shell) of neighbors.")
            exit()
        self.weight = conf.weight
        
        # Save self.nbor into a txt file
        with open("nbor.txt", "w") as nbor_file:
            for shell in self.nbor:
                nbor_file.write(" ".join(map(str, shell)) + "\n")

    def run_optimization(self):
        """
        Run the Local Parallel HCS optimization algorithm.
        """
        logger.info("Running Local Parallel HCS optimization...")
        
        if (self.device == "cpu"):
            logger.info("Running on CPU devices")
            # Run the Local Parallel HCS algorithm
            latts, fitss = acc.run_local_parallel_hcs(
                self.nnet, self.step, self.task, self.depth, self.thr, self.nbor, self.element, self.weight)
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
            if (acc.cuda_available() == False):
                logger.error("CUDA is not available on this machine. Please run on a machine with CUDA support.")
                exit()
            logger.info("Running on GPU devices")
            # Run the Local Parallel HCS algorithm
            latts, fitss = acc.run_local_parallel_hcs_cuda(
                self.nnet, self.step, self.task, self.depth, self.thr, self.nbor, self.element, self.weight)
        else:
            logger.error("Invalid device type. Please specify either 'cpu' or 'gpu'.")
            exit()

        # Print the final results
        logger.info("Optimization completed with final fitness values: \n{}\n".format(fitss))
        logger.info("Best fitness value: {}".format(fitss[0]))

        # Write the final lattice structure to a file
        if (self.comm.Get_rank() == 0):
            file_final_results(latts[0], self.latt, self.ntyp, self.element, "FINAL_POSCAR")
        logger.info("Final lattice structure written to file FINAL_POSCAR.")
        # Return the optimized latts and fitness values
        return latts, fitss