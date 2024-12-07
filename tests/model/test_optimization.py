import unittest
import numpy as np
from unittest.mock import MagicMock, patch
from pyhea.model import opt
from pyhea.utils.logger import logger
from pyhea.utils.device import get_gpu_info

GPU_AVAILABLE = bool(get_gpu_info())

class TestOptimizationModel(unittest.TestCase):
    def setUp(self):
        # Mock the MPI communicator
        self.mock_comm = MagicMock()
        self.mock_comm.Get_rank.return_value = 0
        self.mock_comm.Get_size.return_value = 1
        
        # Mock lattice object
        self.mock_latt = MagicMock()
        self.mock_latt.shells = 2
        self.mock_latt.nbor_list = [[1, 2], [0, 2], [0, 1]]
        
        # Mock configuration object
        self.mock_conf = MagicMock()
        self.mock_conf.type = 2
        self.mock_conf.solutions = 10
        self.mock_conf.total_iter = 100
        self.mock_conf.parallel_task = 4
        self.mock_conf.converge_depth = 3
        self.mock_conf.device = "gpu"  # Default to CPU
        self.mock_conf.element = ["Fe", "Ni"]
        self.mock_conf.weight = [1.0, 0.5]
        self.mock_conf.target_sro = np.array([0.1, 0.2])
        self.mock_conf.output_format = "vasp/poscar"
    
    def test_model_initialization(self):
        """Test model initialization with CPU configuration"""
        model = opt.opt_model(self.mock_latt, self.mock_conf, self.mock_comm)
        self.assertIsNotNone(model)
        self.assertEqual(model.ntyp, 2)
        self.assertEqual(model.weight, [1.0, 0.5])
        self.assertEqual(model.device, "gpu")
    
    @patch('pyhea.model.opt.acc')
    def test_gpu_optimization(self, mock_acc):
        """Test GPU optimization with CUDA available"""
        if not GPU_AVAILABLE:
            self.skipTest("CUDA is not available on this system")
        
        # Configure for GPU
        mock_acc.cuda_available.return_value = True
        mock_acc.run_local_parallel_hcs_cuda.return_value = (
            [[1, 0]],  # lattices (single configuration)
            [0.05]  # fitness value
        )
        
        # Mock the file writing
        with patch('pyhea.model.opt.file_final_results') as mock_write:
            model = opt.opt_model(self.mock_latt, self.mock_conf, self.mock_comm)
            model.run_optimization()
            
            # Verify GPU-specific calls
            mock_acc.cuda_available.assert_called_once()
            
            # Get the actual call arguments
            call_args = mock_acc.run_local_parallel_hcs_cuda.call_args[0]
            
            # Verify each argument individually
            self.assertEqual(call_args[0], 10)  # solutions
            self.assertEqual(call_args[1], 100)  # total_iter
            self.assertEqual(call_args[2], 4)  # parallel_task
            self.assertEqual(call_args[3], 3)  # converge_depth
            self.assertEqual(call_args[4], 0.001)  # tolerance
            self.assertEqual(call_args[5], self.mock_latt.nbor_list)  # neighbor list
            self.assertEqual(call_args[6], ["Fe", "Ni"])  # elements
            self.assertEqual(call_args[7], [1.0, 0.5])  # weights
            self.assertTrue(np.all(call_args[8] == np.array([0.1, 0.2])))  # target SRO
            
            # Verify file writing was called
            mock_write.assert_called_once()
    
    @patch('pyhea.model.opt.acc')
    def test_gpu_no_cuda(self, mock_acc):
        """Test GPU optimization when CUDA is not available"""
        if not GPU_AVAILABLE:
            self.skipTest("CUDA is not available on this system")

        mock_acc.cuda_available.return_value = False
        
        model = opt.opt_model(self.mock_latt, self.mock_conf, self.mock_comm)
        with self.assertRaises(SystemExit):
            model.run_optimization()
    
    @patch('pyhea.model.opt.acc')
    def test_gpu_multiple_processes(self, mock_acc):
        """Test GPU optimization with multiple MPI processes (should fail)"""
        if not GPU_AVAILABLE:
            self.skipTest("CUDA is not available on this system")

        self.mock_comm.Get_size.return_value = 4  # Simulate 4 processes
        mock_acc.cuda_available.return_value = True
        
        model = opt.opt_model(self.mock_latt, self.mock_conf, self.mock_comm)
        with self.assertRaises(SystemExit):
            model.run_optimization()
    
    @patch('pyhea.model.opt.acc')
    def test_cpu_optimization(self, mock_acc):
        """Test CPU optimization with accelerate module available"""
        # Configure for CPU
        self.mock_conf.device = "cpu"
        
        # Mock the CPU optimization results
        mock_acc.run_local_parallel_hcs.return_value = (
            [[1, 0]],  # lattices (single configuration)
            [0.05]  # fitness value
        )
        
        # Mock MPI gather operations
        self.mock_comm.gather.side_effect = [
            [[[1, 0]]],  # lattices from all processes
            [[0.05]]  # fitness values from all processes
        ]
        
        # Mock the file writing
        with patch('pyhea.model.opt.file_final_results') as mock_write:
            model = opt.opt_model(self.mock_latt, self.mock_conf, self.mock_comm)
            latts, fitss = model.run_optimization()
            
            # Verify CPU-specific calls
            mock_acc.run_local_parallel_hcs.assert_called_once()
            
            # Get the actual call arguments
            call_args = mock_acc.run_local_parallel_hcs.call_args[0]
            
            # Verify each argument individually
            self.assertEqual(call_args[0], 10)  # solutions
            self.assertEqual(call_args[1], 100)  # total_iter
            self.assertEqual(call_args[2], 4)  # parallel_task
            self.assertEqual(call_args[3], 3)  # converge_depth
            self.assertEqual(call_args[4], 0.001)  # tolerance
            self.assertEqual(call_args[5], self.mock_latt.nbor_list)  # neighbor list
            self.assertEqual(call_args[6], ["Fe", "Ni"])  # elements
            self.assertEqual(call_args[7], [1.0, 0.5])  # weights
            self.assertTrue(np.all(call_args[8] == np.array([0.1, 0.2])))  # target SRO
            
            # Verify MPI operations
            self.mock_comm.gather.assert_called()
            self.assertEqual(self.mock_comm.gather.call_count, 2)  # Once for latts, once for fitss
            
            # Verify results were properly processed
            self.assertEqual(latts, [[1, 0]])  # Fixed: Expect nested list
            self.assertEqual(fitss, [0.05])
            
            # Verify file writing was called
            mock_write.assert_called_once()

    @patch('pyhea.model.opt.acc')
    def test_cpu_no_accelerate(self, mock_acc):
        """Test CPU optimization when accelerate module is not available"""
        # Configure for CPU and set acc to None
        self.mock_conf.device = "cpu"
        
        # Temporarily replace acc with None
        with patch('pyhea.model.opt.acc', None):
            model = opt.opt_model(self.mock_latt, self.mock_conf, self.mock_comm)
            with self.assertRaises(SystemExit):
                model.run_optimization()

    @patch('pyhea.model.opt.acc')
    def test_cpu_parallel_optimization(self, mock_acc):
        """Test CPU optimization with multiple MPI processes"""
        # Configure for CPU and multiple processes
        self.mock_conf.device = "cpu"
        self.mock_comm.Get_size.return_value = 4
        self.mock_comm.Get_rank.return_value = 1  # Non-root process
        
        # Mock the CPU optimization results for this process
        mock_acc.run_local_parallel_hcs.return_value = (
            [[1, 0]],  # lattices from this process
            [0.05]  # fitness from this process
        )
        
        # Mock MPI gather operations for non-root process
        self.mock_comm.gather.return_value = [None]  # Non-root processes receive None
        
        model = opt.opt_model(self.mock_latt, self.mock_conf, self.mock_comm)
        latts, fitss = model.run_optimization()
        
        # Verify behavior specific to non-root processes
        self.assertEqual(latts, [None])
        self.assertEqual(fitss, [None])
        
        # Verify MPI operations were called
        self.mock_comm.gather.assert_called()
        self.assertEqual(self.mock_comm.gather.call_count, 2)

    @patch('pyhea.model.opt.acc')
    def test_invalid_device(self, mock_acc):
        """Test optimization with invalid device specification"""
        self.mock_conf.device = "invalid_device"
        
        model = opt.opt_model(self.mock_latt, self.mock_conf, self.mock_comm)
        with self.assertRaises(SystemExit):
            model.run_optimization()

    def test_file_final_results(self):
        """Test writing final results to file"""
        with patch('pyhea.model.opt.write_structure') as mock_write:
            opt.file_final_results(
                nest=[1, 0, 1],
                latt=self.mock_latt,
                ntyp=2,
                elem=["Fe", "Ni"],
                file="test.vasp",
                output_format="vasp/poscar"
            )
            mock_write.assert_called_once_with(
                [1, 0, 1],
                self.mock_latt,
                2,
                ["Fe", "Ni"],
                "test.vasp",
                "vasp/poscar"
            )

if __name__ == '__main__':
    unittest.main()
