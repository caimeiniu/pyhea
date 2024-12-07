import unittest
import os
import numpy as np
from unittest.mock import patch, MagicMock
from pyhea.main import main
from pyhea.io.params import parse_args

class TestMain(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.test_config_path = os.path.join(os.path.dirname(__file__), '../examples/config.yaml')
        self.test_structure_file = os.path.join(os.path.dirname(__file__), '../examples/structure.xyz')
        
        # Common mock configuration
        self.mock_config = MagicMock()
        self.mock_config.element = ['Fe', 'Ni']
        self.mock_config.weight = [1.0, 0.5]
        self.mock_config.cell_dim = [10, 10, 10]
        self.mock_config.solutions = 1
        self.mock_config.max_shell_num = 2
        self.mock_config.total_iter = 1000
        self.mock_config.converge_depth = 100
        self.mock_config.parallel_task = 1
        self.mock_config.target_sro = np.array([0.1, 0.2])
        self.mock_config.latt_type = 'BCC'
        self.mock_config.latt_const = 1.0
        self.mock_config.latt_vectors = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]], dtype=np.float16)
        self.mock_config.output_name = 'test_output'
        self.mock_config.output_format = 'xyz'
    
    @patch('pyhea.main.parse_args')
    @patch('pyhea.main.input_config')
    @patch('pyhea.main.lattice')
    @patch('pyhea.main.opt_model')
    @patch('pyhea.main.analyze_result')
    def test_main_run_command(self, mock_analyze_result, mock_opt_model, mock_lattice, 
                             mock_input_config, mock_parse_args):
        """Test the 'run' command functionality"""
        # Mock command line arguments
        mock_args = MagicMock()
        mock_args.command = 'run'
        mock_args.config_file = self.test_config_path
        mock_parse_args.return_value = mock_args
        
        # Set up mock returns
        mock_input_config.return_value = self.mock_config
        mock_lattice_instance = MagicMock()
        mock_lattice.return_value = mock_lattice_instance
        
        mock_model_instance = MagicMock()
        mock_model_instance.run_optimization.return_value = ([], 0.0)
        mock_opt_model.return_value = mock_model_instance
        
        mock_analyze_result.return_value = (np.array([0.1, 0.2]), 0.01, 0.02)
        
        # Run main function
        main()
        
        # Verify all mock calls
        mock_parse_args.assert_called_once()
        mock_input_config.assert_called_once_with(self.test_config_path)
        mock_lattice.assert_called_once_with(
            self.mock_config.cell_dim,
            self.mock_config.latt_type,
            self.mock_config.latt_const,
            self.mock_config.latt_vectors,
            valid=True
        )
        mock_opt_model.assert_called_once()
        mock_analyze_result.assert_called_once_with(
            f'{self.mock_config.output_name}.{self.mock_config.output_format}',
            self.mock_config.target_sro,
            self.mock_config.element,
            self.mock_config.latt_type
        )
    
    @patch('pyhea.main.parse_args')
    @patch('pyhea.main.analyze_structure')
    def test_main_analyze_command(self, mock_analyze_structure, mock_parse_args):
        """Test the 'analyze' command functionality"""
        # Mock command line arguments
        mock_args = MagicMock()
        mock_args.command = 'analyze'
        mock_args.structure_file = self.test_structure_file
        mock_args.lattice_type = 'BCC'
        mock_args.elements = ['Fe', 'Ni']
        mock_args.output = 'test_output.png'
        mock_parse_args.return_value = mock_args
        
        # Set up mock returns
        mock_analyze_structure.return_value = (np.array([0.1, 0.2]), 'test_output.png')
        
        # Run main function
        main()
        
        # Verify mock calls
        mock_parse_args.assert_called_once()
        mock_analyze_structure.assert_called_once_with(
            mock_args.structure_file,
            latt_type=mock_args.lattice_type,
            element_types=mock_args.elements,
            output_file=mock_args.output
        )
    
    @patch('pyhea.main.parse_args')
    def test_main_invalid_command(self, mock_parse_args):
        """Test handling of invalid commands"""
        # Mock command line arguments with invalid command
        mock_args = MagicMock()
        mock_args.command = 'invalid_command'
        mock_parse_args.return_value = mock_args
        
        # Verify that ValueError is raised
        with self.assertRaises(ValueError) as context:
            main()
        
        self.assertIn('Unknown command', str(context.exception))
    
    @patch('pyhea.main.parse_args')
    @patch('pyhea.main.input_config')
    def test_main_run_with_invalid_config(self, mock_input_config, mock_parse_args):
        """Test handling of invalid configuration"""
        # Mock command line arguments
        mock_args = MagicMock()
        mock_args.command = 'run'
        mock_args.config_file = 'nonexistent_config.yaml'
        mock_parse_args.return_value = mock_args
        
        # Mock input_config to raise an exception
        mock_input_config.side_effect = FileNotFoundError("Config file not found")
        
        # Verify that the exception is raised
        with self.assertRaises(FileNotFoundError):
            main()

if __name__ == '__main__':
    unittest.main()
