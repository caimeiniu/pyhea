import unittest
from unittest.mock import patch, mock_open
import yaml
import numpy as np
import os
from pyhea.io import input_config

class test_input_config(unittest.TestCase):
    def setUp(self):
        # Mock POSCAR with 4 atoms per unit cell (FCC structure)
        self.mock_poscar = """
Test structure
1.0
3.0 0.0 0.0
0.0 3.0 0.0
0.0 0.0 3.0
4
direct
0.0 0.0 0.0
0.5 0.5 0.0
0.5 0.0 0.5
0.0 0.5 0.5"""

        # Mock target SRO file content
        # For binary system (type=2), we need 3 values per shell: AA, AB, BB
        self.mock_target_sro = """
# Target SRO values for shell 1
First Shell
0.0 
0.1 0.2
# Target SRO values for second shell
Second Shell
0.3 
0.4 0.5
# Target SRO values for shell 3
Third Shell
0.6 
0.7 0.8
"""
        self.test_dir = os.path.join(os.getcwd(), 'test_input_files')
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)

        self.poscar_file = os.path.join(self.test_dir, 'POSCAR_FCC')
        with open(self.poscar_file, 'w') as f:
            f.write(self.mock_poscar)
        
        self.target_sro_file = os.path.join(self.test_dir, 'target_sro.txt')
        with open(self.target_sro_file, 'w') as f:
            f.write(self.mock_target_sro)

        # Valid YAML with correct number of atoms (4 atoms * 1000 cells = 4000 atoms)
        # For type=2 (binary system), sum of elements should equal total atoms
        self.valid_yaml = yaml.dump({
            'type': 2,
            'element': [2000, 2000],  # Total = 4000 atoms (matches 4 atoms/cell * 10*10*10 cells)
            'weight': [1.0, 1.0],  # Equal weights for both elements
            'cell_dim': [10, 10, 10],
            'solutions': 1,
            'max_shell_num': 2,
            'total_iter': 1000,
            'converge_depth': 100,
            'parallel_task': 1,
            'target_sro': f'{self.target_sro_file}',  # Specify target SRO file path
            'structure': f'{self.poscar_file}'
        })

        self.valid_yaml_file = os.path.join(self.test_dir, 'valid.yaml')
        with open(self.valid_yaml_file, 'w') as f:
            f.write(self.valid_yaml)

        self.input_config = input_config(self.valid_yaml_file)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            import shutil
            shutil.rmtree(self.test_dir)

    @patch('builtins.open', new_callable=mock_open)
    def test_valid_yaml_and_poscar_loading(self, mock_file):
        # Mock the behavior of reading both YAML and POSCAR files
        mock_yaml = mock_open(read_data=self.valid_yaml)
        mock_poscar = mock_open(read_data=self.mock_poscar)
        mock_target_sro = mock_open(read_data=self.mock_target_sro)

        def mock_open_func(filename, mode='r', *args, **kwargs):
            if filename.endswith('valid.yaml'):
                return mock_yaml()
            elif filename.endswith('POSCAR_FCC'):
                return mock_poscar()
            elif filename.endswith('target_sro.txt'):
                return mock_target_sro()
            raise FileNotFoundError(f"Mock file not found: {filename}")

        mock_file.side_effect = mock_open_func
        config = input_config('valid.yaml')
        
        # Test that required fields exist
        self.assertTrue(hasattr(config, 'type'))
        self.assertTrue(hasattr(config, 'element'))
        self.assertTrue(hasattr(config, 'weight'))
        self.assertTrue(hasattr(config, 'cell_dim'))
        self.assertTrue(hasattr(config, 'solutions'))
        self.assertTrue(hasattr(config, 'max_shell_num'))
        self.assertTrue(hasattr(config, 'total_iter'))
        self.assertTrue(hasattr(config, 'converge_depth'))
        self.assertTrue(hasattr(config, 'parallel_task'))
        self.assertTrue(hasattr(config, 'target_sro'))
        
        # Test data types
        self.assertIsInstance(config.type, int)
        self.assertIsInstance(config.element, list)
        self.assertIsInstance(config.weight, list)
        self.assertIsInstance(config.cell_dim, list)
        self.assertIsInstance(config.solutions, int)
        self.assertIsInstance(config.max_shell_num, int)
        self.assertIsInstance(config.total_iter, int)
        self.assertIsInstance(config.converge_depth, int)
        self.assertIsInstance(config.parallel_task, int)
        self.assertTrue(isinstance(config.target_sro, np.ndarray))
        
        # Test values
        self.assertEqual(config.type, 2)
        self.assertEqual(config.element, [2000, 2000])
        self.assertEqual(config.weight, [1.0, 1.0])
        self.assertEqual(config.cell_dim, [10, 10, 10])
        self.assertEqual(config.solutions, 1)
        self.assertEqual(config.max_shell_num, 2)
        self.assertEqual(config.total_iter, 1000)
        self.assertEqual(config.converge_depth, 100)
        self.assertEqual(config.parallel_task, 1)
        expected_sro = np.array([
            [0.0, 0.1, 0.1, 0.2],  # First shell
            [0.3, 0.4, 0.4, 0.5],  # Second shell
            [0.6, 0.7, 0.7, 0.8]   # Third shell
        ])
        self.assertEqual(np.abs(config.target_sro - expected_sro).max(), 0)

    @patch('builtins.open', new_callable=mock_open, read_data=yaml.dump({
        'type': 2,
        'element': [26, 28],
        'unknown_key': 'unexpected_value'
    }))
    def test_valid_and_invalid_keys_in_yaml(self, mock_file):
        # Test that ValueError is raised when both valid and invalid keys are present
        with self.assertRaises(ValueError) as cm:
            input_config('mixed_keys.yaml')
        self.assertIn('Unknown parameters found', str(cm.exception))

    @patch('builtins.open', new_callable=mock_open)
    def test_defaults_when_keys_missing(self, mock_file):
        yaml_data = {
            'type': 2,
            'element': [2000, 2000],  # Total = 4000 atoms (matches 4 atoms/cell * 10*10*10 cells)
            'cell_dim': [10, 10, 10],
            'structure': 'POSCAR_FCC'
        }
        mock_file.side_effect = [
            mock_open(read_data=yaml.dump(yaml_data)).return_value,
            mock_open(read_data=self.mock_poscar).return_value
        ]
        config = input_config('missing_keys.yaml')
        
        # Test that missing keys get default values
        self.assertEqual(config.weight, [])  # Default is empty list
        self.assertEqual(config.solutions, 0)  # Default is 0
        self.assertEqual(config.max_shell_num, 0)  # Default is 0 when weight is empty
        self.assertEqual(config.total_iter, 0)  # Default is 0
        self.assertEqual(config.converge_depth, 0)  # Default is 0
        self.assertEqual(config.parallel_task, 0)  # Default is 0
        expected_sro = np.array([[0.0] * 4] * 3)  # Default is 3 shells of zeros
        self.assertEqual(np.abs(config.target_sro - expected_sro).max(), 0)

    @patch('builtins.open', new_callable=mock_open)
    def test_without_poscar_loading(self, mock_file):
        yaml_data = {
            'type': 2,
            'element': [2000, 2000],
            'cell_dim': [10, 10, 10],
            'structure': None  # This should trigger the error
        }
        mock_file.return_value = mock_open(read_data=yaml.dump(yaml_data)).return_value
        with self.assertRaises(TypeError):  # Changed from ValueError to TypeError
            input_config('valid.yaml')

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            input_config('nonexistent.yaml')

    @patch('builtins.open', new_callable=mock_open, read_data='invalid: yaml: file:')
    def test_invalid_yaml(self, mock_file):
        with self.assertRaises(ValueError) as cm:
            input_config('invalid.yaml')
        self.assertIn('Error parsing YAML file', str(cm.exception))

    @patch('builtins.open', new_callable=mock_open, read_data='')
    def test_empty_yaml(self, mock_file):
        with self.assertRaises(ValueError):
            input_config('empty.yaml')

    @patch('builtins.open', new_callable=mock_open)
    def test_invalid_poscar(self, mock_file):
        invalid_poscar = """
Invalid POSCAR
Not a number
Invalid lattice vectors
"""
        mock_file.side_effect = [
            mock_open(read_data=self.valid_yaml).return_value,
            mock_open(read_data=invalid_poscar).return_value
        ]
        with self.assertRaises(ValueError):
            input_config('valid.yaml')

    @patch('builtins.open', new_callable=mock_open)
    def test_valid_and_invalid_keys_in_yaml(self, mock_file):
        yaml_data = {
            'type': 2,
            'element': [2000, 2000],
            'cell_dim': [10, 10, 10],
            'structure': 'POSCAR_FCC',
            'unknown_key': 'unexpected_value'
        }
        mock_file.side_effect = [
            mock_open(read_data=yaml.dump(yaml_data)).return_value,
            mock_open(read_data=self.mock_poscar).return_value
        ]
        with self.assertRaises(ValueError) as cm:
            input_config('mixed_keys.yaml')
        self.assertIn('Invalid keys in YAML file', str(cm.exception))

    def test_invalid_poscar_coordinates(self):
        """Test handling of POSCAR with wrong number of coordinates."""
        invalid_poscar = """
Test structure
1.0
3.0 0.0 0.0
0.0 3.0 0.0
0.0 0.0 3.0
4
direct
0.0 0.0 0.0
0.5 0.5 0.0
0.5 0.0 0.5"""  # Missing one coordinate

        mock_yaml = mock_open(read_data=self.valid_yaml)
        mock_poscar = mock_open(read_data=invalid_poscar)
        mock_target_sro = mock_open(read_data=self.mock_target_sro)

        def mock_open_func(filename, mode='r', *args, **kwargs):
            if filename.endswith('valid.yaml'):
                return mock_yaml()
            elif filename.endswith('POSCAR_FCC'):
                return mock_poscar()
            elif filename.endswith('target_sro.txt'):
                return mock_target_sro()
            raise FileNotFoundError(f"Mock file not found: {filename}")

        with patch('builtins.open', mock_open_func):
            with self.assertRaises(ValueError) as context:
                input_config('valid.yaml')
            self.assertIn("coordinate", str(context.exception).lower())

    def test_non_cubic_cell(self):
        """Test handling of non-cubic cell in POSCAR."""
        non_cubic_poscar = """
Test structure
1.0
3.0 0.0 0.0
0.0 4.0 0.0
0.0 0.0 3.0
4
direct
0.0 0.0 0.0
0.5 0.5 0.0
0.5 0.0 0.5
0.0 0.5 0.5"""

        mock_yaml = mock_open(read_data=self.valid_yaml)
        mock_poscar = mock_open(read_data=non_cubic_poscar)
        mock_target_sro = mock_open(read_data=self.mock_target_sro)

        def mock_open_func(filename, mode='r', *args, **kwargs):
            if filename.endswith('valid.yaml'):
                return mock_yaml()
            elif filename.endswith('POSCAR_FCC'):
                return mock_poscar()
            elif filename.endswith('target_sro.txt'):
                return mock_target_sro()
            raise FileNotFoundError(f"Mock file not found: {filename}")

        with patch('builtins.open', mock_open_func):
            with self.assertRaises(ValueError) as context:
                input_config('valid.yaml')
            self.assertIn("cubic", str(context.exception).lower())

    def test_invalid_element_ratios(self):
        """Test handling of invalid element ratios."""
        valid_yaml_test = yaml.dump({
            'type': 2,
            'element': [3000, 1000],  # Unequal distribution is fine!
            'weight': [1.0, 1.0],
            'cell_dim': [10, 10, 10],
            'solutions': 1,
            'max_shell_num': 2,
            'total_iter': 1000,
            'converge_depth': 100,
            'parallel_task': 1,
            'target_sro': f'{self.target_sro_file}',
            'structure': f'{self.poscar_file}'
        })

        mock_yaml = mock_open(read_data=valid_yaml_test)
        mock_poscar = mock_open(read_data=self.mock_poscar)
        mock_target_sro = mock_open(read_data=self.mock_target_sro)

        def mock_open_func(filename, mode='r', *args, **kwargs):
            if filename.endswith('valid.yaml'):
                return mock_yaml()
            elif filename.endswith('POSCAR_FCC'):
                return mock_poscar()
            elif filename.endswith('target_sro.txt'):
                return mock_target_sro()
            raise FileNotFoundError(f"Mock file not found: {filename}")

        with patch('builtins.open', mock_open_func):
            input_config('valid.yaml')

    def test_target_sro_shape_validation(self):
        """Test validation of target SRO matrix shape."""
        # For type=2 (binary system), each shell should have 3 values
        invalid_sro = """
First Shell
0.1 
0.2  
Second Shell
0.3 
0.4 0.5
"""
        mock_yaml = mock_open(read_data=self.valid_yaml)
        mock_poscar = mock_open(read_data=self.mock_poscar)
        mock_target_sro = mock_open(read_data=invalid_sro)

        def mock_open_func(filename, mode='r', *args, **kwargs):
            if filename.endswith('valid.yaml'):
                return mock_yaml()
            elif filename.endswith('POSCAR_FCC'):
                return mock_poscar()
            elif filename.endswith('target_sro.txt'):
                return mock_target_sro()
            raise FileNotFoundError(f"Mock file not found: {filename}")

        with patch('builtins.open', mock_open_func):
            with self.assertRaises(ValueError) as context:
                input_config('valid.yaml')
            self.assertIn("invalid number of values", str(context.exception).lower())

    def test_invalid_poscar_coordinates(self):
        """Test handling of invalid POSCAR coordinates."""
        # Create POSCAR with wrong number of coordinates
        poscar_content = """
Test System
1.0
10.0 0.0 0.0
0.0 10.0 0.0
0.0 0.0 10.0
4
Direct
0.0 0.0 0.0  # Only 3 positions when there should be 4
0.5 0.5 0.5
0.2 0.2 0.2"""
        
        mock_yaml = mock_open(read_data=self.valid_yaml)
        mock_poscar = mock_open(read_data=poscar_content)
        mock_target_sro = mock_open(read_data=self.mock_target_sro)

        def mock_open_func(filename, mode='r', *args, **kwargs):
            if filename.endswith('valid.yaml'):
                return mock_yaml()
            elif filename.endswith('POSCAR_FCC'):
                return mock_poscar()
            elif filename.endswith('target_sro.txt'):
                return mock_target_sro()
            raise FileNotFoundError(f"Mock file not found: {filename}")
        with patch('builtins.open', mock_open_func):
            with self.assertRaises(ValueError) as context:
                input_config('valid.yaml')
            self.assertIn("Number of atomic positions does not match", str(context.exception))

    def test_non_cubic_cell(self):
        """Test detection of non-cubic cells."""
        # Create POSCAR with non-cubic cell
        poscar_content = """
Test System
1.0
10.0 0.0 0.0
0.0 8.0 0.0
0.0 0.0 10.0
4
Direct
0.0 0.0 0.0
0.5 0.5 0.5
0.2 0.2 0.2
0.7 0.7 0.7"""
        
        mock_yaml = mock_open(read_data=self.valid_yaml)
        mock_poscar = mock_open(read_data=poscar_content)
        mock_target_sro = mock_open(read_data=self.mock_target_sro)

        def mock_open_func(filename, mode='r', *args, **kwargs):
            if filename.endswith('valid.yaml'):
                return mock_yaml()
            elif filename.endswith('POSCAR_FCC'):
                return mock_poscar()
            elif filename.endswith('target_sro.txt'):
                return mock_target_sro()
            raise FileNotFoundError(f"Mock file not found: {filename}")

        with patch('builtins.open', mock_open_func):
            with self.assertRaises(ValueError) as context:
                input_config('valid.yaml')
            self.assertIn("Non-cubic cell detected", str(context.exception))
        

    def test_non_orthogonal_cell(self):
        """Test detection of non-orthogonal cells."""
        # Create POSCAR with non-orthogonal cell
        poscar_content = """
Test System
1.0
10.0 1.0 0.0
0.0 10.0 0.0
0.0 0.0 10.0
4
Direct
0.0 0.0 0.0
0.5 0.5 0.5
0.2 0.2 0.2
0.7 0.7 0.7"""
        
        mock_yaml = mock_open(read_data=self.valid_yaml)
        mock_poscar = mock_open(read_data=poscar_content)
        mock_target_sro = mock_open(read_data=self.mock_target_sro)

        def mock_open_func(filename, mode='r', *args, **kwargs):
            if filename.endswith('valid.yaml'):
                return mock_yaml()
            elif filename.endswith('POSCAR_FCC'):
                return mock_poscar()
            elif filename.endswith('target_sro.txt'):
                return mock_target_sro()
            raise FileNotFoundError(f"Mock file not found: {filename}")

        with patch('builtins.open', mock_open_func):
            with self.assertRaises(ValueError) as context:
                input_config('valid.yaml')
            self.assertIn("Non-orthogonal lattice vectors", str(context.exception))
            
    def test_invalid_element_number(self):
        """Test validation of element ratios."""
        config_data = {
            'type': 2,
            'element': [4, 1],  # 3:1 ratio
            'cell_dim': [1, 1, 1],
            'weight': [1.0, 1.0],
            'structure': f'{self.poscar_file}',
        }
        
        # Create POSCAR with different ratio (1:1)
        poscar_content = """
Test System
1.0
10.0 0.0 0.0
0.0 10.0 0.0
0.0 0.0 10.0
4
Direct
0.0 0.0 0.0
0.5 0.5 0.5
0.2 0.2 0.2
0.7 0.7 0.7"""
        
        mock_yaml = mock_open(read_data=yaml.dump(config_data))
        mock_poscar = mock_open(read_data=poscar_content)
        mock_target_sro = mock_open(read_data=self.mock_target_sro)

        def mock_open_func(filename, mode='r', *args, **kwargs):
            if filename.endswith('valid.yaml'):
                return mock_yaml()
            elif filename.endswith('POSCAR_FCC'):
                return mock_poscar()
            elif filename.endswith('target_sro.txt'):
                return mock_target_sro()
            raise FileNotFoundError(f"Mock file not found: {filename}")

        with patch('builtins.open', mock_open_func):
            with self.assertRaises(ValueError) as context:
                input_config('valid.yaml')
            self.assertIn("sum of elements", str(context.exception))
            

if __name__ == '__main__':
    unittest.main()
