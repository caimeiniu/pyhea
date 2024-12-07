import unittest
import os
import tempfile
import yaml
import numpy as np
from pyhea.io import input_config

class TestConfig(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create a test POSCAR file
        self.poscar_content = """
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
        self.poscar_file = os.path.join(self.test_dir, 'POSCAR_FCC')
        with open(self.poscar_file, 'w') as f:
            f.write(self.poscar_content)
        
        # Create test SRO file with lower triangular format values for 2 atom types
        # For each shell: [AA, AB, BB] (lower triangular)
        self.sro_content = """
First Shell
0.1
0.2 0.3"""
        self.sro_file = os.path.join(self.test_dir, 'test_sro.txt')
        with open(self.sro_file, 'w') as f:
            f.write(self.sro_content)
        
        # Create a test config file with target SRO file path
        self.config_data = {
            'type': 2,  # Number of element types
            'element': [2000, 2000],  # Equal numbers of each element type for 4000 total atoms
            'weight': [1.0, 0.5],
            'cell_dim': [10, 10, 10],
            'solutions': 1,
            'max_shell_num': 2,
            'total_iter': 1000,
            'converge_depth': 100,
            'parallel_task': 1,
            'target_sro': self.sro_file,  # Path to SRO file
            'structure': self.poscar_file
        }
        self.config_file = os.path.join(self.test_dir, 'test_config.yaml')
        with open(self.config_file, 'w') as f:
            yaml.dump(self.config_data, f)
    
    def tearDown(self):
        # Clean up temporary files
        if os.path.exists(self.test_dir):
            import shutil
            shutil.rmtree(self.test_dir)
    
    def test_config_loading(self):
        config = input_config(self.config_file)
        
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
        self.assertEqual(config.element, [2000, 2000])  # Equal numbers of each element type for 4000 total atoms
        self.assertEqual(config.weight, [1.0, 0.5])
        self.assertEqual(config.cell_dim, [10, 10, 10])
        self.assertEqual(config.solutions, 1)
        self.assertEqual(config.max_shell_num, 2)
        self.assertEqual(config.total_iter, 1000)
        self.assertEqual(config.converge_depth, 100)
        self.assertEqual(config.parallel_task, 1)
        
        # For 2 atom types, target_sro returns a 3x4 array with full matrix format [AA, AB, BA, BB]
        # where BA = AB due to symmetry. Missing values are filled with zeros.
        expected_sro = np.array([
            [0.1, 0.2, 0.2, 0.3],  # 1st shell [AA, AB, BA, BB]
            [0.0, 0.0, 0.0, 0.0],  # 2nd shell [AA, AB, BA, BB]
            [0.0, 0.0, 0.0, 0.0]   # 3rd shell [AA, AB, BA, BB]
        ])
        self.assertEqual(np.abs(config.target_sro - expected_sro).max(), 0)
    
    def test_invalid_config_path(self):
        with self.assertRaises(FileNotFoundError):
            input_config('nonexistent_config.yaml')
    
    def test_invalid_element_count(self):
        # Test when number of elements doesn't match type
        invalid_config = self.config_data.copy()
        invalid_config['type'] = 3  # Changed type but kept same number of elements
        invalid_config_file = os.path.join(self.test_dir, 'invalid_config.yaml')
        with open(invalid_config_file, 'w') as f:
            yaml.dump(invalid_config, f)
        
        with self.assertRaises(ValueError):
            input_config(invalid_config_file)
    
    def test_non_cubic_cell(self):
        # Test when cell dimensions are not cubic
        invalid_config = self.config_data.copy()
        invalid_config['cell_dim'] = [10, 11, 10]  # Non-cubic dimensions
        invalid_config_file = os.path.join(self.test_dir, 'non_cubic_config.yaml')
        with open(invalid_config_file, 'w') as f:
            yaml.dump(invalid_config, f)
        
        with self.assertRaises(ValueError):
            input_config(invalid_config_file)

    def test_non_existent_sro_file(self):
        """Test handling of non-existent SRO file."""
        config_data = self.config_data.copy()
        config_data['target_sro'] = 'nonexistent.txt'
        config_file = os.path.join(self.test_dir, 'test_missing_sro.yaml')
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        with self.assertRaises(FileNotFoundError):
            input_config(config_file)

    def test_wrong_sro_comments(self):
        """Test handling of malformed SRO file."""
        malformed_sro = """Invalid Format
        0.1 0.2 0.3 0.4  # Too many values for binary system
        0.4 0.5"""
        sro_file = os.path.join(self.test_dir, 'malformed_sro.txt')
        with open(sro_file, 'w') as f:
            f.write(malformed_sro)
        
        config_data = self.config_data.copy()
        config_data['target_sro'] = sro_file
        config_file = os.path.join(self.test_dir, 'test_malformed_sro.yaml')
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        with self.assertRaises(ValueError) as context:
            input_config(config_file)
        self.assertIn("Invalid SRO file format: could not convert string to float: '#'", str(context.exception))

    def test_malformed_sro_file(self):
        """Test handling of malformed SRO file."""
        malformed_sro = """Invalid Format
        0.1 0.2 0.3 0.4
        0.4 0.5"""
        sro_file = os.path.join(self.test_dir, 'malformed_sro.txt')
        with open(sro_file, 'w') as f:
            f.write(malformed_sro)
        
        config_data = self.config_data.copy()
        config_data['target_sro'] = sro_file
        config_file = os.path.join(self.test_dir, 'test_malformed_sro.yaml')
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        with self.assertRaises(ValueError) as context:
            input_config(config_file)
        self.assertIn("Too many values for shell", str(context.exception))

    def test_invalid_element_counts(self):
        """Test handling of invalid element counts."""
        config_data = self.config_data.copy()
        config_data['element'] = [2000, 1000]  # Should sum to 4000 for 10x10x10 cell
        config_file = os.path.join(self.test_dir, 'test_invalid_elements.yaml')
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        with self.assertRaises(ValueError) as context:
            input_config(config_file)
        self.assertIn("sum of elements", str(context.exception))

    def test_invalid_weights(self):
        """Test handling of invalid weight values."""
        config_data = self.config_data.copy()
        config_data['weight'] = [1.0, -0.5]  # Negative weight is invalid
        config_file = os.path.join(self.test_dir, 'test_invalid_weights.yaml')
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        with self.assertRaises(ValueError) as context:
            input_config(config_file)
        self.assertIn("weights must be positive", str(context.exception))

    def test_invalid_cell_dimensions(self):
        """Test handling of invalid cell dimensions."""
        test_cases = [
            ([0, 10, 10], "Cell dimensions must be positive integers"),
            ([-1, 10, 10], "Cell dimensions must be positive integers"),
            ([10.5, 10, 10], "Cell dimensions must be positive integers"),
            ([10, 8, 10], "Only cubic cell dimensions are supported")
        ]
        
        for dims, error_msg in test_cases:
            config_data = self.config_data.copy()
            config_data['cell_dim'] = dims
            config_file = os.path.join(self.test_dir, f'test_invalid_cell_{dims[0]}.yaml')
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f)
            with self.assertRaises(ValueError) as context:
                input_config(config_file)
            self.assertIn(error_msg, str(context.exception))

if __name__ == '__main__':
    unittest.main()
