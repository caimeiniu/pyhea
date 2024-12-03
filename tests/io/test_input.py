import unittest
from unittest.mock import patch, mock_open
from pyhea.io.input import input_config
import yaml

class test_input_config(unittest.TestCase):
    def setUp(self):
        # Mock configuration data
        self.mock_config = {
            'type': 3,
            'element': [18, 18, 18],
            'cell_dim': [3, 3, 3],
            'nests': 24,
            'mc_step': 750,
            'total_iter': 10,
            'global_iter': 1000,
            'max_shell_num': 2,
            'weight': [2.0, 1.0, 1.0],
            'structure': 'POSCAR'
        }
        self.valid_yaml = yaml.dump(self.mock_config)
        
        self.non_existent_file_yaml = yaml.dump({})

        # Mock POSCAR data
        self.mock_poscar = """
            BCC structure
            1.0
            1.0 0.0 0.0
            0.0 1.0 0.0
            0.0 0.0 1.0
            2
            direct
            0.0 0.0 0.0
            0.5 0.5 0.5
        """
        self.invalid_mock_poscar = """
            BCC structure
            1.0
            1.0 0.0 0.0
            0.0 1.0 0.0
            0.0 0.0 1.0
            2
            direct
            0.0 0.0 0.0
        """

    @patch('builtins.open', new_callable=mock_open)
    def test_file_not_found(self, mock_file):
        mock_file.side_effect = [
            mock_open(read_data="").return_value,
            mock_open(read_data=self.mock_poscar).return_value
        ]
        # Test that ValueError is raised when file does not exist
        with self.assertRaises(ValueError):
            input_config('non_existent_file.yaml')

    @patch('builtins.open', new_callable=mock_open)
    def test_valid_yaml_and_poscar_loading(self, mock_file):
        # Mock the behavior of reading both YAML and POSCAR files
        mock_file.side_effect = [
            mock_open(read_data=self.valid_yaml).return_value,
            mock_open(read_data=self.mock_poscar).return_value
        ]
        config = input_config('valid.yaml')
        
        # Check YAML attributes
        self.assertEqual(config.element, [18, 18, 18])
        self.assertEqual(config.cell_dim, [3, 3, 3])
        self.assertEqual(config.nests, 24)
        self.assertEqual(config.mc_step, 750)
        self.assertEqual(config.total_iter, 10)
        self.assertEqual(config.global_iter, 1000)
        self.assertEqual(config.max_shell_num, 2)
        self.assertEqual(config.weight, [2.0, 1.0, 1.0])

        # Check POSCAR attributes
        self.assertEqual(config.structure['comment'], 'BCC structure')
        self.assertEqual(config.structure['scaling_factor'], 1.0)
        self.assertEqual(config.structure['lattice_vectors'], [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        self.assertEqual(config.structure['num_atoms'], 2)
        self.assertEqual(config.structure['coordinate_type'], 'direct')
        self.assertEqual(config.structure['positions'], [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5]
        ])

    @patch('builtins.open', new_callable=mock_open, read_data="invalid: yaml: content")
    def test_invalid_yaml(self, mock_file):
        # Test that ValueError is raised when YAML content is invalid
        with self.assertRaises(ValueError):
            input_config('invalid_yaml.yaml')

    @patch('builtins.open', new_callable=mock_open, read_data="")
    def test_empty_yaml(self, mock_file):
        # Test that ValueError is raised when YAML file is empty
        with self.assertRaises(ValueError):
            input_config('empty.yaml')

    @patch('builtins.open', new_callable=mock_open)
    def test_unknown_key_in_config(self, mock_file):
        mock_file.side_effect = [
            mock_open(read_data=yaml.dump({"unknown: value"})).return_value,
            mock_open(read_data=self.mock_poscar).return_value
        ]
        # Test that ValueError is raised when YAML file contains unknown parameters
        with self.assertRaises(ValueError):
            input_config('unknown_key.yaml')

    @patch('builtins.open', new_callable=mock_open, read_data=yaml.dump({
        'element': ['Fe'],
        'unknown_key': 'unexpected_value'
    }))
    def test_valid_and_invalid_keys_in_yaml(self, mock_file):
        # Test that ValueError is raised when both valid and invalid keys are present
        with self.assertRaises(ValueError):
            input_config('mixed_keys.yaml')

    @patch('builtins.open', new_callable=mock_open, read_data=yaml.dump({
        'type': 3,
        'element': [18, 18, 18],
        'cell_dim': [3, 3, 3],
        'nests': 24,
        'mc_step': 750,
        'total_iter': 10,
        'global_iter': 1000,
        'max_shell_num': 2,
        'weight': [2.0, 1.0, 1.0]
    }))
    def test_without_poscar_loading(self, mock_file):
        # Test that a valid YAML file loads correctly
        with self.assertRaises(ValueError):
            config = input_config('valid.yaml')

    @patch('builtins.open', new_callable=mock_open)
    def test_defaults_when_keys_missing(self, mock_file):
        mock_file.side_effect = [
            mock_open(read_data=yaml.dump({})).return_value,
            mock_open(read_data=self.mock_poscar).return_value
        ]
        # Test that default values are returned when keys are missing in YAML
        config = input_config('missing_keys.yaml')
        self.assertEqual(config.element, [])
        self.assertEqual(config.cell_dim, [])
        self.assertEqual(config.nests, 0)
        self.assertEqual(config.mc_step, 0)
        self.assertEqual(config.total_iter, 0)
        self.assertEqual(config.global_iter, 0)
        self.assertEqual(config.max_shell_num, 0)
        self.assertEqual(config.weight, [])
        
    @patch('builtins.open', new_callable=mock_open)
    def test_invalid_poscar(self, mock_file):
        mock_file.side_effect = [
            mock_open(read_data=self.valid_yaml).return_value,
            mock_open(read_data=self.invalid_mock_poscar).return_value
        ]
        # Test that a valid YAML file loads correctly
        with self.assertRaises(IndexError):
            config = input_config('valid.yaml')
