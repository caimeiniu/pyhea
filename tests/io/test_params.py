import unittest
from unittest.mock import patch
from pyhea.io.params import parse_args

class TestParams(unittest.TestCase):
    def test_parse_args_run(self):
        # Test 'run' command with config file
        with patch('sys.argv', ['script.py', 'run', 'test_config.yaml']):
            args = parse_args()
            self.assertEqual(args.command, 'run')
            self.assertEqual(args.config_file, 'test_config.yaml')
    
    def test_parse_args_analyze(self):
        # Test 'analyze' command with input file
        with patch('sys.argv', ['script.py', 'analyze', 'test_input.dat']):
            args = parse_args()
            self.assertEqual(args.command, 'analyze')
            self.assertEqual(args.structure_file, 'test_input.dat')
    
    def test_parse_args_no_command(self):
        # Test with no command
        with patch('sys.argv', ['script.py']):
            with self.assertRaises(SystemExit):
                parse_args()
    
    def test_parse_args_invalid_command(self):
        # Test with invalid command
        with patch('sys.argv', ['script.py', 'invalid']):
            with self.assertRaises(SystemExit):
                parse_args()
    
    def test_parse_args_run_no_config(self):
        # Test 'run' command without config file
        with patch('sys.argv', ['script.py', 'run']):
            with self.assertRaises(SystemExit):
                parse_args()
    
    def test_parse_args_analyze_no_data(self):
        # Test 'analyze' command without data file
        with patch('sys.argv', ['script.py', 'analyze']):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_analyze_with_optional_args(self):
        """Test analyze command with all optional arguments."""
        with patch('sys.argv', [
            'script.py', 'analyze', 'test_input.dat',
            '--format', 'poscar',
            '--lattice-type', 'BCC',
            '--elements', 'Fe', 'Ni', 'Cr',
            '-o', 'output.png'
        ]):
            args = parse_args()
            self.assertEqual(args.command, 'analyze')
            self.assertEqual(args.structure_file, 'test_input.dat')
            self.assertEqual(args.format, 'poscar')
            self.assertEqual(args.lattice_type, 'BCC')
            self.assertEqual(args.elements, ['Fe', 'Ni', 'Cr'])
            self.assertEqual(args.output, 'output.png')

    def test_analyze_invalid_format(self):
        """Test analyze command with invalid format."""
        with patch('sys.argv', [
            'script.py', 'analyze', 'test_input.dat',
            '--format', 'invalid'
        ]):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_analyze_invalid_lattice(self):
        """Test analyze command with invalid lattice type."""
        with patch('sys.argv', [
            'script.py', 'analyze', 'test_input.dat',
            '--lattice-type', 'invalid'
        ]):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_help_command(self):
        """Test help command."""
        for help_arg in ['-h', '--help']:
            with patch('sys.argv', ['script.py', help_arg]):
                with self.assertRaises(SystemExit):
                    parse_args()

    def test_version(self):
        """Test version command."""
        with patch('sys.argv', ['script.py', '--version']):
            with self.assertRaises(SystemExit):
                parse_args()

if __name__ == '__main__':
    unittest.main()
