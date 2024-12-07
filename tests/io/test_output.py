import unittest
import os
import tempfile
import shutil
import numpy as np
from pyhea.io import output
from pyhea.version import __version__

class MockLattice:
    def __init__(self):
        self.latt_con = 3.5
        self.latt_vec = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
        self.coords = [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5]
        ]

class TestOutput(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_data = {
            'sro': np.array([0.1, 0.2]),
            'energy': -1.5,
            'temperature': 300.0
        }
        self.mock_latt = MockLattice()
        self.nest = [0, 0, 1, 1]  # Two atoms of type A, two of type B
        self.ntyp = 2
        self.elem = ['Fe', 'Ni']  # Will be replaced with A, B in output
    
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_write_poscar(self):
        poscar_file = os.path.join(self.test_dir, 'POSCAR')
        output.write_poscar(self.nest, self.mock_latt, self.ntyp, self.elem, poscar_file)
        
        self.assertTrue(os.path.exists(poscar_file))
        with open(poscar_file, 'r') as f:
            lines = f.readlines()
            # Check header
            self.assertEqual(lines[0].strip(), f"PyHEA v{__version__}")
            # Check lattice constant
            self.assertEqual(float(lines[1].strip()), 3.5)
            # Check lattice vectors
            self.assertEqual(lines[2].strip(), "1.0\t0.0\t0.0")
            self.assertEqual(lines[3].strip(), "0.0\t1.0\t0.0")
            self.assertEqual(lines[4].strip(), "0.0\t0.0\t1.0")
            # Check element labels
            self.assertEqual(lines[5].strip(), "A   B")
            # Check atom counts
            self.assertEqual(lines[6].strip(), "2   2")
            # Check coordinate type
            self.assertEqual(lines[7].strip(), "Cartesian")
            # Check number of coordinate lines
            self.assertEqual(len(lines[8:]), 4)
    
    def test_write_structure_poscar(self):
        output_file = os.path.join(self.test_dir, 'structure.poscar')
        output.write_structure(self.nest, self.mock_latt, self.ntyp, self.elem, 
                             output_file, output_format='vasp/poscar')
        self.assertTrue(os.path.exists(output_file))
    
    def test_write_structure_lammps(self):
        output_file = os.path.join(self.test_dir, 'structure.lmp')
        output.write_structure(self.nest, self.mock_latt, self.ntyp, self.elem, 
                             output_file, output_format='lammps/lmp')
        self.assertTrue(os.path.exists(output_file))
    
    def test_write_structure_invalid_format(self):
        output_file = os.path.join(self.test_dir, 'structure.xyz')
        with self.assertRaises(ValueError) as context:
            output.write_structure(self.nest, self.mock_latt, self.ntyp, self.elem, 
                                 output_file, output_format='xyz')
        self.assertIn("Unsupported output format", str(context.exception))
    
    def test_save_output(self):
        output_file = os.path.join(self.test_dir, 'test_results.dat')
        output.save_output(self.test_data, output_file)
        self.assertTrue(os.path.exists(output_file))
        
        with open(output_file, 'r') as f:
            content = f.read()
            self.assertIn('sro: 0.1 0.2', content)
            self.assertIn('energy: -1.5', content)
            self.assertIn('temperature: 300.0', content)
    
    def test_save_invalid_output(self):
        output_file = os.path.join(self.test_dir, 'test_invalid.dat')
        with self.assertRaises(ValueError):
            output.save_output(None, output_file)
    
    def test_save_output_invalid_path(self):
        output_file = os.path.join('/nonexistent/directory', 'test.dat')
        with self.assertRaises(OSError):
            output.save_output(self.test_data, output_file)
    
    def test_save_output_empty_data(self):
        output_file = os.path.join(self.test_dir, 'test_empty.dat')
        empty_data = {}
        with self.assertRaises(ValueError):
            output.save_output(empty_data, output_file)

if __name__ == '__main__':
    unittest.main()
