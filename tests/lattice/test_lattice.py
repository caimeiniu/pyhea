import unittest
import numpy as np
from pyhea.lattice import lattice

class TestLattice(unittest.TestCase):
    def setUp(self):
        """Set up test parameters for both BCC and FCC lattices"""
        self.cell_dim = [4]  # 4x4x4 supercell
        self.latt_con = 1.0  # Unit lattice constant
        self.latt_vec = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]], dtype=np.float16)
    
    def test_bcc_initialization(self):
        """Test BCC lattice initialization and basic properties"""
        bcc = lattice(self.cell_dim, 'BCC', self.latt_con, self.latt_vec)
        
        # Check lattice type
        self.assertEqual(bcc.latt_typ, 'BCC')
        
        # Check number of atoms (2 atoms per unit cell * 4^3 cells)
        expected_atoms = 2 * 4**3
        self.assertEqual(len(bcc.coords), expected_atoms)
        
        # Check lattice vectors
        self.assertTrue(np.allclose(bcc.latt_vec, self.latt_vec, rtol=1e-3))

        # Check that coordinates are within bounds
        self.assertTrue(np.all(bcc.coords >= 0))
        self.assertTrue(np.all(bcc.coords < 4))
    
    def test_fcc_initialization(self):
        """Test FCC lattice initialization and basic properties"""
        fcc = lattice(self.cell_dim, 'FCC', self.latt_con, self.latt_vec)
        
        # Check lattice type
        self.assertEqual(fcc.latt_typ, 'FCC')
        
        # Check number of atoms (4 atoms per unit cell * 4^3 cells)
        expected_atoms = 4 * 4**3
        self.assertEqual(len(fcc.coords), expected_atoms)

        # Check that coordinates are within bounds
        self.assertTrue(np.all(fcc.coords >= 0))
        self.assertTrue(np.all(fcc.coords < 4))
    
    def test_invalid_lattice_type(self):
        """Test that invalid lattice types raise ValueError"""
        with self.assertRaises(ValueError):
            lattice(self.cell_dim, 'INVALID', self.latt_con, self.latt_vec)

        # Test empty string
        with self.assertRaises(ValueError):
            lattice(self.cell_dim, '', self.latt_con, self.latt_vec)

        # Test None
        with self.assertRaises(ValueError):
            lattice(self.cell_dim, None, self.latt_con, self.latt_vec)
    
    def test_periodic_boundary_conditions(self):
        """Test periodic boundary conditions for both lattice types"""
        bcc = lattice(self.cell_dim, 'BCC', self.latt_con, self.latt_vec)
        
        test_cases = [
            # Test coordinates outside the box
            (np.array([4.5, -0.5, 8.0], dtype=np.float16), np.array([0.5, 3.5, 0.0], dtype=np.float16)),
            # Test coordinates at boundaries
            (np.array([4.0, 0.0, 4.0], dtype=np.float16), np.array([0.0, 0.0, 0.0], dtype=np.float16)),
            # Test negative coordinates
            (np.array([-1.0, -2.0, -3.0], dtype=np.float16), np.array([3.0, 2.0, 1.0], dtype=np.float16))
        ]
        
        for input_coord, expected_coord in test_cases:
            pbc_coord = bcc.calc_pbc(input_coord)
            self.assertTrue(np.allclose(pbc_coord, expected_coord, rtol=1e-3))
            # Verify coordinates are within bounds
            self.assertTrue(np.all(pbc_coord >= 0))
            self.assertTrue(np.all(pbc_coord < 4))
    
    def test_neighbor_list_bcc(self):
        """Test BCC neighbor list properties"""
        bcc = lattice(self.cell_dim, 'BCC', self.latt_con, self.latt_vec)
        
        # Check neighbor list structure
        self.assertEqual(len(bcc.nbor_list), 2 * 4**3)  # Total atoms
        
        # Check number of neighbors in each shell
        for atom_neighbors in bcc.nbor_list:
            self.assertEqual(len(atom_neighbors[0]), 8)  # First shell
            self.assertEqual(len(atom_neighbors[1]), 6)  # Second shell
            self.assertEqual(len(atom_neighbors[2]), 12)  # Third shell

            # Check that neighbor indices are valid
            for shell in atom_neighbors:
                self.assertTrue(np.all(shell >= 0))
                self.assertTrue(np.all(shell < 2 * 4**3))
                # Check for duplicates
                self.assertEqual(len(shell), len(np.unique(shell)))
    
    def test_neighbor_list_fcc(self):
        """Test FCC neighbor list properties"""
        fcc = lattice(self.cell_dim, 'FCC', self.latt_con, self.latt_vec)
        
        # Check neighbor list structure
        self.assertEqual(len(fcc.nbor_list), 4 * 4**3)  # Total atoms
        
        # Check number of neighbors in each shell
        for atom_neighbors in fcc.nbor_list:
            self.assertEqual(len(atom_neighbors[0]), 12)  # First shell
            self.assertEqual(len(atom_neighbors[1]), 6)   # Second shell
            self.assertEqual(len(atom_neighbors[2]), 12)  # Third shell

            # Check that neighbor indices are valid
            for shell in atom_neighbors:
                self.assertTrue(np.all(shell >= 0))
                self.assertTrue(np.all(shell < 4 * 4**3))
                # Check for duplicates
                self.assertEqual(len(shell), len(np.unique(shell)))
    
    def test_neighbor_validation(self):
        """Test neighbor list validation for both lattice types"""
        # Test BCC validation
        bcc = lattice(self.cell_dim, 'BCC', self.latt_con, self.latt_vec, valid=True)
        bcc.valid_nbors(bcc.coords, bcc.cell_dim, 'BCC', bcc.nbor_list)
        
        # Test FCC validation
        fcc = lattice(self.cell_dim, 'FCC', self.latt_con, self.latt_vec, valid=True)
        fcc.valid_nbors(fcc.coords, fcc.cell_dim, 'FCC', fcc.nbor_list)
    
    def test_distance_calculations(self):
        """Test distance calculations between atoms"""
        bcc = lattice(self.cell_dim, 'BCC', self.latt_con, self.latt_vec)
        
        test_cases = [
            # First shell distance (√3/2)
            (np.array([0.0, 0.0, 0.0], dtype=np.float16),
             np.array([0.5, 0.5, 0.5], dtype=np.float16),
             np.sqrt(3.0) / 2.0),
            # Second shell distance (1.0)
            (np.array([0.0, 0.0, 0.0], dtype=np.float16),
             np.array([1.0, 0.0, 0.0], dtype=np.float16),
             1.0),
            # Third shell distance (√2)
            (np.array([0.0, 0.0, 0.0], dtype=np.float16),
             np.array([1.0, 1.0, 0.0], dtype=np.float16),
             np.sqrt(2.0))
        ]
        
        for point1, point2, expected_dist in test_cases:
            calculated_dist = bcc.calc_pbc_dist(point1, point2)
            # Use relative tolerance for float16 comparison
            self.assertTrue(abs(calculated_dist - expected_dist) / expected_dist < 1e-2)

    def test_edge_cases(self):
        """Test edge cases and potential error conditions"""
        # Test with minimum cell dimension
        min_cell = [1]
        min_vec = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float16)
        
        bcc_min = lattice(min_cell, 'BCC', self.latt_con, min_vec)
        self.assertEqual(len(bcc_min.coords), 2)  # 2 atoms for 1x1x1 BCC
        
        fcc_min = lattice(min_cell, 'FCC', self.latt_con, min_vec)
        self.assertEqual(len(fcc_min.coords), 4)  # 4 atoms for 1x1x1 FCC

if __name__ == '__main__':
    unittest.main()
