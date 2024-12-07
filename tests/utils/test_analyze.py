import unittest
import os
import numpy as np
from unittest.mock import patch, MagicMock
from pyhea.utils.analyze import analyze_structure, analyze_result, calculate_sro, plot_sro_heatmap

class TestAnalyze(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.test_data_dir = os.path.join(self.test_dir, 'test_data')
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # Create test LAMMPS data file with minimal valid structure
        self.lmp_file = os.path.join(self.test_data_dir, 'test.lmp')
        with open(self.lmp_file, 'w') as f:
            f.write("""# LAMMPS data file
2 atoms
2 atom types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Atoms

1 1 0.0 0.0 0.0
2 2 5.0 5.0 5.0
""")
        
        # Create test POSCAR file with minimal valid structure
        self.poscar_file = os.path.join(self.test_data_dir, 'test.poscar')
        os.makedirs(os.path.dirname(self.poscar_file), exist_ok=True)
        with open(self.poscar_file, 'w') as f:
            f.write("""Fe Ni alloy
1.0
10.0 0.0 0.0
0.0 10.0 0.0
0.0 0.0 10.0
Fe Ni
1 1
Direct
0.0 0.0 0.0
0.5 0.5 0.5
""")
        
        # Test parameters
        self.element_types = ['Fe', 'Ni']
        self.target_sro = np.array([[[0.1, -0.2], [-0.2, 0.3]]])
        self.mock_sro = np.array([[[0.15, -0.25], [-0.25, 0.35]]])
    
    def tearDown(self):
        """Clean up test files"""
        if os.path.exists(self.test_data_dir):
            for root, dirs, files in os.walk(self.test_data_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(self.test_data_dir)
    
    @patch('pyhea.utils.analyze.import_file')
    @patch('pyhea.utils.analyze.wc.WarrenCowleyParameters')
    def test_calculate_sro_lammps(self, mock_wc, mock_import):
        """Test SRO calculation from LAMMPS file"""
        # Mock OVITO pipeline
        mock_pipeline = MagicMock()
        mock_import.return_value = mock_pipeline
        
        # Mock Warren-Cowley calculation
        mock_wc_instance = MagicMock()
        mock_wc.return_value = mock_wc_instance
        
        # Mock computed data
        mock_data = MagicMock()
        mock_data.attributes = {
            "Warren-Cowley parameters": self.mock_sro
        }
        mock_pipeline.compute.return_value = mock_data
        
        # Test FCC calculation
        sro_values = calculate_sro(self.lmp_file, 'FCC')
        self.assertTrue(np.array_equal(sro_values, self.mock_sro))
        mock_wc.assert_called_with(nneigh=[0, 12, 18], only_selected=False)
        
        # Test BCC calculation
        calculate_sro(self.lmp_file, 'BCC')
        mock_wc.assert_called_with(nneigh=[0, 8, 14], only_selected=False)
    
    @patch('pyhea.utils.analyze.import_file')
    @patch('pyhea.utils.analyze.dpdata')
    def test_calculate_sro_poscar(self, mock_dpdata, mock_import):
        """Test SRO calculation from POSCAR file"""
        # Mock VASP to LAMMPS conversion
        mock_system = MagicMock()
        mock_dpdata.System.return_value = mock_system
        
        # Mock OVITO pipeline
        mock_pipeline = MagicMock()
        mock_import.return_value = mock_pipeline
        mock_data = MagicMock()
        mock_data.attributes = {"Warren-Cowley parameters": self.mock_sro}
        mock_pipeline.compute.return_value = mock_data
        # Test POSCAR file handling
        sro_values = calculate_sro(self.poscar_file, 'FCC')

        self.assertTrue(np.array_equal(sro_values, self.mock_sro))
    
    @patch('pyhea.utils.analyze.calculate_sro')
    @patch('pyhea.utils.analyze.plot_sro_heatmap')
    def test_analyze_structure(self, mock_plot, mock_calc_sro):
        """Test structure analysis"""
        # Mock SRO calculation
        mock_calc_sro.return_value = self.mock_sro
        
        # Test with default parameters
        sro_values, output_file = analyze_structure(self.lmp_file)
        self.assertTrue(np.array_equal(sro_values, self.mock_sro[0]))
        self.assertEqual(output_file, 'heatmap.png')
        mock_plot.assert_called_once()
        
        # Test with custom parameters
        output_file = os.path.join(self.test_data_dir, 'custom.png')
        sro_values, out_file = analyze_structure(
            self.lmp_file,
            latt_type='BCC',
            element_types=self.element_types,
            output_file=output_file
        )
        self.assertEqual(out_file, output_file)
    
    @patch('pyhea.utils.analyze.calculate_sro')
    @patch('pyhea.utils.analyze.plot_sro_heatmap')
    def test_analyze_result(self, mock_plot, mock_calc_sro):
        """Test result analysis"""
        # Mock SRO calculation
        mock_calc_sro.return_value = self.mock_sro
        
        # Test analysis
        result_sro, mae, rmse = analyze_result(
            self.lmp_file,
            self.target_sro,
            self.element_types,
            'FCC'
        )
        
        # Verify results
        self.assertTrue(np.array_equal(result_sro, self.mock_sro))
        self.assertIsInstance(mae, float)
        self.assertIsInstance(rmse, float)
        
        # Verify error metrics
        expected_diff = self.mock_sro - self.target_sro
        expected_mae = np.mean(np.abs(expected_diff))
        expected_rmse = np.sqrt(np.mean(expected_diff**2))
        self.assertAlmostEqual(mae, expected_mae)
        self.assertAlmostEqual(rmse, expected_rmse)
    
    def test_plot_sro_heatmap(self):
        """Test SRO heatmap plotting"""
        sro_values = np.array([[0.1, -0.2], [-0.2, 0.3]])
        atom_labels = ['A', 'B']
        output_file = os.path.join(self.test_data_dir, 'test_heatmap.png')
        
        with patch('pyhea.utils.analyze.plt') as mock_plt:
            plot_sro_heatmap(sro_values, atom_labels, output_file)
            mock_plt.figure.assert_called_once()
            mock_plt.savefig.assert_called_once_with(output_file, bbox_inches='tight', dpi=300)
    
    @patch('pyhea.utils.analyze.calculate_sro')
    def test_invalid_lattice_type(self, mock_calc_sro):
        """Test handling of invalid lattice type"""
        mock_calc_sro.side_effect = ValueError("Invalid lattice type")
        with self.assertRaises(ValueError):
            analyze_structure(self.lmp_file, latt_type='INVALID')
    
    @patch('pyhea.utils.analyze.calculate_sro')
    def test_invalid_target_sro(self, mock_calc_sro):
        """Test handling of invalid target SRO values"""
        invalid_target = np.array([[[1.5, 0.2], [0.2, 0.3]]])  # SRO > 1
        
        def mock_validate_sro(*args, **kwargs):
            if np.any(np.abs(invalid_target) > 1):
                raise ValueError("SRO values must be between -1 and 1")
            return self.mock_sro
            
        mock_calc_sro.side_effect = mock_validate_sro
        with self.assertRaises(ValueError):
            analyze_result(self.lmp_file, invalid_target, self.element_types, 'FCC')

if __name__ == '__main__':
    unittest.main()
