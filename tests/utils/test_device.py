"""Unit tests for device utilities."""

import unittest
from unittest.mock import patch, MagicMock
import psutil
import subprocess
from pyhea.utils.device import (
    get_cpu_info,
    get_gpu_info,
    check_device_info,
    check_device_availability,
)

class TestDeviceUtils(unittest.TestCase):
    """Test device utility functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_cpu_info = {
            'processor': 'Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz',
            'cores': 8,
            'threads': 16,
            'frequency': '3600.00 MHz',
            'memory': '32.0 GB'
        }
        
        self.mock_gpu_info = [{
            'index': '0',
            'name': 'NVIDIA GeForce RTX 4090',
            'memory': '10.0 GB',
            'driver': '535.129.03'
        }]

    @patch('platform.system')
    @patch('psutil.cpu_count')
    @patch('psutil.cpu_freq')
    @patch('psutil.virtual_memory')
    @patch('platform.processor')
    def test_get_cpu_info(self, mock_processor, mock_vmem, mock_freq, mock_cpu_count, mock_system):
        """Test getting CPU information."""
        # Mock system and CPU info
        mock_system.return_value = "Linux"
        mock_processor.return_value = self.mock_cpu_info['processor']
        mock_cpu_count.side_effect = [8, 16]  # physical, logical
        mock_freq.return_value = MagicMock(current=3600.0)
        mock_vmem.return_value = MagicMock(total=32 * 1024**3)  # 32GB

        # Mock /proc/cpuinfo reading for Linux
        mock_open = unittest.mock.mock_open(read_data="""
processor       : 0
model name      : Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz
""")
        with patch('builtins.open', mock_open):
            cpu_info = get_cpu_info()
            self.assertEqual(cpu_info['processor'], self.mock_cpu_info['processor'])
            self.assertEqual(cpu_info['cores'], 8)
            self.assertEqual(cpu_info['threads'], 16)

    @patch('subprocess.run')
    def test_get_gpu_info_with_gpu(self, mock_run):
        """Test getting GPU information when GPU is available."""
        mock_run.return_value = MagicMock(
            stdout='0, NVIDIA GeForce RTX 3080, 10240, 535.129.03\n',
            returncode=0
        )

        gpu_info = get_gpu_info()
        self.assertEqual(len(gpu_info), 1)
        self.assertEqual(gpu_info[0]['name'], 'NVIDIA GeForce RTX 3080')
        self.assertEqual(gpu_info[0]['memory'], '10.0 GB')

    @patch('subprocess.run')
    def test_get_gpu_info_no_gpu(self, mock_run):
        """Test getting GPU information when no GPU is available."""
        mock_run.side_effect = subprocess.CalledProcessError(1, 'nvidia-smi')
        gpu_info = get_gpu_info()
        self.assertEqual(gpu_info, [])

    @patch('pyhea.utils.device.get_cpu_info')
    @patch('pyhea.utils.device.get_gpu_info')
    def test_check_device_info(self, mock_gpu_info, mock_cpu_info):
        """Test checking device information."""
        mock_cpu_info.return_value = self.mock_cpu_info
        mock_gpu_info.return_value = self.mock_gpu_info

        cpu_info, gpu_info = check_device_info()
        self.assertEqual(cpu_info, self.mock_cpu_info)
        self.assertEqual(gpu_info, self.mock_gpu_info)

    @patch('pyhea.utils.device.get_gpu_info')
    def test_check_device_availability(self, mock_gpu_info):
        """Test checking device availability."""
        # Test with GPU available
        mock_gpu_info.return_value = self.mock_gpu_info
        cpu_available, gpu_available = check_device_availability()
        self.assertTrue(cpu_available)
        self.assertTrue(gpu_available)

        # Test without GPU
        mock_gpu_info.return_value = []
        cpu_available, gpu_available = check_device_availability()
        self.assertTrue(cpu_available)
        self.assertFalse(gpu_available)

if __name__ == '__main__':
    unittest.main()
