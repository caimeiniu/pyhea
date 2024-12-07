import unittest
from mpi4py import MPI
import numpy as np

class TestComm(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.comm = MPI.COMM_WORLD
    
    def test_comm_size(self):
        """Test getting communicator size"""
        size = self.comm.Get_size()
        self.assertIsInstance(size, int)
        self.assertGreater(size, 0)
        
    def test_comm_rank(self):
        """Test getting process rank"""
        rank = self.comm.Get_rank()
        size = self.comm.Get_size()
        self.assertIsInstance(rank, int)
        self.assertGreaterEqual(rank, 0)
        self.assertLess(rank, size)
    
    def test_broadcast(self):
        """Test broadcasting data from root to all processes"""
        rank = self.comm.Get_rank()
        
        # Test with different data types
        test_data = None
        if rank == 0:
            test_data = {'value': 42, 'message': 'test'}
            
        result = self.comm.bcast(test_data, root=0)
        self.assertEqual(result['value'], 42)
        self.assertEqual(result['message'], 'test')
        
        # Test broadcasting numpy array
        array_data = None
        if rank == 0:
            array_data = np.array([1.0, 2.0, 3.0])
            
        result = self.comm.bcast(array_data, root=0)
        if rank != 0:
            self.assertTrue(np.array_equal(result, np.array([1.0, 2.0, 3.0])))
    
    def test_scatter_gather(self):
        """Test scatter and gather operations"""
        rank = self.comm.Get_rank()
        size = self.comm.Get_size()
        
        # Prepare data for scatter
        send_data = None
        if rank == 0:
            send_data = list(range(size))
            
        # Test scatter
        local_data = self.comm.scatter(send_data, root=0)
        self.assertEqual(local_data, rank)
        
        # Test gather
        local_result = local_data * 2
        gathered = self.comm.gather(local_result, root=0)
        
        if rank == 0:
            expected = [i * 2 for i in range(size)]
            self.assertEqual(gathered, expected)
    
    def test_allreduce(self):
        """Test allreduce operation"""
        rank = self.comm.Get_rank()
        size = self.comm.Get_size()
        
        # Each process contributes its rank
        local_value = rank
        
        # Sum up all ranks
        total = self.comm.allreduce(local_value, op=MPI.SUM)
        
        # Expected sum is 0 + 1 + 2 + ... + (size-1)
        expected_sum = (size - 1) * size // 2
        self.assertEqual(total, expected_sum)
    
    def test_barrier(self):
        """Test synchronization barrier"""
        # All processes should reach this point
        try:
            self.comm.Barrier()
            self.assertTrue(True)  # If we reach here, barrier worked
        except Exception as e:
            self.fail(f"Barrier failed: {str(e)}")

if __name__ == '__main__':
    unittest.main()
