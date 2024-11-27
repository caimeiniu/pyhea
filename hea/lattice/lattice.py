import numpy as np
from hea.utils.logger import logger

# Define constants for the number of neighbors per shell in BCC lattice
BCC_NUM_NEIGHBORS_SHELL1 = 8
BCC_NUM_NEIGHBORS_SHELL2 = 6
BCC_NUM_NEIGHBORS_SHELL3 = 12
BCC_TOTAL_NEIGHBORS_PER_ATOM = BCC_NUM_NEIGHBORS_SHELL1 + BCC_NUM_NEIGHBORS_SHELL2 + BCC_NUM_NEIGHBORS_SHELL3

FCC_NUM_NEIGHBORS_SHELL1 = 12
FCC_NUM_NEIGHBORS_SHELL2 = 6
FCC_NUM_NEIGHBORS_SHELL3 = 12
FCC_TOTAL_NEIGHBORS_PER_ATOM = FCC_NUM_NEIGHBORS_SHELL1 + FCC_NUM_NEIGHBORS_SHELL2 + FCC_NUM_NEIGHBORS_SHELL3

class lattice:
    """
    Class representing a Body-Centered Cubic (BCC) lattice.

    This class provides methods to generate the atomic coordinates of a BCC lattice,
    calculate periodic boundary conditions, build neighbor lists, and validate
    neighbor relationships based on shell distances.
    """
    def __init__(self, cell_dim, latt_typ, latt_con, valid=False):
        """
        Initializes the lattice object.

        @param cell_dim The number of unit cells along each dimension (e.g., for a 4x4x4 grid, cell_dim is 4).
        @param latt_con The lattice constant, representing the distance between atoms in the lattice.
        """
        # BCC lattice has 2 atoms per unit cell
        self.latt_typ  = latt_typ
        self.cell_dim  = cell_dim[0]
        self.latt_con  = np.float16(latt_con)
        self.latt_vec  = np.array([self.cell_dim * latt_con] * 3, dtype=np.float16)
        
        if   self.latt_typ == 'FCC':
            # Generate FCC lattice
            self.coords    = self.calc_supercell_fcc(self.cell_dim, latt_con)
            self.nbor_list = self.build_neighbor_list_fcc(self.cell_dim, latt_con)
        elif self.latt_typ == 'BCC':
            # Generate BCC lattice
            self.coords    = self.calc_supercell_bcc(self.cell_dim, latt_con)
            self.nbor_list = self.build_neighbor_list_bcc(self.cell_dim, latt_con)
        else:
            raise ValueError(f"Invalid lattice type: {self.latt_typ}")
        
        if valid:
            self.valid_nbors(self.coords, self.cell_dim, latt_con, latt_typ, self.nbor_list)

    def calc_pbc(self, coord):
        """
        Applies periodic boundary conditions (PBC) to a coordinate.

        @param coord The coordinate of the atom.
        @return The coordinate adjusted for periodic boundary conditions.
        """
        return coord - np.floor(coord / self.latt_vec) * self.latt_vec


    def calc_pbc_dist(self, a, b):
        """
        Calculates the periodic boundary condition distance between two atoms.

        @param a The first atom's coordinate.
        @param b The second atom's coordinate.
        @return The periodic boundary condition distance between the two atoms.
        """
        delta = a - b
        delta = np.where(delta > self.latt_vec / np.float16(2.0), delta - self.latt_vec, delta)
        delta = np.where(delta < -self.latt_vec / np.float16(2.0), delta + self.latt_vec, delta)
        return np.sqrt(np.sum(delta ** 2, axis=-1)).astype(np.float16)


    def calc_atom_id_bcc(self, coord, cell_dim):
        """
        Calculates the atom index in the BCC lattice based on its coordinate.

        @param coord The coordinate of the atom.
        @param cell_dim The dimension of the supercell grid (e.g., 4 for a 4x4x4 grid).
        @return The atom index corresponding to the given coordinate.
        """
        natms = cell_dim ** 3 * 2
        # Calculate the indices of the unit cell along each dimension
        xx = int(coord[0] // self.latt_con)
        yy = int(coord[1] // self.latt_con)
        zz = int(coord[2] // self.latt_con)

        # Calculate the relative position within the unit cell to determine the atom type (0 or 1)
        # BCC unit cell has 2 atoms: (0,0,0), (0.5,0.5,0.5)
        __xx = np.mod(coord[0] / self.latt_con, 1)
        __yy = np.mod(coord[1] / self.latt_con, 1)
        __zz = np.mod(coord[2] / self.latt_con, 1)

        if   __xx == 0.0 and __yy == 0.0 and __zz == 0.0:
            loc = 0   # (0.0, 0.0, 0.0)
        elif __xx == 0.5 and __yy == 0.5 and __zz == 0.5:
            loc = 1   # (0.5, 0.5, 0.5)
        else:
            raise ValueError(f"Invalid coordinate for BCC lattice: {coord}")

        # Calculate the atom's index within the supercell
        id = 2 * (xx * cell_dim**2 + yy * cell_dim + zz) + loc

        return id % natms


    def calc_atom_id_fcc(self, coord, cell_dim):
        """
        Calculates the atom index in the FCC lattice based on its coordinate.

        @param coord The coordinate of the atom.
        @param cell_dim The dimension of the supercell grid (e.g., 4 for a 4x4x4 grid).
        @return The atom index corresponding to the given coordinate.
        """
        natms = cell_dim ** 3 * 4
        # Calculate the indices of the unit cell along each dimension
        xx = int(coord[0] // self.latt_con)
        yy = int(coord[1] // self.latt_con)
        zz = int(coord[2] // self.latt_con)

        # Calculate the relative position within the unit cell to determine the atom type (0 to 3)
        # FCC unit cell has 4 atoms: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
        __xx = np.mod(coord[0] / self.latt_con, 1)
        __yy = np.mod(coord[1] / self.latt_con, 1)
        __zz = np.mod(coord[2] / self.latt_con, 1)

        if   __xx == 0.0 and __yy == 0.0 and __zz == 0.0:
            loc = 0  # (0.0, 0.0, 0.0)
        elif __xx == 0.5 and __yy == 0.5 and __zz == 0.0:
            loc = 1  # (0.5, 0.5, 0.0)
        elif __xx == 0.5 and __yy == 0.0 and __zz == 0.5:
            loc = 2  # (0.5, 0.0, 0.5)
        elif __xx == 0.0 and __yy == 0.5 and __zz == 0.5:
            loc = 3  # (0.0, 0.5, 0.5)
        else:
            raise ValueError(f"Invalid coordinate for FCC lattice: {coord}")

        # Calculate the atom's index within the supercell
        id = 4 * (xx * cell_dim**2 + yy * cell_dim + zz) + loc
        
        return id % natms

    def calc_supercell_bcc(self, cell_dim, latt_con):
        """
        Generates the atomic coordinates for a BCC supercell.
        
        @param cell_dim The dimension of the supercell grid (e.g., 4 for a 4x4x4 grid).
        @param latt_con The lattice constant, representing the distance between atoms in the lattice.

        @return A NumPy array containing the coordinates of all atoms in the BCC supercell.
        """
        id = 0
        natms = cell_dim ** 3 * 2
        coords = np.zeros((natms, 3), dtype=np.float16)
        for ix in range(cell_dim):
            for iy in range(cell_dim):
                for iz in range(cell_dim):
                    # First atom at (0,0,0)
                    coords[id] = [(ix + 0.0) * latt_con, (iy + 0.0) * latt_con, (iz + 0.0) * latt_con]
                    id += 1
                    # Second atom at (0.5,0.5,0.5)
                    coords[id] = [(ix + 0.5) * latt_con, (iy + 0.5) * latt_con, (iz + 0.5) * latt_con]
                    id += 1
        return coords

    def calc_supercell_fcc(self, cell_dim, latt_con):
        """
        Generates the atomic coordinates for an FCC supercell.
        
        @param cell_dim The dimension of the supercell grid (e.g., 4 for a 4x4x4 grid).
        @param latt_con The lattice constant, representing the distance between atoms in the lattice.

        @return A NumPy array containing the coordinates of all atoms in the FCC supercell.
        """
        id = 0
        natms = cell_dim ** 3 * 4
        coords = np.zeros((natms, 3), dtype=np.float16)
        for ix in range(cell_dim):
            for iy in range(cell_dim):
                for iz in range(cell_dim):
                    # First atom at (0,0,0)
                    coords[id] = [(ix + 0.0) * latt_con, (iy + 0.0) * latt_con, (iz + 0.0) * latt_con]
                    id += 1
                    # Second atom at (0.5,0.5,0)
                    coords[id] = [(ix + 0.5) * latt_con, (iy + 0.5) * latt_con, (iz + 0.0) * latt_con]
                    id += 1
                    # Third atom at (0.5,0,0.5)
                    coords[id] = [(ix + 0.5) * latt_con, (iy + 0.0) * latt_con, (iz + 0.5) * latt_con]
                    id += 1
                    # Fourth atom at (0,0.5,0.5)
                    coords[id] = [(ix + 0.0) * latt_con, (iy + 0.5) * latt_con, (iz + 0.5) * latt_con]
                    id += 1
        return coords

    def build_neighbor_list_bcc(self, cell_dim, latt_con):
        """
        Builds the neighbor list for each atom in the BCC lattice.

        @param cell_dim The dimension of the supercell grid (e.g., 4 for a 4x4x4 grid).
        @return A NumPy array containing the neighbor indices for each atom.
        """
        natms = cell_dim ** 3 * 2
        nbor = []

        # Define neighbor vectors for the three shells
        neighbor_vectors_shell1 = np.array([
            [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5], [0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, -0.5]
        ], dtype=np.float16) * latt_con

        neighbor_vectors_shell2 = np.array([
            [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]
        ], dtype=np.float16) * latt_con

        neighbor_vectors_shell3 = np.array([
            [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, -1.0, 0.0],
            [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 0.0, -1.0],
            [0.0, 1.0, 1.0], [0.0, -1.0, 1.0], [0.0, 1.0, -1.0], [0.0, -1.0, -1.0]
        ], dtype=np.float16) * latt_con

        # Assign neighbors for each atom
        for id in range(natms):
            coord = self.coords[id]
            count = 0

            arr = []
            var = []
            for delta in neighbor_vectors_shell1:
                nbor_coord = self.calc_pbc(coord + delta)
                nbor_id = self.calc_atom_id_bcc(nbor_coord, cell_dim)
                # nbor[id, count] = nbor_id
                arr.append(nbor_id)
                count += 1
            var.append(np.sort(arr))
            
            arr = []
            for delta in neighbor_vectors_shell2:
                nbor_coord = self.calc_pbc(coord + delta)
                nbor_id = self.calc_atom_id_bcc(nbor_coord, cell_dim)
                # nbor[id, count] = nbor_id
                arr.append(nbor_id)
                count += 1
            # nbor[id, BCC_NUM_NEIGHBORS_SHELL1:BCC_NUM_NEIGHBORS_SHELL1 + BCC_NUM_NEIGHBORS_SHELL2] = np.sort(arr)
            var.append(np.sort(arr))
            
            arr = []
            for delta in neighbor_vectors_shell3:
                nbor_coord = self.calc_pbc(coord + delta)
                nbor_id = self.calc_atom_id_bcc(nbor_coord, cell_dim)
                # nbor[id, count] = nbor_id
                arr.append(nbor_id)
                count += 1
            # nbor[id, BCC_NUM_NEIGHBORS_SHELL1 + BCC_NUM_NEIGHBORS_SHELL2:] = np.sort(arr)
            var.append(np.sort(arr))
            nbor.append(var)

        return nbor

    def build_neighbor_list_fcc(self, cell_dim, latt_con):
        """
        Builds the neighbor list for each atom in the FCC lattice.

        @param cell_dim The dimension of the supercell grid (e.g., 4 for a 4x4x4 grid).
        @return A NumPy array containing the neighbor indices for each atom.
        """
        natms = cell_dim ** 3 * 4
        nbor = []

        # Define neighbor vectors for the three shells
        neighbor_vectors_shell1 = np.array([
            [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0], [0.5, -0.5, 0.0], [-0.5, -0.5, 0.0],
            [0.5, 0.0, 0.5], [-0.5, 0.0, 0.5], [0.5, 0.0, -0.5], [-0.5, 0.0, -0.5],
            [0.0, 0.5, 0.5], [0.0, -0.5, 0.5], [0.0, 0.5, -0.5], [0.0, -0.5, -0.5]
        ], dtype=np.float16) * latt_con

        neighbor_vectors_shell2 = np.array([
            [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]
        ], dtype=np.float16) * latt_con

        neighbor_vectors_shell3 = np.array([
            [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, -1.0, 0.0],
            [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 0.0, -1.0],
            [0.0, 1.0, 1.0], [0.0, -1.0, 1.0], [0.0, 1.0, -1.0], [0.0, -1.0, -1.0]
        ], dtype=np.float16) * latt_con

        # Assign neighbors for each atom
        for id in range(natms):
            coord = self.coords
            count = 0
        
            arr = []
            var = []
            for delta in neighbor_vectors_shell1:
                nbor_coord = self.calc_pbc(coord[id] + delta)
                nbor_id = self.calc_atom_id_fcc(nbor_coord, cell_dim)
                # nbor[id, count] = nbor_id
                arr.append(nbor_id)
                count += 1
            var.append(np.sort(arr))

            arr = []
            for delta in neighbor_vectors_shell2:
                nbor_coord = self.calc_pbc(coord[id] + delta)
                nbor_id = self.calc_atom_id_fcc(nbor_coord, cell_dim)
                # nbor[id, count] = nbor_id
                arr.append(nbor_id)
                count += 1
            var.append(np.sort(arr))

            arr = []
            for delta in neighbor_vectors_shell3:
                nbor_coord = self.calc_pbc(coord[id] + delta)
                nbor_id = self.calc_atom_id_fcc(nbor_coord, cell_dim)
                # nbor[id, count] = nbor_id
                arr.append(nbor_id)
                count += 1
            var.append(np.sort(arr))
            nbor.append(var)

        return nbor
    

    def valid_nbors_bcc(self, coords, cell_dim, latt_con, nbor_list):
        """
        Validates the neighbor list for the BCC lattice.

        This function compares the calculated distances between atoms and their
        neighbors against expected distances for three neighbor shells. It reports
        any discrepancies and prints whether the neighbor list validation passed.
        """
        shell1 = np.sqrt(np.float16(3.0)) * latt_con / np.float16(2.0)
        shell2 = latt_con
        shell3 = np.sqrt(np.float16(2.0)) * latt_con

        natms = cell_dim ** 3 * 2
        
        passed = True
        threhold = np.float16(1e-2)  # Adjusted for float16 precision

        for ii in range(natms):
            # Validate first shell neighbors
            for id in range(BCC_NUM_NEIGHBORS_SHELL1):
                nbor = nbor_list[ii][0][id]
                dist = self.calc_pbc_dist(coords[ii], coords[nbor])
                if np.abs(dist - shell1) > threhold:
                    logger.info(f"Error: Atom {ii} has an incorrect first shell neighbor at dist {dist} (expected {shell1})")
                    passed = False

            # Validate second shell neighbors
            for id in range(BCC_NUM_NEIGHBORS_SHELL1, BCC_NUM_NEIGHBORS_SHELL1 + BCC_NUM_NEIGHBORS_SHELL2):
                nbor = nbor_list[ii][1][id - BCC_NUM_NEIGHBORS_SHELL1]
                dist = self.calc_pbc_dist(coords[ii], coords[nbor])
                if np.abs(dist - shell2) > threhold:
                    logger.info(f"Error: Atom {ii} has an incorrect second shell neighbor at dist {dist} (expected {shell2})")
                    passed = False

            # Validate third shell neighbors
            for id in range(BCC_NUM_NEIGHBORS_SHELL1 + BCC_NUM_NEIGHBORS_SHELL2, BCC_TOTAL_NEIGHBORS_PER_ATOM):
                nbor = nbor_list[ii][2][id - BCC_NUM_NEIGHBORS_SHELL1 - BCC_NUM_NEIGHBORS_SHELL2]
                dist = self.calc_pbc_dist(coords[ii], coords[nbor])
                if np.abs(dist - shell3) > threhold:
                    logger.info(f"Error: Atom {ii} has an incorrect third shell neighbor at dist {dist} (expected {shell3})")
                    passed = False

        if passed:
            logger.info("Neighbor list validation passed: All distances match the expected shell values.")
        else:
            logger.info("Neighbor list validation failed.")

    def valid_nbors_fcc(self, coords, cell_dim, latt_con, nbor_list):
        """
        Validates the neighbor list for the FCC lattice.

        This function compares the calculated distances between atoms and their
        neighbors against expected distances for three neighbor shells. It reports
        any discrepancies and prints whether the neighbor list validation passed.
        """
        shell1 = latt_con / np.float16(np.sqrt(2.0))
        shell2 = latt_con
        shell3 = latt_con * np.float16(np.sqrt(2.0))

        natms = cell_dim ** 3 * 4
        
        passed = True
        threhold = np.float16(1e-2)
        
        for ii in range(natms):
            # Validate first shell neighbors
            for id in range(FCC_NUM_NEIGHBORS_SHELL1):
                nbor = nbor_list[ii][0][id]
                dist = self.calc_pbc_dist(coords[ii], coords[nbor])
                if np.abs(dist - shell1) > threhold:
                    logger.info(f"Error: Atom {ii} has an incorrect first shell neighbor at dist {dist} (expected {shell1})")
                    passed = False

            # Validate second shell neighbors
            for id in range(FCC_NUM_NEIGHBORS_SHELL1, FCC_NUM_NEIGHBORS_SHELL1 + FCC_NUM_NEIGHBORS_SHELL2):
                nbor = nbor_list[ii][1][id - FCC_NUM_NEIGHBORS_SHELL1]
                dist = self.calc_pbc_dist(coords[ii], coords[nbor])
                if np.abs(dist - shell2) > threhold:
                    logger.info(f"Error: Atom {ii} has an incorrect second shell neighbor at dist {dist} (expected {shell2})")
                    passed = False

            # Validate third shell neighbors
            for id in range(FCC_NUM_NEIGHBORS_SHELL1 + FCC_NUM_NEIGHBORS_SHELL2, FCC_TOTAL_NEIGHBORS_PER_ATOM):
                nbor = nbor_list[ii][2][id - FCC_NUM_NEIGHBORS_SHELL1 - FCC_NUM_NEIGHBORS_SHELL2]
                dist = self.calc_pbc_dist(coords[ii], coords[nbor])
                if np.abs(dist - shell3) > threhold:
                    logger.info(f"Error: Atom {ii} has an incorrect third shell neighbor at dist {dist} (expected {shell3})")
                    passed = False
                    
        if passed:
            logger.info("Neighbor list validation passed: All distances match the expected shell values.")
        else:
            logger.info("Neighbor list validation failed.")
    
    def valid_nbors(self, coords, cell_dim, latt_con, latt_typ, nbor_list):
        """
        Validates the neighbor list for the lattice.

        This function compares the calculated distances between atoms and their
        neighbors against expected distances for three neighbor shells. It reports
        any discrepancies and prints whether the neighbor list validation passed.
        """
        if   latt_typ == 'BCC':
            self.valid_nbors_bcc(coords, cell_dim, latt_con, nbor_list)
        elif latt_typ == 'FCC':
            self.valid_nbors_fcc(coords, cell_dim, latt_con, nbor_list)
        else:
            raise ValueError(f"Invalid lattice type: {latt_typ}")
    
    @property
    def shell1(self):
        if   self.latt_typ == 'FCC':
            return FCC_NUM_NEIGHBORS_SHELL1
        elif self.latt_typ == 'BCC':
            return BCC_NUM_NEIGHBORS_SHELL1
        else:
            raise ValueError(f"Invalid lattice type: {self.latt_typ}")
    
    @property
    def shell2(self):
        if   self.latt_typ == 'FCC':
            return FCC_NUM_NEIGHBORS_SHELL2
        elif self.latt_typ == 'BCC':
            return BCC_NUM_NEIGHBORS_SHELL2
        else:
            raise ValueError(f"Invalid lattice type: {self.latt_typ}")
        
    @property
    def shell3(self):
        if   self.latt_typ == 'FCC':
            return FCC_NUM_NEIGHBORS_SHELL3
        elif self.latt_typ == 'BCC':
            return BCC_NUM_NEIGHBORS_SHELL3
        else:
            raise ValueError(f"Invalid lattice type: {self.latt_typ}")
        
    @property
    def nbor_size(self):
        if   self.latt_typ == 'FCC':
            return FCC_TOTAL_NEIGHBORS_PER_ATOM
        elif self.latt_typ == 'BCC':
            return BCC_TOTAL_NEIGHBORS_PER_ATOM
        else:
            raise ValueError(f"Invalid lattice type: {self.latt_typ}")
    
    @property
    def shells(self):
        # Return the number of shells for the lattice
        # We build neighbor lists for three shells
        return 3
    