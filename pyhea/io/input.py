import os
import yaml
import numpy as np

class input_config:
    """
    A class to handle configuration input from a YAML file.

    Attributes
    ----------
    file_path : str
        The path to the YAML configuration file.
    config : dict
        The configuration data loaded from the YAML file.
    structure_data : dict
        The structure data loaded from the POSCAR file.

    Methods
    -------
    element:
        Returns the list of elements from the configuration.
    cell_dim:
        Returns the cell dimensions from the configuration.
    solutions:
        Returns the number of solutions from the configuration.
    mc_step:
        Returns the number of Monte Carlo steps from the configuration.
    total_iter:
        Returns the total number of iterations from the configuration.
    global_iter:
        Returns the number of global iterations from the configuration.
    max_shell_num:
        Returns the maximum shell number from the configuration.
    weight:
        Returns the weight list from the configuration.
    structure:
        Returns the structure from the configuration.
    _read_structure:
        Reads and validates the POSCAR structure file.
    """

    # Allowed keys in the configuration file
    ALLOWED_KEYS = {
        'type', 'element', 'cell_dim', 'solutions', 'device', 'total_iter', 'weight',
        'parallel_task', 'converge_depth', 'max_shell_num', 'structure', 'output', 'target_sro'
    }

    # Supported output formats
    SUPPORTED_FORMATS = {
        'vasp/poscar': 'VASP POSCAR format (default)',
        'lammps/lmp': 'LAMMPS data format'
    }

    def __init__(self, file_path):
        self.file_path = file_path
        self.config = self._read_yaml()
        self.structure = self._read_structure()
        self._validate_config()

    def _read_yaml(self):
        try:
            with open(self.file_path, 'r') as file:
                data = yaml.safe_load(file)
                if data is None or not isinstance(data, dict):
                    raise ValueError(f"Invalid or empty YAML file: {self.file_path}")
                if not all(key in self.ALLOWED_KEYS for key in data):
                    raise ValueError(f"Invalid keys in YAML file: {self.file_path}")
                return data
        except FileNotFoundError as e:
            raise FileNotFoundError(f"YAML file not found: {self.file_path}") from e
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {self.file_path}") from e

    def _read_structure(self):
        """
        Reads and validates the POSCAR structure file.

        Raises
        ------
        ValueError:
            If the POSCAR file is malformed or contains invalid data.
        """

        structure_file = self.config.get('structure', None)
        structure_data = {}
        current_dir = os.getcwd()
        structure_file = os.path.join(current_dir, structure_file)
        try:
            with open(structure_file, 'r') as file:
                # Read all lines and filter out empty lines and comments
                lines = [line.split('#')[0].strip() for line in file if line.strip()]
                # Extracting relevant data from the POSCAR file
                structure_data['comment'] = lines[0].strip()
                structure_data['lattice_constant'] = float(lines[1].strip())

                structure_data['lattice_vectors'] = [
                    list(map(float, lines[2].split())),
                    list(map(float, lines[3].split())),
                    list(map(float, lines[4].split()))
                ]

                structure_data['num_atoms'] = int(lines[5].strip())
                structure_data['coordinate_type'] = lines[6].strip().lower()
                if structure_data['coordinate_type'] not in ['direct', 'cartesian']:
                    raise ValueError("Coordinate type must be either 'direct' or 'cartesian'.")

                structure_type = ''
                if structure_data['num_atoms'] == 2:
                    structure_type = 'BCC'
                elif structure_data['num_atoms'] == 4:
                    structure_type = 'FCC'
                else:
                    raise ValueError(f"The lattice type is not supported for {structure_data['num_atoms']} atoms.")

                if len(lines[7:]) != structure_data['num_atoms']:
                    raise ValueError("Number of atomic positions does not match declared atom count")

                positions = []
                for line in lines[7:]:
                    if len(line.split()) != 3:
                        raise ValueError("Invalid atomic position format")
                    positions.append(list(map(float, line.split())))
                structure_data['positions'] = positions

        except FileNotFoundError as e:
            raise FileNotFoundError(f"POSCAR file not found: {structure_file}") from e

        self._validate_structure(structure_data)
        return structure_data

    def _validate_config(self):
        """Validate the configuration data.

        Raises
        ------
        ValueError:
            If any configuration values are invalid.
        """
        # Validate element counts
        total_atoms = sum(self.config.get('element', []))
        cell_volume = np.prod(self.config.get('cell_dim', [1, 1, 1]))
        atoms_per_cell = self.structure.get('num_atoms', 0)
        expected_atoms = cell_volume * atoms_per_cell

        # Validate weights
        weights = self.config.get('weight', [])
        if any(w <= 0 for w in weights):
            raise ValueError("All weights must be positive values")

        # Validate cell dimensions
        cell_dim = self.config.get('cell_dim', [])
        if not all(isinstance(d, int) and d > 0 for d in cell_dim):
            raise ValueError("Cell dimensions must be positive integers")
        if len(set(cell_dim)) != 1:
            raise ValueError("Only cubic cell dimensions are supported. All dimensions must be equal.")

        if total_atoms != expected_atoms:
            raise ValueError(f"The sum of elements {total_atoms} does not match the total number of atoms {expected_atoms}")

        # Validate target SRO file
        target_sro_file = self.config.get('target_sro')
        if target_sro_file:
            if not os.path.exists(target_sro_file):
                raise FileNotFoundError(f"Target SRO file not found: {target_sro_file}")
            self._validate_sro_file(target_sro_file)
        
        # size of elements and type should be same
        if len(self.config.get('element', [])) != self.config.get('type'):
            raise ValueError("Number of elements should be equal to type")

    def _validate_sro_file(self, sro_file):
        """Validate the SRO file format and content.

        Parameters
        ----------
        sro_file : str
            Path to the SRO file.

        Raises
        ------
        ValueError:
            If the SRO file format is invalid.
        """
        try:
            with open(sro_file, 'r') as f:
                lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            num_types = self.config.get('type', 2)
            expected_values = (num_types * (num_types + 1)) // 2  # Number of values in lower triangular matrix
            
            # Check each shell's data
            current_shell = None
            count = 0
            for line in lines:
                if not any(char.isdigit() for char in line):
                    current_shell = line  # Shell name/header
                    count = 0
                else:
                    if current_shell is None:
                        raise ValueError("SRO file must start with a shell name")
                    count += 1
                    values = list(map(float, line.split()))
                    if len(values) > expected_values:
                        raise ValueError(f"Too many values for shell {current_shell}. Expected {expected_values} values.")
                    if len(values) != count:
                        raise ValueError(f"Invalid number of values for shell {current_shell}. Expected {expected_values} values.")
        except ValueError as e:
            raise ValueError(f"Invalid SRO file format: {str(e)}")

    def _validate_structure(self, structure_data):
        """Validate the structure data from POSCAR.

        Parameters
        ----------
        structure_data : dict
            The structure data to validate.

        Raises
        ------
        ValueError:
            If the structure data is invalid.
        """
        # Check number of atoms matches coordinates
        if len(structure_data.get('positions', [])) != structure_data.get('num_atoms', 0):
            raise ValueError("Number of atomic positions does not match declared atom count")

        # Check for cubic cell
        vectors = structure_data.get('lattice_vectors', [])
        if not all(len(v) == 3 for v in vectors):
            raise ValueError("Invalid lattice vectors")
        
        # Check for orthogonal vectors
        for i in range(3):
            for j in range(i+1, 3):
                dot_product = sum(a*b for a, b in zip(vectors[i], vectors[j]))
                if abs(dot_product) > 1e-10:  # Allow for numerical precision
                    raise ValueError("Non-orthogonal lattice vectors are not supported")
        
        # Check for equal lengths (cubic)
        lengths = [sum(x*x for x in v)**0.5 for v in vectors]
        if not all(abs(lengths[0] - l) < 1e-10 for l in lengths[1:]):
            raise ValueError("Non-cubic cell detected")

    @property
    def type(self):
        return self.config.get('type', [])

    @property
    def element(self):
        return self.config.get('element', [])

    @property
    def cell_dim(self):
        return self.config.get('cell_dim', [])

    @property
    def solutions(self):
        return self.config.get('solutions', 0)

    @property
    def parallel_task(self):
        return self.config.get('parallel_task', 0)

    @property
    def converge_depth(self):
        return self.config.get('converge_depth', 0)

    @property
    def total_iter(self):
        return self.config.get('total_iter', 0)

    @property
    def max_shell_num(self):
        if self.config.get('weight', 0) == 0:
            return 0
        else:
            return len(self.config.get('weight', []))

    @property
    def weight(self):
        return self.config.get('weight', [])
    
    @property
    def device(self):
        return self.config.get('device', 'cpu')

    @property
    def latt_type(self):
        if   self.structure.get('num_atoms', 0) == 2:
            return 'BCC'
        elif self.structure.get('num_atoms', 0) == 4:
            return 'FCC'
        else:
            raise ValueError(f"The lattice type is not supported for {self.config.get('num_atoms', 0)} atoms.")
    
    @property
    def latt_const(self):
        return self.structure.get('lattice_constant', 0)
    
    @property
    def latt_vectors(self):
        return self.structure.get('lattice_vectors', [])

    @property
    def output_format(self):
        """Get the output format configuration.

        Returns:
            str: The output format (default: 'poscar')
        """
        output_config = self.config.get('output', {})
        format_name = output_config.get('format', 'vasp/poscar').lower()
        
        if format_name not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported output format: {format_name}. "
                           f"Supported formats are: {', '.join(self.SUPPORTED_FORMATS.keys())}")
        
        return format_name

    @property
    def output_name(self):
        """Get the output name.
        
        @return str Output name (default: 'output')
        """
        output_config = self.config.get('output', {})
        return output_config.get('name', 'output')

    @property
    def target_sro(self):
        """Get the target SRO values from the specified file.
        
        Returns:
            list[numpy.ndarray]: A list of arrays containing the target SRO values for each shell.
            Always returns 3 shells, with default values of 0 if not specified in the file.
            Each array has length num_types * (num_types + 1) / 2 for upper triangular format.
        
        Raises:
            FileNotFoundError: If the specified target_sro file doesn't exist
            ValueError: If the file format is invalid or values don't match the number of atom types
        """
        num_types = self.type
        expected_length = (num_types * (num_types + 1)) // 2
        
        # Initialize default values: 3 shells of zeros
        default_shell = np.zeros(expected_length)
        default_shells = [default_shell.copy() for _ in range(3)]
        
        target_sro_file = self.config.get('target_sro', None)
        if target_sro_file is None:
            return np.array([[0.0] * (num_types * num_types)] * 3)
            
        current_dir = os.getcwd()
        target_sro_file = os.path.join(current_dir, target_sro_file)
        
        try:
            with open(target_sro_file, 'r') as f:
                # Read all lines and filter out empty lines
                all_lines = [line.strip() for line in f if line.strip()]

                
                # Initialize variables
                current_shell = []
                all_shells = []
                shell_count = 0
                
                # Process each line
                for line in all_lines:
                    # Skip comment lines
                    if line.startswith('#'):
                        continue
                        
                    # If we find a shell marker, start a new shell
                    if line.lower().startswith('shell') or line.lower().startswith('second shell') or line.lower().startswith('third shell'):
                        if current_shell:
                            if len(current_shell) != expected_length:
                                raise ValueError(f"Shell {shell_count+1} has incorrect number of values. Expected {expected_length}, got {len(current_shell)}")
                            all_shells.append(np.array(current_shell))
                            shell_count += 1
                            current_shell = []
                        continue
                    
                    # Parse values
                    try:
                        values = list(map(float, line.split()))
                        current_shell.extend(values)
                    except ValueError:
                        continue  # Skip lines that can't be parsed as floats
                
                # Add the last shell if it exists
                if current_shell:
                    if len(current_shell) != expected_length:
                        raise ValueError(f"Shell {shell_count+1} has incorrect number of values. Expected {expected_length}, got {len(current_shell)}")
                    all_shells.append(np.array(current_shell))
                    shell_count += 1
                
                # Replace default shells with read values
                for i, shell in enumerate(all_shells):
                    if i < 3:  # Only use first 3 shells
                        default_shells[i] = shell

                full_shells = np.array([[0.0] * (num_types * num_types)] * 3)
                for shell in range(3):
                    if num_types == 2:
                        full_shells[shell][0] = default_shells[shell][0]
                        full_shells[shell][1] = default_shells[shell][1]
                        full_shells[shell][2] = default_shells[shell][1]
                        full_shells[shell][3] = default_shells[shell][2]
                    elif num_types == 3:
                        full_shells[shell][0] = default_shells[shell][0]
                        full_shells[shell][1] = default_shells[shell][1]
                        full_shells[shell][2] = default_shells[shell][3]
                        full_shells[shell][3] = default_shells[shell][1]
                        full_shells[shell][4] = default_shells[shell][2]
                        full_shells[shell][5] = default_shells[shell][4]
                        full_shells[shell][6] = default_shells[shell][3]
                        full_shells[shell][7] = default_shells[shell][4]
                        full_shells[shell][8] = default_shells[shell][5]
                    elif num_types == 4:
                        full_shells[shell][0] = default_shells[shell][0]
                        full_shells[shell][1] = default_shells[shell][1]
                        full_shells[shell][2] = default_shells[shell][3]
                        full_shells[shell][3] = default_shells[shell][6]
                        full_shells[shell][4] = default_shells[shell][1]
                        full_shells[shell][5] = default_shells[shell][2]
                        full_shells[shell][6] = default_shells[shell][4]
                        full_shells[shell][7] = default_shells[shell][7]
                        full_shells[shell][8] = default_shells[shell][3]
                        full_shells[shell][9] = default_shells[shell][4]
                        full_shells[shell][10] = default_shells[shell][5]
                        full_shells[shell][11] = default_shells[shell][8]
                        full_shells[shell][12] = default_shells[shell][6]
                        full_shells[shell][13] = default_shells[shell][7]
                        full_shells[shell][14] = default_shells[shell][8]
                        full_shells[shell][15] = default_shells[shell][9]
                    else:
                        raise ValueError(f"Unsupported number of atom types: {num_types}")
                    
                return full_shells
                
        except FileNotFoundError:
            raise FileNotFoundError(f"Target SRO file not found: {target_sro_file}")
        except ValueError as e:
            raise ValueError(f"Error parsing target SRO file: {str(e)}")