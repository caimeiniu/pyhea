"""Output module for PyHEA.

This module handles the output functionality for PyHEA, including writing structure files
in various formats using dpdata for conversion.
"""

import os
import tempfile
from pathlib import Path
import dpdata
from pyhea.version import __version__

def write_structure(nest, latt, ntyp, elem, file, output_format='vasp/poscar'):
    """Write atomic structure data to the specified format.

    Creates a structure file in the requested format. The structure is first written
    as a POSCAR file and then converted using dpdata if a different format is requested.

    @param nest list List of atom types, where each entry is an integer index
    @param latt Lattice Lattice object containing vectors and coordinates
    @param ntyp int Number of atom types in the system
    @param elem list List of element symbols (will be replaced with A, B, C, ...)
    @param file str Output file path
    @param output_format str Desired output format (default: 'vasp/poscar')
                           Supported formats: vasp/poscar, lammps/lmp
    @return None
    @raises ValueError If the output format is not supported
    """
    # First write to POSCAR format
    poscar_file = file if output_format == 'vasp/poscar' else tempfile.NamedTemporaryFile(suffix='.poscar', delete=False).name
    write_poscar(nest, latt, ntyp, elem, poscar_file)

    # If a different format is requested, convert using dpdata
    if output_format != 'vasp/poscar':
        try:
            # Load the POSCAR file
            system = dpdata.System(poscar_file, fmt='vasp/poscar')
            
            # Determine the output format and extension
            format_extensions = {
                'lammps/lmp': '.lmp'
            }
            
            # Get the base path without extension
            base_path = os.path.splitext(file)[0]
            output_path = f"{base_path}{format_extensions.get(output_format, '')}"
            
            # Convert and save in the requested format
            if output_format == 'lammps/lmp':
                system.to_lammps_lmp(output_path)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
            
            # Clean up temporary POSCAR file
            if poscar_file != file:
                os.unlink(poscar_file)
                
        except Exception as e:
            if poscar_file != file:
                os.unlink(poscar_file)
            raise ValueError(f"Failed to convert to {output_format} format: {str(e)}")

def write_poscar(nest, latt, ntyp, elem, file):
    """Write atomic structure data in VASP POSCAR format.
    
    @param nest list List of atom types
    @param latt Lattice Lattice object containing vectors and coordinates
    @param ntyp int Number of atom types
    @param elem list List of elements (will be replaced with A, B, C, ...)
    @param file str Output file name
    @return None
    """
    natm = len(nest)
    coords = [""] * ntyp  # List to store coordinate strings for each type of atom
    
    # Count atoms of each type
    atom_counts = [0] * ntyp
    for i in range(natm):
        atom_counts[nest[i]] += 1
    
    # Generate element labels (A, B, C, ...)
    element_labels = [chr(65 + i) for i in range(ntyp)]  # 65 is ASCII for 'A'
    
    with open(file, "w") as final_coords:
        # Write header with version
        final_coords.write(f"PyHEA v{__version__}\n")
        # Write scaling factor
        final_coords.write(f"{latt.latt_con}\n")
        # Write lattice vectors
        final_coords.write(f"{latt.latt_vec[0][0]}\t{latt.latt_vec[0][1]}\t{latt.latt_vec[0][2]}\n")
        final_coords.write(f"{latt.latt_vec[1][0]}\t{latt.latt_vec[1][1]}\t{latt.latt_vec[1][2]}\n")
        final_coords.write(f"{latt.latt_vec[2][0]}\t{latt.latt_vec[2][1]}\t{latt.latt_vec[2][2]}\n")
        # Write element labels
        final_coords.write("   ".join(element_labels) + "\n")
        # Write atom counts
        final_coords.write("   ".join(map(str, atom_counts)) + "\n")
        # Write coordinate type
        final_coords.write("Cartesian\n")
        
        # Collect coordinates by type
        for i in range(natm):
            coords[nest[i]] += f"{latt.coords[i][0]}\t{latt.coords[i][1]}\t{latt.coords[i][2]}\n"
        
        # Write coordinates for each type
        for i in range(ntyp):
            final_coords.write(coords[i])