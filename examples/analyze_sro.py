#!/usr/bin/env python3
"""
Analysis script for PyHEA output to calculate Short Range Order (SRO) parameters.
This script reads LAMMPS data file format and calculates SRO parameters between different atom types.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

def read_lammps_data(filename):
    """Read LAMMPS data file.
    
    @param filename str Path to LAMMPS data file
    @return tuple (positions, types, box, ntypes)
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    natoms = int(lines[1].split()[0])
    ntypes = int(lines[2].split()[0])
    
    # Parse box bounds
    box = np.zeros((3, 2))
    for i in range(3):
        box[i] = [float(x) for x in lines[3+i].split()[:2]]
    
    # Skip header lines until we find "Atoms"
    start_idx = 0
    for i, line in enumerate(lines):
        if "Atoms" in line:
            start_idx = i + 2  # Skip the "Atoms" line and the blank line
            break
    
    # Parse atoms
    positions = np.zeros((natoms, 3))
    types = np.zeros(natoms, dtype=int)
    
    for i in range(natoms):
        parts = lines[start_idx + i].split()
        atom_id = int(parts[0]) - 1  # Convert to 0-based indexing
        types[atom_id] = int(parts[1])
        positions[atom_id] = [float(x) for x in parts[2:5]]
    
    return positions, types, box, ntypes

def find_neighbors(positions, box, cutoff=3.5):
    """Find neighbors within cutoff using periodic boundary conditions.
    
    @param positions ndarray Atomic positions
    @param box ndarray Box dimensions
    @param cutoff float Cutoff radius
    @return list List of neighbor lists for each atom
    """
    # Create supercell for periodic boundaries
    box_size = box[:, 1] - box[:, 0]
    natoms = len(positions)
    
    # Create KD-tree with periodic boundaries
    tree = cKDTree(positions, boxsize=box_size)
    
    # Find all pairs within cutoff
    neighbors = [[] for _ in range(natoms)]
    pairs = tree.query_pairs(cutoff, output_type='ndarray')
    
    # Build neighbor lists
    for i, j in pairs:
        neighbors[i].append(j)
        neighbors[j].append(i)
    
    return neighbors

def calculate_sro(positions, types, box, cutoff=3.5):
    """Calculate Short Range Order (SRO) parameters.
    
    @param positions ndarray Atomic positions
    @param types ndarray Atom types
    @param box ndarray Box dimensions
    @param cutoff float Cutoff radius for neighbor analysis
    @return tuple (ndarray, dict) SRO values matrix and atom type counts
    """
    natoms = len(positions)
    unique_types = np.unique(types)
    ntypes = len(unique_types)
    
    # Count atoms of each type
    type_counts = {t: np.sum(types == t) for t in unique_types}
    
    # Find neighbors
    neighbors = find_neighbors(positions, box, cutoff)
    
    # Initialize pair counting matrix
    pair_counts = np.zeros((ntypes, ntypes), dtype=float)
    
    # Count neighbor pairs
    for i in range(natoms):
        type_i = types[i]
        for j in neighbors[i]:
            type_j = types[j]
            pair_counts[type_i-1, type_j-1] += 1
    
    # Calculate SRO parameters
    sro_values = np.zeros((ntypes, ntypes))
    for i, type_i in enumerate(unique_types):
        for j, type_j in enumerate(unique_types):
            ci = type_counts[type_i] / natoms
            cj = type_counts[type_j] / natoms
            zij = pair_counts[i, j] / natoms  # Average number of j neighbors around i
            
            # Warren-Cowley SRO parameter
            if zij > 0:
                sro_values[i, j] = 1 - zij / (ci * cj * np.sum(pair_counts) / natoms)
    
    return sro_values, type_counts

def plot_sro_heatmap(sro_values, atom_labels, output_file='sro_heatmap.png'):
    """Plot SRO parameters as a heatmap.
    
    @param sro_values ndarray Matrix of SRO values
    @param atom_labels list List of atom type labels
    @param output_file str Output file path for the plot
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(sro_values, cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar(label='SRO Parameter')
    
    # Add labels
    plt.xticks(range(len(atom_labels)), atom_labels)
    plt.yticks(range(len(atom_labels)), atom_labels)
    plt.xlabel('Element j')
    plt.ylabel('Element i')
    
    # Add text annotations
    for i in range(len(atom_labels)):
        for j in range(len(atom_labels)):
            plt.text(j, i, f'{sro_values[i,j]:.3f}', 
                    ha='center', va='center')
    
    plt.title('Short Range Order Parameters')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main analysis function."""
    # Configuration
    input_file = 'conf.lmp'
    atom_labels = ['Fe', 'Mn', 'Cr']  # Adjust based on your system
    cutoff = 3.5  # Adjust based on your lattice parameter
    
    # Read system
    print(f"Reading system from {input_file}...")
    positions, types, box, ntypes = read_lammps_data(input_file)
    
    print(f"Calculating SRO parameters (cutoff = {cutoff}Å)...")
    sro_values, type_counts = calculate_sro(positions, types, box, cutoff)
    
    # Print results
    print("\nAtom counts:")
    for type_i, count in type_counts.items():
        print(f"  Type {type_i} ({atom_labels[type_i-1]}): {count} atoms")
    
    print("\nSRO parameters:")
    for i in range(len(atom_labels)):
        for j in range(len(atom_labels)):
            print(f"  {atom_labels[i]}-{atom_labels[j]}: {sro_values[i,j]:.3f}")
    
    # Plot results
    print("\nGenerating heatmap plot...")
    plot_sro_heatmap(sro_values, atom_labels)
    print("Analysis complete! Check sro_heatmap.png for visualization.")

if __name__ == "__main__":
    main()
