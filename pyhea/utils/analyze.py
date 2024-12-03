"""
Analysis utilities for PyHEA output, including Short Range Order (SRO) parameter calculations.
"""
import os
import dpdata
import numpy as np
import matplotlib.pyplot as plt
from ovito.io import import_file
import WarrenCowleyParameters as wc
from hea.utils import logger

def calculate_sro(filename, latt_type):
    """Calculate Short Range Order (SRO) parameters using WarrenCowleyParameters.
    
    @param filename str Path to LAMMPS data file
    @param latt_type str Lattice type
    @return tuple (ndarray, dict) SRO values matrix and atom type counts
    """

    if filename.endswith('lammps/lmp'):
        filename = filename.replace('lammps/lmp', 'lmp')
        pipeline = import_file(filename)
    elif filename.endswith('vasp/poscar'):
        filename = filename.replace('vasp/poscar', 'poscar')
        data = dpdata.System(filename, fmt='vasp/poscar')
        filename = filename.replace('poscar', 'lmp')
        data.to_lammps_lmp(filename)
        pipeline = import_file(filename)
        os.remove(filename)
    
    # Setup Warren-Cowley parameter calculation
    # First shell (12 neighbors for FCC), second shell (6 neighbors)
    mod = wc.WarrenCowleyParameters(nneigh=[0, 12, 18] if latt_type == 'FCC' else [0, 8, 14], only_selected=False)
    pipeline.modifiers.append(mod)
    
    # Compute Warren-Cowley parameters
    data = pipeline.compute()
    wc_parameters = data.attributes["Warren-Cowley parameters"]
    
    # Get first nearest neighbor parameters (index 0)
    sro_values = wc_parameters
    
    return sro_values

def plot_sro_heatmap(sro_values, atom_labels, output_file='sro_heatmap.png'):
    """Plot SRO parameters as a heatmap.
    
    @param sro_values ndarray Matrix of SRO values
    @param atom_labels list List of atom type labels
    @param output_file str Output file path for the plot
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(sro_values, cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar(label='Warren-Cowley Parameter')
    
    # Add labels
    plt.xticks(range(len(atom_labels)), atom_labels)
    plt.yticks(range(len(atom_labels)), atom_labels)
    plt.xlabel('Atom Type')
    plt.ylabel('Atom Type')
    plt.title('Warren-Cowley Parameters (1st Shell)')
    
    # Add value annotations
    for i in range(len(atom_labels)):
        for j in range(len(atom_labels)):
            plt.text(j, i, f'{sro_values[i,j]:.2f}', 
                    ha='center', va='center')
    
    plt.savefig(output_file)
    plt.close()

def analyze_sro_results(output_file, target_sro, element_types, latt_type):
    """Analyze SRO results and compare with target values.
    
    @param output_file str Path to LAMMPS output data file
    @param target_sro ndarray Target SRO values
    @param element_types list List of element types
    @param latt_type str Lattice type
    """
    TYPE_LIST = ["A", "B", "C", "D", "E", "F", "G", "H", "I"][: len(element_types)]
    # Calculate actual SRO values using Warren-Cowley parameters
    actual_sro = calculate_sro(output_file, latt_type)
    target_sro = target_sro.reshape(-1, len(element_types), len(element_types))[:len(actual_sro)]
    logger.info(f"Warren-Cowley Parameters calculated by the WarrenCowleyParameters repo: {actual_sro[0].tolist()}")
    logger.info(f"target_sro: {target_sro[0].tolist()}")

    # Plot actual SRO values
    plot_sro_heatmap(actual_sro[0], TYPE_LIST, 'actual_sro_heatmap.png')

    # Calculate and report differences
    sro_diff = actual_sro - target_sro
    logger.info("SRO Analysis Results:")
    logger.info("===============================================")
    logger.info("Type | Ref SRO    | Target SRO    | Difference:")
    for shell in range(1):
        for i in range(len(element_types)):
            for j in range(i + 1):
                logger.info(
                    f"{TYPE_LIST[i]}-{TYPE_LIST[j]}  | "
                    f"Ref={actual_sro[shell][i, j]:>6.3f} | "
                    f"Target={target_sro[shell][i, j]:>6.3f} | "
                    f"Diff={sro_diff[shell][i, j]:>6.3f}"
                )
    logger.info("===============================================")
    # Calculate overall error metrics
    mae = np.mean(np.abs(sro_diff))
    rmse = np.sqrt(np.mean(sro_diff**2))
    
    logger.info("Error Metrics:")
    logger.info(f"Mean Absolute Error: {mae:.3f}")
    logger.info(f"Root Mean Square Error: {rmse:.3f}")
    
    return actual_sro, mae, rmse
