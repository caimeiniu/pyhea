"""
Command line argument parsing for PyHEA.
"""

import argparse
from pyhea.version import __version__

def create_parser():
    """Create and return the argument parser for PyHEA.
    
    @return argparse.ArgumentParser The configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="PyHEA: High Entropy Alloys Model simulation and analysis."
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Add the main simulation command
    sim_parser = subparsers.add_parser('simulate', help='Run HEA simulation')
    sim_parser.add_argument('config_file', type=str, help='Path to the configuration YAML file.')
    
    # Add the analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze structure files')
    analyze_parser.add_argument('structure_file', help='Path to structure file to analyze')
    analyze_parser.add_argument('--format', choices=['lmp', 'poscar'], default='lmp',
                              help='Input file format (default: lmp)')
    analyze_parser.add_argument('--lattice-type', choices=['FCC', 'BCC'], default='FCC',
                              help='Lattice type for analysis (default: FCC)')
    analyze_parser.add_argument('--elements', nargs='+', help='List of element types in the structure')
    analyze_parser.add_argument('-o', '--output', help='Output filename for the heatmap')
    
    # Add version option to main parser
    parser.add_argument('--version', action='version', version=f'PyHEA {__version__}')
    
    return parser

def parse_args():
    """Parse command line arguments.
    
    @return argparse.Namespace The parsed command line arguments
    """
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        exit(1)
        
    return args
