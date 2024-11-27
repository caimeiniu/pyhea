"""
PyHEA (Python High Entropy Alloys) build configuration.

This module configures the build process for the PyHEA package, handling both
CPU and CUDA extensions through CMake. It automatically detects CUDA availability
and adjusts the build accordingly.

Author: Charmy Niu
"""

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import platform
import shutil
import os
import sys
from pathlib import Path

class CMakeExtension(Extension):
    """
    A custom Extension class for CMake-based extensions.

    This class represents a CMake-based Python extension, storing information about
    the extension's name and source directory.

    Attributes:
        name (str): The name of the extension
        source_dir (str): The absolute path to the extension's source directory
    """
    def __init__(self, name: str, source_dir: str = None):
        """
        Initialize the CMake extension.

        Args:
            name: The name of the extension
            source_dir: Optional path to the source directory. If None, name is used
        """
        Extension.__init__(self, name, sources=[])
        self.source_dir = os.path.abspath(source_dir if source_dir is not None else name)

def check_cuda_availability():
    """
    Check if CUDA is available on the system.

    Returns:
        bool: True if CUDA is available, False otherwise
    """
    if platform.system() == "Windows":
        nvcc_path = shutil.which("nvcc.exe")
    else:
        nvcc_path = shutil.which("nvcc")
    
    if nvcc_path is None:
        return False
    
    try:
        # Try to run nvcc to verify it works
        subprocess.run([nvcc_path, "--version"], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE)
        return True
    except subprocess.SubprocessError:
        return False

class CMakeBuild(build_ext):
    """
    A custom build_ext command for CMake-based extensions.

    This class handles the building of CMake extensions, supporting both CPU-only
    and CUDA-enabled builds depending on system capabilities.
    """
    
    def run(self):
        """Execute the build command for all extensions."""
        for ext in self.extensions:
            self.build_cmake(ext)

    def build_cmake(self, ext):
        """
        Build a CMake extension.

        Args:
            ext: The CMakeExtension instance to build

        Raises:
            subprocess.CalledProcessError: If CMake build fails
        """
        build_dir = Path(self.build_temp) / ext.name
        build_dir.mkdir(parents=True, exist_ok=True)
        
        extdir = Path(self.get_ext_fullpath(ext.name)).parent.absolute()
        
        # Get pybind11 CMake dir
        result = subprocess.run(['python3', '-m', 'pybind11', '--cmakedir'],
                              capture_output=True, text=True)
        cmake_prefix_path = result.stdout.strip()
        
        # Basic CMake arguments
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-DCMAKE_PREFIX_PATH={cmake_prefix_path}',
            f'-DCMAKE_BUILD_TYPE=Release'
        ]
        
        # Add CUDA flag based on availability
        has_cuda = check_cuda_availability()
        cmake_args.append(f'-DUSE_CUDA={str(has_cuda).upper()}')
        
        if has_cuda:
            print("CUDA detected - building with CUDA support")
        else:
            print("CUDA not detected - building CPU-only version")
        
        build_args = ['--config', 'Release']

        subprocess.check_call(['cmake', str(ext.source_dir)] + cmake_args, 
                            cwd=str(build_dir))
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                            cwd=str(build_dir))

setup(
    name="pyhea",
    version="1.0.0",
    author="Charmy Niu",
    description="A High performance implementation for building the High Entropy Alloys Model",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.7.0", 
        "pyyaml>=5.1",
        "mpi4py>=3.0.0",
        "pybind11>=2.6.0",
    ],
    entry_points={
        'console_scripts': [
            'hea=hea.main:main',
        ],
    },
    ext_modules=[
        CMakeExtension("hea.cpp.accelerate", "hea/cpp")
    ],
    cmdclass={
        "build_ext": CMakeBuild,
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Programming Language :: CUDA",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
)