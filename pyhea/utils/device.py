"""
Device utility functions for PyHEA.

This module provides functions to check and display information about available
computing devices (CPU and GPU) in the system.
"""

import os
import platform
import psutil
import subprocess
from typing import Dict, List, Optional, Tuple

from pyhea.utils.logger import logger
try:
    from pyhea.cpp import accelerate as acc
except ImportError:
    logger.warning("Failed to import accelerate module. GPU support may not be available.")
    acc = None

def get_cpu_info() -> Dict[str, str]:
    """Get detailed CPU information.
    
    Returns:
        Dict containing CPU information with the following keys:
        - processor: CPU model name
        - cores: Number of physical cores
        - threads: Number of logical cores (threads)
        - frequency: Current CPU frequency
        - memory: Total system memory
    """
    cpu_info = {}
    
    # Get CPU model name
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        cpu_info["processor"] = line.split(":")[1].strip()
                        break
        except Exception:
            cpu_info["processor"] = platform.processor() or "Unknown"
    else:
        cpu_info["processor"] = platform.processor() or "Unknown"
    
    # Get core count
    cpu_info["cores"] = psutil.cpu_count(logical=False)
    cpu_info["threads"] = psutil.cpu_count(logical=True)
    
    # Get CPU frequency
    freq = psutil.cpu_freq()
    if freq:
        cpu_info["frequency"] = f"{freq.current:.2f} MHz"
    else:
        cpu_info["frequency"] = "Unknown"
    
    # Get system memory
    memory = psutil.virtual_memory()
    cpu_info["memory"] = f"{memory.total / (1024**3):.1f} GB"
    
    return cpu_info

def get_gpu_info() -> List[Dict[str, str]]:
    """Get detailed GPU information using nvidia-smi.
    
    Returns:
        List of dictionaries containing GPU information with the following keys:
        - index: GPU index
        - name: GPU model name
        - memory: Total memory
        - driver: NVIDIA driver version
        Or empty list if no NVIDIA GPUs are found
    """
    gpu_info = []
    
    try:
        # Check if nvidia-smi is available
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", 
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse nvidia-smi output
        for line in result.stdout.strip().split("\n"):
            if line:
                index, name, memory, driver = line.split(", ")
                gpu_info.append({
                    "index": index,
                    "name": name,
                    "memory": f"{float(memory)/1024:.1f} GB",
                    "driver": driver
                })
                
    except (subprocess.CalledProcessError, FileNotFoundError):
        # nvidia-smi not available or failed
        pass
        
    return gpu_info

def check_device_info() -> Tuple[Dict, Dict]:
    """Check whether CPU and GPU computation is available.
    
    Returns:
        Tuple of (cpu_info: Dict, gpu_info: Dict)
    """
    # Check CPU availability (always True as CPU is required)
    cpu_available = True
    
    # Check GPU availability
    gpu_available = False
    gpu_info = get_gpu_info()
    if gpu_info:
        gpu_available = True
    
    return get_cpu_info(), get_gpu_info()

def check_device_availability() -> Tuple[bool, bool]:
    """Check whether CPU and GPU computation is available.
    
    Returns:
        Tuple of (cpu_available: bool, gpu_available: bool)
    """
    cpu_available = True
    gpu_available = False
    gpu_info = get_gpu_info()
    if gpu_info:
        gpu_available = True
    
    return cpu_available, gpu_available

def print_device_info(verbose: bool = True) -> None:
    """Print information about available computing devices.
    
    Args:
        verbose: If True, print detailed device information
    """
    cpu_available, gpu_available = check_device_availability()
    
    logger.info("PyHEA Device Information:")
    logger.info("=" * 50)
    
    # Print CPU information
    logger.info("CPU Information:")
    if cpu_available:
        cpu_info = get_cpu_info()
        if verbose:
            logger.info(f"  Processor: {cpu_info['processor']}")
            logger.info(f"  Physical cores: {cpu_info['cores']}")
            logger.info(f"  Logical cores: {cpu_info['threads']}")
            logger.info(f"  Current frequency: {cpu_info['frequency']}")
            logger.info(f"  System memory: {cpu_info['memory']}")
        else:
            logger.info(f"  {cpu_info['processor']} ({cpu_info['cores']} cores, {cpu_info['threads']} threads)")
    else:
        logger.warning("  No CPU detected (This should never happen)")
    
    # Print GPU information
    logger.info("\nGPU Information:")
    if gpu_available:
        gpu_info = get_gpu_info()
        if gpu_info:
            for gpu in gpu_info:
                if verbose:
                    logger.info(f"  GPU {gpu['index']}:")
                    logger.info(f"    Model: {gpu['name']}")
                    logger.info(f"    Memory: {gpu['memory']}")
                    logger.info(f"    Driver: {gpu['driver']}")
                else:
                    logger.info(f"  {gpu['name']} ({gpu['memory']})")
        else:
            logger.info("  CUDA support available but no NVIDIA GPUs detected")
    else:
        logger.warning("  No GPU support available")
    
    # Print recommended device
    logger.info("\nDevice Availability:")
    logger.info(f"  CPU Computing: {'Available' if cpu_available else 'Not Available'}")
    logger.info(f"  GPU Computing: {'Available' if gpu_available else 'Not Available'}")
    
    recommended = "gpu" if gpu_available else "cpu"
    logger.info(f"\nRecommended device for PyHEA: {recommended}")
    logger.info("=" * 50)
