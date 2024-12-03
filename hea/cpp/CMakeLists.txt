# Minimum CMake version requirement
cmake_minimum_required(VERSION 3.12)

# Project declaration
project(pyhea_accelerate CXX)

# Option for CUDA support
option(USE_CUDA "Enable CUDA support" OFF)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Find required packages
find_package(MPI REQUIRED)
set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Include directories
include_directories(${MPI_INCLUDE_PATH})

# Define source files
set(SOURCES
    bind.cpp
    accelerate.cpp
)

# Configuration based on CUDA availability
if(USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 86)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -rdc=true")
    list(APPEND SOURCES accelerate.cu)
    add_definitions(-DUSE_CUDA)
endif()

# Create pybind11 module
pybind11_add_module(accelerate ${SOURCES})

# Link libraries
target_link_libraries(accelerate PRIVATE 
    pybind11::module 
    MPI::MPI_CXX
)

# Optional CUDA linking
if(USE_CUDA)
    set_target_properties(accelerate PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# Installation settings
install(TARGETS accelerate
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)