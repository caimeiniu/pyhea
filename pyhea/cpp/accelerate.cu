#include "accelerate.h"

#include <ctime>
#include <chrono>
#include <iomanip>
#include <sys/time.h>
#include <cstdint>

#include <numeric>
#include <iostream>
#include <algorithm>

#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <thrust/reduce.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>

#include <thrust/shuffle.h>
#include <thrust/random.h>
#include <thrust/execution_policy.h>

namespace accelerate {
namespace gpu {

/**
 * @brief Checks if CUDA is available on the system.
 * 
 * @return true if at least one CUDA device is available, false otherwise.
 */
bool cuda_available() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    return device_count > 0;
}

#define THREADS_PER_BLOCK 256

/**
 * @brief Gets the current time in microseconds.
 * 
 * @return unsigned long long Current time in microseconds since epoch.
 */
unsigned long long get_microseconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}

namespace bit_utils {
    template <typename Integer> 
    __device__ __host__ __forceinline__ Integer pack_4types(Integer a, Integer b, Integer c, Integer d) {
        return (a & 0x3) | ((b & 0x3) << 2) | ((c & 0x3) << 4) | ((d & 0x3) << 6);
    }

    template <typename Integer>
    __device__ __host__ __forceinline__ Integer get_type(Integer packed, int index) {
        return (packed >> (index * 2)) & 0x3;
    }

    template <typename Integer>
    __device__ __host__ __forceinline__ void set_type(Integer& packed, int index, Integer value) {
        Integer mask = ~(0x3 << (index * 2));
        packed = (packed & mask) | ((value & 0x3) << (index * 2));
    }

    template <typename Integer>
    __device__ __host__ __forceinline__ Integer pack_type(Integer type) {
        return type & 0x3;
    }
}

/**
 * @brief Calculates fitness values for multiple lattices based on Short-Range Order (SRO) coefficients.
 * 
 * @tparam Integer Integer type used for indices and counts.
 * @tparam Real Floating-point type used for calculations.
 * @param[in] num_atoms Number of atoms in each lattice.
 * @param[in] num_types Number of different atom types.
 * @param[in] num_shells Number of neighbor shells to consider.
 * @param[in] neighbor_list List of neighbor indices for each atom.
 * @param[in] neighbor_list_indices Starting indices in neighbor_list for each atom.
 * @param[in] species Array containing count of each atom type.
 * @param[in] weights Weight factors for each shell.
 * @param[in] target_sro Target SRO values to compare against.
 * @param[out] coefficients Computed SRO coefficients.
 * @param[out] fitness Computed fitness values for each lattice.
 * @param[in] lattices Array of lattice configurations.
 */
template <typename Integer, typename Real>
__global__ void calculate_fitness_of_lattices(
        Integer num_atoms, Integer num_types, Integer num_shells,
        const Integer* neighbor_list, const Integer* neighbor_list_indices, 
        const Integer* species, const Real* weights, 
        const Real* target_sro, Real* coefficients, 
        Real* fitness, Integer* lattices)
{
    const Integer bid = blockIdx.x;
    const Integer tid = threadIdx.x;
    const Integer num_coefficients = num_types * num_types * num_shells;

    // Each block addresses a different lattice
    const Integer* lattice = lattices + bid * num_atoms;

    extern __shared__ Real shared_sro_coefficients[];

    for (Integer ii = tid; ii < num_coefficients; ii += THREADS_PER_BLOCK) {
        shared_sro_coefficients[ii] = 0.0;
    }
    __syncthreads();
    Real* shared_gamma = shared_sro_coefficients;
    
    for (Integer ii = tid; ii < num_atoms; ii += THREADS_PER_BLOCK) {
        Integer atom = lattice[ii];
        for (Integer jj = 0; jj < num_shells; ++jj) {
            Integer neighbor_size = neighbor_list_indices[ii * num_shells + jj + 1] - neighbor_list_indices[ii * num_shells + jj];
            for (Integer kk = 0; kk < neighbor_size; ++kk) {
                // Ensure the access is within bounds
                Integer neighbor = lattice[neighbor_list[neighbor_list_indices[ii * num_shells + jj] + kk]];
                // Check bounds for pair_gr, avg_gr, count_gr
                Integer indices = atom * num_types * num_shells + neighbor * num_shells + jj;
                atomicAdd(&shared_gamma[indices], 1.0 / (neighbor_size * num_atoms));
            }
        }
    }
    __syncthreads();

    for (Integer ii = tid; ii < num_coefficients; ii += THREADS_PER_BLOCK) {
        coefficients[bid * num_coefficients + ii] = shared_sro_coefficients[ii];
    }

    if (tid != 0) return;

    Real error = 0.0;
    for (Integer shell = 0; shell < num_shells; ++shell) {
        Real fitness_shell = 0.0;
        for (Integer ii = 0; ii < num_types; ++ii) {
            for (Integer jj = 0; jj < num_types; ++jj) {
                Integer index = ii * num_types * num_shells + jj * num_shells + shell;
                Real sro = 1 - shared_gamma[index] / ((static_cast<Real>(species[ii]) / static_cast<Real>(num_atoms)) * (static_cast<Real>(species[jj]) / static_cast<Real>(num_atoms)));
                Real shell_sro = target_sro[shell * num_types * num_types + ii * num_types + jj];
                fitness_shell += (sro - shell_sro) * (sro - shell_sro);
            }
        }
        // Weighted sum of error across shells
        error += weights[shell] * fitness_shell;
    }
    // Return the square root of the total error
    fitness[bid] = sqrt(error);
}

/**
 * @brief Initializes CURAND states for random number generation.
 * 
 * @param[out] rng_states Array of CURAND states to initialize.
 * @param[in] seed Random seed value.
 */
__global__ void init_curand(curandState* rng_states, unsigned long long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, id, 0, &rng_states[id]);
}

/**
 * @brief Generates initial lattice configurations based on species distribution.
 * 
 * @tparam Integer Integer type used for indices and counts.
 * @param[in] num_atoms Total number of atoms per lattice.
 * @param[in] num_types Number of different atom types.
 * @param[in] species Array containing count of each atom type.
 * @param[out] lattices Output array for generated lattices.
 */
template <typename Integer>
__global__ void generate_normal_lattices(const Integer num_atoms, const Integer num_types, const Integer* species, Integer* lattices) 
{
    const Integer bid = blockIdx.x;
    const Integer tid = threadIdx.x;
    
    // Pointer to the start of the lattice for the current block
    Integer* lattice = lattices + bid * num_atoms;

    for (Integer ii = tid; ii < num_atoms; ii += THREADS_PER_BLOCK) {
        Integer sum = 0;
        for (Integer type = 0; type < num_types; ++type) {
            sum += species[type];
            if (ii < sum) {
                lattice[ii] = type;
                break;
            }
        }
    }
}

/**
 * @brief Locates the index of the best fitness value in a block.
 * 
 * @tparam Integer Integer type used for indices.
 * @tparam Real Floating-point type used for fitness values.
 * @param[in] fitness Array of fitness values.
 * @param[in,out] indices Array of indices.
 * @return int Index of the best fitness value.
 */
template <typename Integer, typename Real>
__device__ __forceinline__ int locate_best_fitness(Real* fitness, Integer* indices) {
    const Integer tid = threadIdx.x;
    const Integer num_threads = blockDim.x;
    // Shared memory for indices
    indices[tid] = tid; // Initialize indices
    __syncthreads();
    // Reduction to find the minimum value and its index
    for (int ii = num_threads / 2; ii > 0; ii /= 2) {
        if (tid < ii) {
            if (fitness[tid] > fitness[tid + ii]) {
                fitness[tid] = fitness[tid + ii];
                indices[tid] = indices[tid + ii];
            }
        }
        __syncthreads();
    }
    // Return the index of the minimum value
    return indices[0]; // Index of the minimum value in shared memory
}

/**
 * @brief Calculates SRO coefficients for a given lattice configuration.
 * 
 * @tparam Integer Integer type for indices.
 * @tparam Real Floating-point type for calculations.
 * @tparam LOW_Integer Lower precision integer type for lattice values.
 * @tparam LOW_Real Lower precision floating-point type for intermediate calculations.
 * @tparam NUM_TYPES Number of atom types (compile-time constant).
 * @tparam NUM_SHELLS Number of neighbor shells (compile-time constant).
 * @param[in] num_atoms Number of atoms in the lattice.
 * @param[in] species Array containing count of each atom type.
 * @param[in] weights Weight factors for each shell.
 * @param[in] neighbor_list List of neighbor indices.
 * @param[in] neighbor_list_indices Starting indices in neighbor_list.
 * @param[out] coefficients Output array for computed coefficients.
 * @param[in] lattice Current lattice configuration.
 */
template <typename Integer, typename Real,
          typename LOW_Integer, typename LOW_Real,
          int NUM_TYPES, int NUM_SHELLS>
__device__ __forceinline__ void calculate_coefficients(
    const Integer& num_atoms,
    const Integer* species, const Real* weights,
    const Integer* neighbor_list, const Integer* neighbor_list_indices,
    Real* coefficients,
    LOW_Integer* lattice)
{
    __syncthreads();
    // Define the number of coefficients
    const Integer NUM_COEFFICIENTS = NUM_TYPES * NUM_TYPES * NUM_SHELLS;
    // Split coefficients into three parts: alpha (FP16), gamma (FP16), count (FP16)
    Real* gamma = coefficients;  // FP16 type for alpha

    // Loop over each atom
    for (Integer ii = threadIdx.x; ii < num_atoms; ii += blockDim.x) {
        LOW_Integer atom = lattice[ii];
        // Loop over each shell
        for (Integer shell = 0; shell < NUM_SHELLS; ++shell) {
            Integer neighbor_size = neighbor_list_indices[ii * NUM_SHELLS + shell + 1] 
                                  - neighbor_list_indices[ii * NUM_SHELLS + shell];

            for (Integer kk = 0; kk < neighbor_size; ++kk) {
                Integer idx = neighbor_list[neighbor_list_indices[ii * NUM_SHELLS + shell] + kk];
                LOW_Integer neighbor = lattice[idx];

                // Compute the index for accessing the arrays
                Integer index = atom * NUM_TYPES * NUM_SHELLS + neighbor * NUM_SHELLS + shell;
                // gamma[index] += 1.0 / neighbor_size;
                atomicAdd(&gamma[index], 1.0 / (neighbor_size * num_atoms));
            }
        }
    }
    __syncthreads();
}

template <typename Integer, typename Real,
          typename LOW_Integer, typename LOW_Real,
          int NUM_TYPES, int NUM_SHELLS>
__device__ __forceinline__ void calculate_coefficients_bit_packed(
    const Integer& num_atoms,
    const Integer* species, const Real* weights,
    const Integer* neighbor_list, const Integer* neighbor_list_indices,
    Real* coefficients,
    const uint8_t* lattice)
{
    __syncthreads();
    const Integer NUM_COEFFICIENTS = NUM_TYPES * NUM_TYPES * NUM_SHELLS;
    Real* gamma = coefficients;

    for (Integer ii = threadIdx.x; ii < num_atoms; ii += blockDim.x) {
        LOW_Integer atom = bit_utils::get_type(lattice[ii / 4], ii % 4);

        for (Integer shell = 0; shell < NUM_SHELLS; ++shell) {
            Integer neighbor_size = neighbor_list_indices[ii * NUM_SHELLS + shell + 1] 
                                  - neighbor_list_indices[ii * NUM_SHELLS + shell];

            for (Integer kk = 0; kk < neighbor_size; ++kk) {
                Integer idx = neighbor_list[neighbor_list_indices[ii * NUM_SHELLS + shell] + kk];
                LOW_Integer neighbor = bit_utils::get_type(lattice[idx / 4], idx % 4);

                Integer index = atom * NUM_TYPES * NUM_SHELLS + neighbor * NUM_SHELLS + shell;
                atomicAdd(&gamma[index], 1.0 / (neighbor_size * num_atoms));
            }
        }
    }
    __syncthreads();
}

/**
 * @brief Calculates fitness for a proposed lattice configuration with two atoms swapped.
 * 
 * @tparam Integer Integer type for indices.
 * @tparam Real Floating-point type for calculations.
 * @tparam LOW_Integer Lower precision integer type for lattice values.
 * @tparam LOW_Real Lower precision floating-point type for intermediate calculations.
 * @tparam NUM_TYPES Number of atom types (compile-time constant).
 * @tparam NUM_SHELLS Number of neighbor shells (compile-time constant).
 * @param[in] atom1 Index of first atom to swap.
 * @param[in] atom2 Index of second atom to swap.
 * @param[in] num_atoms Total number of atoms.
 * @param[in] species Array containing count of each atom type.
 * @param[in] weights Weight factors for each shell.
 * @param[in] neighbor_list List of neighbor indices.
 * @param[in] neighbor_list_indices Starting indices in neighbor_list.
 * @param[in] target_sro Target SRO values to compare against.
 * @param[out] coefficients Temporary array for coefficient calculations.
 * @param[in,out] lattice Current lattice configuration.
 * @return Real Computed fitness value for the proposed configuration.
 */
template <typename Integer, typename Real,
          typename LOW_Integer, typename LOW_Real,
          int NUM_TYPES, int NUM_SHELLS>
__device__ __forceinline__ Real calculate_fitness(
    const Integer& atom1, const Integer& atom2,
    const Integer& num_atoms,
    const Integer* species, const Real* weights,
    const Integer* neighbor_list, const Integer* neighbor_list_indices,
    const Real* target_sro, Real* coefficients,
    LOW_Integer* lattice)
{
    // Define the number of coefficients
    const Integer NUM_COEFFICIENTS = NUM_TYPES * NUM_TYPES * NUM_SHELLS;

    // Split coefficients into three parts: alpha (FP16), gamma (FP16), count (FP16)
    Real* gamma = coefficients;  // FP16 type for alpha

    // Loop over each atom
    for (Integer ii = 0; ii < num_atoms; ++ii) {
        LOW_Integer atom = lattice[ii];
        if (ii == atom1) {
            atom = lattice[atom2];
        } else if (ii == atom2) {
            atom = lattice[atom1];
        }
        // Loop over each shell
        for (Integer shell = 0; shell < NUM_SHELLS; ++shell) {
            Integer neighbor_size = neighbor_list_indices[ii * NUM_SHELLS + shell + 1] 
                                  - neighbor_list_indices[ii * NUM_SHELLS + shell];

            for (Integer kk = 0; kk < neighbor_size; ++kk) {
                Integer idx = neighbor_list[neighbor_list_indices[ii * NUM_SHELLS + shell] + kk];
                LOW_Integer neighbor = lattice[idx];
                
                if (idx == atom1) {
                    neighbor = lattice[atom2];
                } else if (idx == atom2) {
                    neighbor = lattice[atom1];
                }

                // Compute the index for accessing the arrays
                Integer index = atom * NUM_TYPES * NUM_SHELLS + neighbor * NUM_SHELLS + shell;
                // Update gamma (FP16), increment by 1
                gamma[index] += 1.0 / (neighbor_size * num_atoms);
            }
        }
    }

    // Fitness calculation
    Real fitness = 0.0;
    for (Integer shell = 0; shell < NUM_SHELLS; ++shell) {
        Real fitness_shell = 0.0;
        for (Integer ii = 0; ii < NUM_TYPES; ++ii) {
            for (Integer jj = 0; jj < NUM_TYPES; ++jj) {
                Integer index = ii * NUM_TYPES * NUM_SHELLS + jj * NUM_SHELLS + shell;
                Real sro = 1 - gamma[index] / ((static_cast<Real>(species[ii]) / static_cast<Real>(num_atoms)) * (static_cast<Real>(species[jj]) / static_cast<Real>(num_atoms)));
                Real shell_sro = target_sro[shell * NUM_TYPES * NUM_TYPES + ii * NUM_TYPES + jj];
                fitness_shell += (sro - shell_sro) * (sro - shell_sro);
            }
        }
        // Weighted sum of fitness across shells
        fitness += weights[shell] * fitness_shell;
    }
    // Return the square root of the total fitness
    return sqrt(fitness);
}

/**
 * @brief Calculates fitness for a proposed lattice configuration with two atoms swapped.
 * 
 * @tparam Integer Integer type for indices.
 * @tparam Real Floating-point type for calculations.
 * @tparam LOW_Integer Lower precision integer type for lattice values.
 * @tparam LOW_Real Lower precision floating-point type for intermediate calculations.
 * @tparam NUM_TYPES Number of atom types (compile-time constant).
 * @tparam NUM_SHELLS Number of neighbor shells (compile-time constant).
 * @param[in] atom1 Index of first atom to swap.
 * @param[in] atom2 Index of second atom to swap.
 * @param[in] num_atoms Total number of atoms.
 * @param[in] species Array containing count of each atom type.
 * @param[in] weights Weight factors for each shell.
 * @param[in] neighbor_list List of neighbor indices.
 * @param[in] neighbor_list_indices Starting indices in neighbor_list.
 * @param[in] target_sro Target SRO values to compare against.
 * @param[out] coefficients Temporary array for coefficient calculations.
 * @param[in,out] lattice Current lattice configuration.
 * @return Real Computed fitness value for the proposed configuration.
 */
template <typename Integer, typename Real,
          typename LOW_Integer, typename LOW_Real,
          int NUM_TYPES, int NUM_SHELLS>
__device__ __forceinline__ Real calculate_fitness_incremental(
    const Integer& atom1, const Integer& atom2,
    const Integer& num_atoms,
    const Integer* species, const Real* weights,
    const Integer* neighbor_list, const Integer* neighbor_list_indices,
    const Real* target_sro, Real* coefficients,
    LOW_Integer* lattice)
{
    // Define the number of coefficients
    const Integer NUM_COEFFICIENTS = NUM_TYPES * NUM_TYPES * NUM_SHELLS;

    // Split coefficients into three parts: alpha (FP16), gamma (FP16), count (FP16)
    Real* gamma = coefficients;  // FP16 type for alpha

    for (Integer shell = 0; shell < NUM_SHELLS; ++shell) {
        LOW_Integer atom = lattice[atom1];
        LOW_Integer atom_after_swap = lattice[atom2];
        Integer neighbor_size = neighbor_list_indices[atom1 * NUM_SHELLS + shell + 1] 
                              - neighbor_list_indices[atom1 * NUM_SHELLS + shell];
        for (Integer kk = 0; kk < neighbor_size; ++kk) {
            Integer idx = neighbor_list[neighbor_list_indices[atom1 * NUM_SHELLS + shell] + kk];
            LOW_Integer neighbor = lattice[idx];
            LOW_Integer neighbor_after_swap = neighbor;
            if (idx == atom2) {
                neighbor_after_swap = lattice[atom1];
            } else if (idx == atom1) {
                neighbor_after_swap = lattice[atom2];
            }

            // Compute the index for accessing the arrays
            Integer __ij = atom * NUM_TYPES * NUM_SHELLS + neighbor * NUM_SHELLS + shell;
            Integer __ji = neighbor * NUM_TYPES * NUM_SHELLS + atom * NUM_SHELLS + shell;

            Integer __ij_after_swap = atom_after_swap * NUM_TYPES * NUM_SHELLS + neighbor_after_swap * NUM_SHELLS + shell;
            Integer __ji_after_swap = neighbor_after_swap * NUM_TYPES * NUM_SHELLS + atom_after_swap * NUM_SHELLS + shell;

            // Update gamma (FP16), increment by 1
            gamma[__ij] -= 1.0 / (neighbor_size * num_atoms);
            gamma[__ji] -= 1.0 / (neighbor_size * num_atoms);
            gamma[__ij_after_swap] += 1.0 / (neighbor_size * num_atoms);
            gamma[__ji_after_swap] += 1.0 / (neighbor_size * num_atoms);
        }

        atom = lattice[atom2];
        atom_after_swap = lattice[atom1];
        neighbor_size = neighbor_list_indices[atom2 * NUM_SHELLS + shell + 1] 
                      - neighbor_list_indices[atom2 * NUM_SHELLS + shell];
        for (Integer kk = 0; kk < neighbor_size; ++kk) {
            Integer idx = neighbor_list[neighbor_list_indices[atom2 * NUM_SHELLS + shell] + kk];
            LOW_Integer neighbor = lattice[idx];
            LOW_Integer neighbor_after_swap = neighbor;
            if (idx == atom1) {
                neighbor_after_swap = lattice[atom2];
            } else if (idx == atom2) {
                neighbor_after_swap = lattice[atom1];
            }

            // Compute the index for accessing the arrays
            Integer __ij = atom * NUM_TYPES * NUM_SHELLS + neighbor * NUM_SHELLS + shell;
            Integer __ji = neighbor * NUM_TYPES * NUM_SHELLS + atom * NUM_SHELLS + shell;

            Integer __ij_after_swap = atom_after_swap * NUM_TYPES * NUM_SHELLS + neighbor_after_swap * NUM_SHELLS + shell;
            Integer __ji_after_swap = neighbor_after_swap * NUM_TYPES * NUM_SHELLS + atom_after_swap * NUM_SHELLS + shell;

            // Update gamma (FP16), increment by 1
            gamma[__ij] -= 1.0 / (neighbor_size * num_atoms);
            gamma[__ji] -= 1.0 / (neighbor_size * num_atoms);
            gamma[__ij_after_swap] += 1.0 / (neighbor_size * num_atoms);
            gamma[__ji_after_swap] += 1.0 / (neighbor_size * num_atoms);
        }
    }

    // Fitness calculation
    Real fitness = 0.0;
    for (Integer shell = 0; shell < NUM_SHELLS; ++shell) {
        Real fitness_shell = 0.0;
        for (Integer ii = 0; ii < NUM_TYPES; ++ii) {
            for (Integer jj = 0; jj < NUM_TYPES; ++jj) {
                Integer index = ii * NUM_TYPES * NUM_SHELLS + jj * NUM_SHELLS + shell;
                Real sro = 1 - gamma[index] / ((static_cast<Real>(species[ii]) / static_cast<Real>(num_atoms)) * (static_cast<Real>(species[jj]) / static_cast<Real>(num_atoms)));
                Real shell_sro = target_sro[shell * NUM_TYPES * NUM_TYPES + ii * NUM_TYPES + jj];
                fitness_shell += (sro - shell_sro) * (sro - shell_sro);
            }
        }
        // Weighted sum of fitness across shells
        fitness += weights[shell] * fitness_shell;
    }
    // Return the square root of the total fitness
    return sqrt(fitness);
}

template <typename Integer, typename Real,
          typename LOW_Integer, typename LOW_Real,
          int NUM_TYPES, int NUM_SHELLS>
__device__ __forceinline__ Real calculate_fitness_incremental_bit_packed(
    const Integer& atom1, const Integer& atom2,
    const Integer& num_atoms,
    const Integer* species, const Real* weights,
    const Integer* neighbor_list, const Integer* neighbor_list_indices,
    const Real* target_sro,
    Real* coefficients,
    uint8_t* lattice)
{
    const Integer NUM_COEFFICIENTS = NUM_TYPES * NUM_TYPES * NUM_SHELLS;
    Real* gamma = coefficients;

    for (Integer shell = 0; shell < NUM_SHELLS; ++shell) {
        LOW_Integer atom = bit_utils::get_type(lattice[atom1 / 4], atom1 % 4);
        LOW_Integer atom_after_swap = bit_utils::get_type(lattice[atom2 / 4], atom2 % 4);
        
        Integer neighbor_size = neighbor_list_indices[atom1 * NUM_SHELLS + shell + 1] 
                              - neighbor_list_indices[atom1 * NUM_SHELLS + shell];
        
        for (Integer kk = 0; kk < neighbor_size; ++kk) {
            Integer idx = neighbor_list[neighbor_list_indices[atom1 * NUM_SHELLS + shell] + kk];
            LOW_Integer neighbor = bit_utils::get_type(lattice[idx / 4], idx % 4);
            LOW_Integer neighbor_after_swap = neighbor;
            
            if (idx == atom2) {
                neighbor_after_swap = bit_utils::get_type(lattice[atom1 / 4], atom1 % 4);
            } else if (idx == atom1) {
                neighbor_after_swap = bit_utils::get_type(lattice[atom2 / 4], atom2 % 4);
            }

            Integer __ij = atom * NUM_TYPES * NUM_SHELLS + neighbor * NUM_SHELLS + shell;
            Integer __ji = neighbor * NUM_TYPES * NUM_SHELLS + atom * NUM_SHELLS + shell;
            Integer __ij_after_swap = atom_after_swap * NUM_TYPES * NUM_SHELLS + neighbor_after_swap * NUM_SHELLS + shell;
            Integer __ji_after_swap = neighbor_after_swap * NUM_TYPES * NUM_SHELLS + atom_after_swap * NUM_SHELLS + shell;

            gamma[__ij] -= 1.0 / (neighbor_size * num_atoms);
            gamma[__ji] -= 1.0 / (neighbor_size * num_atoms);
            gamma[__ij_after_swap] += 1.0 / (neighbor_size * num_atoms);
            gamma[__ji_after_swap] += 1.0 / (neighbor_size * num_atoms);
        }

        atom = bit_utils::get_type(lattice[atom2 / 4], atom2 % 4);
        atom_after_swap = bit_utils::get_type(lattice[atom1 / 4], atom1 % 4);
        neighbor_size = neighbor_list_indices[atom2 * NUM_SHELLS + shell + 1] 
                      - neighbor_list_indices[atom2 * NUM_SHELLS + shell];
                      
        for (Integer kk = 0; kk < neighbor_size; ++kk) {
            Integer idx = neighbor_list[neighbor_list_indices[atom2 * NUM_SHELLS + shell] + kk];
            LOW_Integer neighbor = bit_utils::get_type(lattice[idx / 4], idx % 4);
            LOW_Integer neighbor_after_swap = neighbor;
            
            if (idx == atom1) {
                neighbor_after_swap = bit_utils::get_type(lattice[atom2 / 4], atom2 % 4);
            } else if (idx == atom2) {
                neighbor_after_swap = bit_utils::get_type(lattice[atom1 / 4], atom1 % 4);
            }

            Integer __ij = atom * NUM_TYPES * NUM_SHELLS + neighbor * NUM_SHELLS + shell;
            Integer __ji = neighbor * NUM_TYPES * NUM_SHELLS + atom * NUM_SHELLS + shell;
            Integer __ij_after_swap = atom_after_swap * NUM_TYPES * NUM_SHELLS + neighbor_after_swap * NUM_SHELLS + shell;
            Integer __ji_after_swap = neighbor_after_swap * NUM_TYPES * NUM_SHELLS + atom_after_swap * NUM_SHELLS + shell;

            gamma[__ij] -= 1.0 / (neighbor_size * num_atoms);
            gamma[__ji] -= 1.0 / (neighbor_size * num_atoms);
            gamma[__ij_after_swap] += 1.0 / (neighbor_size * num_atoms);
            gamma[__ji_after_swap] += 1.0 / (neighbor_size * num_atoms);
        }
    }

    Real fitness = 0.0;
    for (Integer shell = 0; shell < NUM_SHELLS; ++shell) {
        Real fitness_shell = 0.0;
        for (Integer ii = 0; ii < NUM_TYPES; ++ii) {
            for (Integer jj = 0; jj < NUM_TYPES; ++jj) {
                Integer index = ii * NUM_TYPES * NUM_SHELLS + jj * NUM_SHELLS + shell;
                Real sro = 1 - gamma[index] / ((static_cast<Real>(species[ii]) / static_cast<Real>(num_atoms)) * 
                                             (static_cast<Real>(species[jj]) / static_cast<Real>(num_atoms)));
                Real shell_sro = target_sro[shell * NUM_TYPES * NUM_TYPES + ii * NUM_TYPES + jj];
                fitness_shell += (sro - shell_sro) * (sro - shell_sro);
            }
        }
        fitness += weights[shell] * fitness_shell;
    }
    return sqrt(fitness);
}

/**
 * @brief Performs parallel Monte Carlo optimization of lattice configurations.
 * 
 * @tparam Integer Integer type for indices.
 * @tparam Real Floating-point type for calculations.
 * @tparam LOW_Integer Lower precision integer type for lattice values.
 * @tparam LOW_Real Lower precision floating-point type for intermediate calculations.
 * @tparam NUM_TYPES Number of atom types (compile-time constant).
 * @tparam NUM_SHELLS Number of neighbor shells (compile-time constant).
 * @param[in] threshold Acceptance threshold for Monte Carlo moves.
 * @param[in] search_depth Number of Monte Carlo steps to perform.
 * @param[in] seed Random seed value.
 * @param[in] num_atoms Number of atoms per lattice.
 * @param[in] species Array containing count of each atom type.
 * @param[in] weights Weight factors for each shell.
 * @param[in] neighbor_list List of neighbor indices.
 * @param[in] neighbor_list_indices Starting indices in neighbor_list.
 * @param[in] target_sro Target SRO values to compare against.
 * @param[out] fitness Array for computed fitness values.
 * @param[in,out] lattices Array of lattice configurations.
 */
template <typename Integer, typename Real,
          typename LOW_Integer, typename LOW_Real,
          int NUM_TYPES, int NUM_SHELLS>
__global__ void parallel_monte_carlo(
    const Real threshold, const Integer search_depth, const unsigned long long random_seed,
    const Integer num_atoms,
    const Integer* species, const Real* weights,
    const Integer* neighbor_list, const Integer* neighbor_list_indices,
    const Real* target_sro,
    Real* fitness, Integer* lattices)
{
    extern __shared__ Real shared[];

    const Integer bid = blockIdx.x;
    const Integer tid = threadIdx.x;

    const Integer num_blocks = gridDim.x;
    const Integer num_threads = blockDim.x;
    const Integer NUM_COEFFICIENTS = NUM_SHELLS * NUM_TYPES * NUM_TYPES;

    curandState local_state;
    curand_init(random_seed, tid, 0, &local_state);

    Integer*     shared_depth     = reinterpret_cast<Integer*>(shared);
    Integer*     shared_indices   = shared_depth   + num_threads;
    LOW_Integer* shared_lattice   = reinterpret_cast<LOW_Integer*>(shared_indices + num_threads); // LOW_Integer type for shared lattice
    Real* shared_fitness          = reinterpret_cast<Real*>(shared_lattice + num_atoms); // Adjust for LOW_Integer
    Real* shared_coefficients     = shared_fitness + num_threads;

    shared_fitness[tid] = 100;

    // Init fitness
    if (tid == 0) {
        shared_depth[0] = 0;
    }
    Real reg_fitness = fitness[bid];

    // Load lattice from global memory (int) to shared memory (LOW_Integer)
    for (Integer ii = tid; ii < num_atoms; ii += num_threads) {
        shared_lattice[ii] = static_cast<LOW_Integer>(lattices[bid * num_atoms + ii]);  // Cast int to LOW_Integer
    }
    for (Integer ii = tid; ii < NUM_COEFFICIENTS; ii += num_threads) {
        shared_coefficients[ii] = 0.0;
    }

    calculate_coefficients<Integer, Real, LOW_Integer, LOW_Real, NUM_TYPES, NUM_SHELLS>(
        num_atoms, species, weights, neighbor_list, neighbor_list_indices, shared_coefficients, shared_lattice);

    Real coefficients[NUM_COEFFICIENTS];
    Integer reg_depth = shared_depth[0];

    while (reg_depth < search_depth) {
        // Initialize coefficients
        for (Integer ii = 0; ii < NUM_COEFFICIENTS; ii++) {
            coefficients[ii] = shared_coefficients[ii];
        }

        Integer atom1 = curand(&local_state) % num_atoms;
        Integer atom2 = curand(&local_state) % num_atoms;

        // Ensure atom1 and atom2 are different
        while (atom1 == atom2 || shared_lattice[atom1] == shared_lattice[atom2]) {
            atom1 = curand(&local_state) % num_atoms;
            atom2 = curand(&local_state) % num_atoms;
        }

        // Calculate fitness incrementally
        shared_fitness[tid] = calculate_fitness_incremental<Integer, Real, LOW_Integer, LOW_Real, NUM_TYPES, NUM_SHELLS>(
            atom1, atom2, num_atoms, species, weights, neighbor_list, neighbor_list_indices, target_sro, coefficients, shared_lattice);
        // shared_fitness[tid] = calculate_fitness<Integer, Real, LOW_Integer, LOW_Real, NUM_TYPES, NUM_SHELLS>(
            // atom1, atom2, num_atoms, species, weights, neighbor_list, neighbor_list_indices, target_sro, coefficients, shared_lattice);

        Integer id = locate_best_fitness(shared_fitness, shared_indices);

        if ((shared_fitness[id] < reg_fitness)) {
            reg_fitness = shared_fitness[id];
            if (tid == id) {
                // Swap atom1 and atom2 in the shared lattice (LOW_Integer)
                LOW_Integer swap = shared_lattice[atom1];
                shared_lattice[atom1] = shared_lattice[atom2];
                shared_lattice[atom2] = swap;
                for (Integer ii = 0; ii < NUM_COEFFICIENTS; ii++) {
                    shared_coefficients[ii] = coefficients[ii];
                }
            }
            if (tid == 0) {
                shared_depth[0] = 0;
            }
        } else {
            if (tid == 0) {
                shared_depth[0] += 1;
            }
        }
        __syncthreads();
        reg_depth = shared_depth[0];
    }
    __syncthreads();

    if (tid == 0) {
        fitness[bid] = reg_fitness;
    }
    // Write the final lattice from shared memory (LOW_Integer) back to global memory (int)
    for (Integer ii = tid; ii < num_atoms; ii += num_threads) {
        lattices[bid * num_atoms + ii] = static_cast<Integer>(shared_lattice[ii]);  // Cast LOW_Integer back to int
    }
}


template <typename Integer, typename Real,
          typename LOW_Integer, typename LOW_Real,
          int NUM_TYPES, int NUM_SHELLS>
__global__ void parallel_monte_carlo_bit_packed(
    const Real threshold, const Integer search_depth,
    const unsigned long long seed, const Integer num_atoms,
    const Integer* species, const Real* weights,
    const Integer* neighbor_list, const Integer* neighbor_list_indices,
    const Real* target_sro,
    Real* fitness, Integer* lattices)
{
    static_assert(NUM_TYPES <= 4, "Bit-packed version only supports up to 4 types");
    
    extern __shared__ Real shared[];
    
    const Integer bid = blockIdx.x;
    const Integer tid = threadIdx.x;

    const Integer num_blocks = gridDim.x;
    const Integer num_threads = blockDim.x;
    const Integer NUM_COEFFICIENTS = NUM_SHELLS * NUM_TYPES * NUM_TYPES;

    // Initialize random state
    curandState local_state;
    curand_init(seed + bid * blockDim.x + tid, 0, 0, &local_state);
    
    // Shared memory layout for bit-packed version
    Integer*     shared_depth     = reinterpret_cast<Integer*>(shared);
    Integer*     shared_indices   = shared_depth   + num_threads;
    LOW_Integer* shared_lattice   = reinterpret_cast<LOW_Integer*>(shared_indices + num_threads); // LOW_Integer type for shared lattice
    Real* shared_fitness          = reinterpret_cast<Real*>(shared_lattice + (num_atoms + 3) / 4); // Adjust for LOW_Integer
    Real* shared_coefficients     = shared_fitness + num_threads;

    shared_fitness[tid] = 100;

    // Init fitness
    if (tid == 0) {
        shared_depth[0] = 0;
    }
    Real reg_fitness = fitness[bid];

    // Pack 4 atoms into each byte
    for (Integer ii = tid; ii < (num_atoms + 3) / 4; ii += num_threads) {
        LOW_Integer packed = 0;
        for (Integer jj = 0; jj < 4 && ii * 4 + jj < num_atoms; jj++) {
            bit_utils::set_type(packed, jj, LOW_Integer(lattices[bid * num_atoms + ii * 4 + jj]));
        }
        shared_lattice[ii] = packed;
    }
    for (Integer ii = tid; ii < NUM_COEFFICIENTS; ii += num_threads) {
        shared_coefficients[ii] = 0.0;
    }
    
    calculate_coefficients_bit_packed<Integer, Real, LOW_Integer, LOW_Real, NUM_TYPES, NUM_SHELLS>(
        num_atoms, species, weights, neighbor_list, neighbor_list_indices, shared_coefficients, shared_lattice);
    
    Real coefficients[NUM_COEFFICIENTS];
    Integer reg_depth = shared_depth[0];

    while (reg_depth < search_depth) {
        // Initialize coefficients
        for (Integer ii = 0; ii < NUM_COEFFICIENTS; ii++) {
            coefficients[ii] = shared_coefficients[ii];
        }

        Integer atom1 = curand(&local_state) % num_atoms;
        Integer atom2 = curand(&local_state) % num_atoms;

        // Ensure atom1 and atom2 are different
        while (atom1 == atom2 || bit_utils::get_type(shared_lattice[atom1 / 4], atom1 % 4) == bit_utils::get_type(shared_lattice[atom2 / 4], atom2 % 4)) {
            atom1 = curand(&local_state) % num_atoms;
            atom2 = curand(&local_state) % num_atoms;
        }

        // Calculate fitness incrementally
        shared_fitness[tid] = calculate_fitness_incremental_bit_packed<Integer, Real, LOW_Integer, LOW_Real, NUM_TYPES, NUM_SHELLS>(
            atom1, atom2, num_atoms, species, weights, neighbor_list, neighbor_list_indices, target_sro, coefficients, shared_lattice);
        
        Integer id = locate_best_fitness(shared_fitness, shared_indices);

        if ((shared_fitness[id] < reg_fitness)) {
            reg_fitness = shared_fitness[id];
            if (tid == id) {
                // Swap atom1 and atom2 in the shared lattice (LOW_Integer)
                uint8_t type1 = bit_utils::get_type(shared_lattice[atom1 / 4], atom1 % 4);
                uint8_t type2 = bit_utils::get_type(shared_lattice[atom2 / 4], atom2 % 4);
                bit_utils::set_type(shared_lattice[atom1 / 4], atom1 % 4, type2);
                bit_utils::set_type(shared_lattice[atom2 / 4], atom2 % 4, type1);
                for (Integer ii = 0; ii < NUM_COEFFICIENTS; ii++) {
                    shared_coefficients[ii] = coefficients[ii];
                }
            }
            if (tid == 0) {
                shared_depth[0] = 0;
            }
        } else {
            if (tid == 0) {
                shared_depth[0] += 1;
            }
        }   
        __syncthreads();
        reg_depth = shared_depth[0];
    }
    __syncthreads();
    
    if (tid == 0) {
        fitness[bid] = reg_fitness;
    }
    
    // Save final lattice configuration
    for (Integer i = tid; i < (num_atoms + 3) / 4; i += blockDim.x) {
        uint8_t packed = shared_lattice[i];
        for (int j = 0; j < 4 && i * 4 + j < num_atoms; j++) {
            lattices[bid * num_atoms + i * 4 + j] = bit_utils::get_type(packed, j);
        }
    }
}

/**
 * @brief Sorts lattices based on their fitness values.
 * 
 * @tparam Integer Integer type for indices.
 * @tparam Real Floating-point type for fitness values.
 * @param[in,out] lattices Array of lattice configurations.
 * @param[in,out] fitness Array of fitness values.
 * @param[in] num_lattices Number of lattices to sort.
 * @param[in] num_atoms Number of atoms per lattice.
 */
template <typename Integer, typename Real>
void sort_lattices_by_fitness(Integer* lattices, Real* fitness, size_t num_lattices, size_t num_atoms) {
    Integer* indices = new Integer[num_lattices];
    std::iota(indices, indices + num_lattices, 0);
    
    Real* host_fitness = new Real[num_lattices * 2];
    cudaMemcpy(host_fitness, fitness, num_lattices * sizeof(Real), cudaMemcpyDeviceToHost);
    for (size_t id = 0; id < num_lattices; ++id) {
        host_fitness[num_lattices + id] = host_fitness[id];
    }
    
    std::sort(indices, indices + num_lattices, [&host_fitness](Integer a, Integer b) {
        return host_fitness[a] < host_fitness[b];
    });
    
    Integer* sorted_lattices = nullptr;
    cudaMalloc(&sorted_lattices, num_lattices * num_atoms * sizeof(Integer));
    
    for (size_t id = 0; id < num_lattices; ++id) {
        Integer sorted_index = indices[id];
        host_fitness[id] = host_fitness[num_lattices + sorted_index];
        cudaMemcpy(sorted_lattices + id * num_atoms, lattices + sorted_index * num_atoms, num_atoms * sizeof(Integer), cudaMemcpyDeviceToDevice);
    }
    
    cudaMemcpy(fitness, host_fitness, num_lattices * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(lattices, sorted_lattices, num_lattices * num_atoms * sizeof(Integer), cudaMemcpyDeviceToDevice);

    delete[] indices;
    delete[] host_fitness;
    cudaFree(sorted_lattices);
}

/**
 * @brief Generates random lattice configurations using parallel Monte Carlo.
 * 
 * @tparam Integer Integer type for indices.
 * @tparam Real Floating-point type for calculations.
 * @param[in] num_lattices Number of lattices to generate.
 * @param[in] num_types Number of atom types.
 * @param[in] num_atoms Number of atoms per lattice.
 * @param[in] num_shells Number of neighbor shells.
 * @param[in] species Array containing count of each atom type.
 * @param[in] weights Weight factors for each shell.
 * @param[in] neighbor_list List of neighbor indices.
 * @param[in] neighbor_list_indices Starting indices in neighbor_list.
 * @param[in] target_sro Target SRO values to compare against.
 * @param[out] coefficients Array for computed coefficients.
 * @param[out] fitness Array for computed fitness values.
 * @param[out] lattices Output array for generated lattices.
 */
template <typename Integer, typename Real>
void generate_random_lattices(
        Integer num_lattices, 
        Integer num_types, Integer num_atoms, Integer num_shells,
        const Integer* species, const Real* weights,
        const Integer* neighbor_list, const Integer* neighbor_list_indices,
        const Real* target_sro, Real* coefficients,
        Real* fitness, Integer* lattices)
{
    generate_normal_lattices<<<num_lattices, THREADS_PER_BLOCK>>>(
        /* inputs  */ num_atoms, num_types, species, 
        /* outputs */ lattices);

    unsigned long long seed = get_microseconds();
    thrust::default_random_engine rand_generator(seed);
    for (Integer ii = 0; ii < num_lattices; ii++) {
        thrust::shuffle(thrust::device, lattices + ii * num_atoms, lattices + (ii + 1) * num_atoms, rand_generator);
    }

    // Calculate fitness
    calculate_fitness_of_lattices<<<num_lattices, THREADS_PER_BLOCK, num_types * num_types * num_shells * 3 * sizeof(Real)>>>(
        /* inputs  */ num_atoms, num_types, num_shells, neighbor_list, neighbor_list_indices, species, weights, 
        /* outputs */ target_sro, coefficients, fitness, lattices);
}

/**
 * @brief Selects best lattice configurations from current and new populations.
 * 
 * @tparam Integer Integer type for indices.
 * @tparam Real Floating-point type for fitness values.
 * @param[in] num_lattices Number of lattices to select.
 * @param[in] num_atoms Number of atoms per lattice.
 * @param[in,out] lattices Current lattice configurations.
 * @param[in] new_lattices New lattice configurations.
 * @param[in,out] fitness Current fitness values.
 * @param[in] new_fitness New fitness values.
 */
template <typename Integer, typename Real>
void local_parallel_monte_carlo(
    const Integer& num_lattices, const Integer& num_atoms, 
    const Integer& num_types, const Integer& num_shells, const Integer& num_tasks,
    const Integer& search_depth, const Real& threshold,    
    const Integer* neighbor_list, const Integer* neighbor_list_indices,
    const Integer* species, const Real* weights,
    const Real* target_sro,
    Real* fitness, Integer* lattices)
{
    using LOW_Integer = uint8_t;
    unsigned long long seed = get_microseconds();
    const int num_coefficients = num_types * num_types * num_shells;
    size_t shared_usage = 2 * num_tasks * sizeof(Integer) 
        + num_atoms * sizeof(LOW_Integer)
        + num_tasks * sizeof(Real)
        + num_coefficients * sizeof(Real);

    // Get device properties to check maximum shared memory
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // Set maximum shared memory for the kernel
    cudaFuncAttributes attr;
    if (num_types == 3 && num_shells == 3) {
        cudaFuncGetAttributes(&attr, parallel_monte_carlo<Integer, Real, LOW_Integer, Real, 3, 3>);
        cudaFuncSetAttribute(parallel_monte_carlo<Integer, Real, LOW_Integer, Real, 3, 3>, 
                           cudaFuncAttributeMaxDynamicSharedMemorySize, 
                           prop.sharedMemPerBlockOptin);
    } else if (num_types == 4 && num_shells == 3) {
        cudaFuncGetAttributes(&attr, parallel_monte_carlo_bit_packed<Integer, Real, LOW_Integer, Real, 4, 3>);
        cudaFuncSetAttribute(parallel_monte_carlo_bit_packed<Integer, Real, LOW_Integer, Real, 4, 3>, 
                           cudaFuncAttributeMaxDynamicSharedMemorySize, 
                           prop.sharedMemPerBlockOptin);
    } else if (num_types == 5 && num_shells == 3) {
        cudaFuncGetAttributes(&attr, parallel_monte_carlo<Integer, Real, LOW_Integer, Real, 5, 3>);
        cudaFuncSetAttribute(parallel_monte_carlo<Integer, Real, LOW_Integer, Real, 5, 3>, 
                           cudaFuncAttributeMaxDynamicSharedMemorySize, 
                           prop.sharedMemPerBlockOptin);
    } else if (num_types == 6 && num_shells == 3) {
        cudaFuncGetAttributes(&attr, parallel_monte_carlo<Integer, Real, LOW_Integer, Real, 6, 3>);
        cudaFuncSetAttribute(parallel_monte_carlo<Integer, Real, LOW_Integer, Real, 6, 3>, 
                           cudaFuncAttributeMaxDynamicSharedMemorySize, 
                           prop.sharedMemPerBlockOptin);
    } else if (num_types == 4 && num_shells == 2) {
        cudaFuncGetAttributes(&attr, parallel_monte_carlo_bit_packed<Integer, Real, LOW_Integer, Real, 4, 2>);
        cudaFuncSetAttribute(parallel_monte_carlo_bit_packed<Integer, Real, LOW_Integer, Real, 4, 2>, 
                           cudaFuncAttributeMaxDynamicSharedMemorySize, 
                           prop.sharedMemPerBlockOptin);
    } else if (num_types == 5 && num_shells == 2) {
        cudaFuncGetAttributes(&attr, parallel_monte_carlo<Integer, Real, LOW_Integer, Real, 5, 2>);
        cudaFuncSetAttribute(parallel_monte_carlo<Integer, Real, LOW_Integer, Real, 5, 2>, 
                           cudaFuncAttributeMaxDynamicSharedMemorySize, 
                           prop.sharedMemPerBlockOptin);
    }

    // Check if shared memory requirement exceeds maximum available
    if (shared_usage > prop.sharedMemPerBlockOptin && (
        num_types != 4 || shared_usage / 4 > prop.sharedMemPerBlockOptin)) {
        auto now = std::chrono::system_clock::now();
        auto now_time_t = std::chrono::system_clock::to_time_t(now);
        std::tm* now_tm = std::localtime(&now_time_t);
        std::cerr << std::put_time(now_tm, "%Y-%m-%d %H:%M:%S") << " - PyHEA - INFO - " 
                  << "Error: Required shared memory (" << shared_usage 
                  << " bytes) exceeds maximum available shared memory (" 
                  << prop.sharedMemPerBlockOptin << " bytes)" << std::endl;
        exit(1);
    }
    shared_usage = prop.sharedMemPerBlockOptin;
    
    if (     num_types == 3 && num_shells == 3) {
        parallel_monte_carlo<Integer, Real, LOW_Integer, Real, 3, 3><<<num_lattices, num_tasks, shared_usage>>>(
            /* inputs  */ threshold, search_depth, seed, num_atoms, species, weights, neighbor_list, neighbor_list_indices, 
            /* outputs */ target_sro, fitness, lattices);
    }
    else if (num_types == 4 && num_shells == 3) {
        parallel_monte_carlo_bit_packed<Integer, Real, LOW_Integer, Real, 4, 3><<<num_lattices, num_tasks, shared_usage>>>(
            /* inputs  */ threshold, search_depth, seed, num_atoms, species, weights, neighbor_list, neighbor_list_indices, 
            /* outputs */ target_sro, fitness, lattices);
    } 
    else if (num_types == 5 && num_shells == 3) {
        parallel_monte_carlo<Integer, Real, LOW_Integer, Real, 5, 3><<<num_lattices, num_tasks, shared_usage>>>(
            /* inputs  */ threshold, search_depth, seed, num_atoms, species, weights, neighbor_list, neighbor_list_indices, 
            /* outputs */ target_sro, fitness, lattices);
    } 
    else if (num_types == 6 && num_shells == 3) {
        parallel_monte_carlo<Integer, Real, LOW_Integer, Real, 6, 3><<<num_lattices, num_tasks, shared_usage>>>(
            /* inputs  */ threshold, search_depth, seed, num_atoms, species, weights, neighbor_list, neighbor_list_indices, 
            /* outputs */ target_sro, fitness, lattices);
    }
    else if (num_types == 4 && num_shells == 2) {
        parallel_monte_carlo_bit_packed<Integer, Real, LOW_Integer, Real, 4, 2><<<num_lattices, num_tasks, shared_usage>>>(
            /* inputs  */ threshold, search_depth, seed, num_atoms, species, weights, neighbor_list, neighbor_list_indices, 
            /* outputs */ target_sro, fitness, lattices);
    } 
    else if (num_types == 5 && num_shells == 2) {
        parallel_monte_carlo<Integer, Real, LOW_Integer, Real, 5, 2><<<num_lattices, num_tasks, shared_usage>>>(
            /* inputs  */ threshold, search_depth, seed, num_atoms, species, weights, neighbor_list, neighbor_list_indices, 
            /* outputs */ target_sro, fitness, lattices);
    } 
    else if (num_types == 6 && num_shells == 2) {
        parallel_monte_carlo<Integer, Real, LOW_Integer, Real, 6, 2><<<num_lattices, num_tasks, shared_usage>>>(
            /* inputs  */ threshold, search_depth, seed, num_atoms, species, weights, neighbor_list, neighbor_list_indices, 
            /* outputs */ target_sro, fitness, lattices);
    } 
    else {
        auto now = std::chrono::system_clock::now();
        auto now_time_t = std::chrono::system_clock::to_time_t(now);
        std::tm* now_tm = std::localtime(&now_time_t);
        std::cerr <<  std::put_time(now_tm, "%Y-%m-%d %H:%M:%S") << " - PyHEA - INFO - " 
                  << "Error: Parallel Monte Carlo optimization "
                  << "GPU implementation is not available for the given number of types " << num_types 
                  << " and shells " << num_shells << std::endl;
        exit(1);
    } 
}

template <typename Integer, typename Real>
void calculate_best_lattices(Integer num_lattices, Integer num_atoms, Integer* lattices, Integer* new_lattices, Real* fitness, Real* new_fitness) {
    sort_lattices_by_fitness(lattices, fitness, 2 * num_lattices, num_atoms);
}

/**
 * @brief Main optimization function for finding optimal lattice configurations.
 * 
 * @param[in] num_iters Number of optimization iterations.
 * @param[in] num_lattices Number of lattices in population.
 * @param[in] host_species Array containing count of each atom type.
 * @param[in] host_weights Weight factors for each shell.
 * @param[in] host_nbor Neighbor list information.
 * @param[in] host_target_sro Target SRO values to achieve.
 * @param[in] host_lattices Initial lattice configurations.
 * @param[out] best_lattices Output array for best found configurations.
 * @param[out] best_fitness Output array for best fitness values.
 */
std::tuple<std::vector<std::vector<int>>, std::vector<double>> 
run_local_parallel_hcs_cuda(
        const int num_lattices,
        const int num_iters,
        const int num_tasks,
        const int search_depth,
        const double threshold,
        const std::vector<std::vector<std::vector<int>>>& neighbor_list,
        const std::vector<int>& host_species,
        const std::vector<double>& host_weights,
        const std::vector<std::vector<double>>& host_target_sro)
{
    // Flatten neighbor_list for easier copying to device
    int idx = 0;
    const int num_shells = host_weights.size();
    const int num_types  = host_species.size();
    const int num_atoms  = std::accumulate(host_species.begin(), host_species.end(), 0);
    const int num_coefficients = num_types * num_types * num_shells;
    std::vector<int> host_flat_nbor;
    std::vector<int> host_flat_nbor_idx;
    for (const auto& shell : neighbor_list) {
        int count = 0;
        for (const auto& neighbor_list : shell) {
            host_flat_nbor.insert(host_flat_nbor.end(), neighbor_list.begin(), neighbor_list.end());
            host_flat_nbor_idx.push_back(idx);
            idx += neighbor_list.size();
            count++;
            if (count == num_shells) {
                break;
            }
        }
    }
    host_flat_nbor_idx.push_back(idx);
    // Flatten target SRO for device
    std::vector<float> host_flat_target_sro;
    for (const auto& shell_sro : host_target_sro) {
        host_flat_target_sro.insert(host_flat_target_sro.end(), shell_sro.begin(), shell_sro.end());
    }

    int *species = nullptr, *flat_nbor = nullptr, *flat_nbor_idx = nullptr, *lattices = nullptr, *new_lattices = nullptr;
    float *weights = nullptr, *fitness = nullptr, *new_fitness = nullptr, *target_sro = nullptr, *coefficients = nullptr;

    cudaMalloc(&species, host_species.size() * sizeof(int));
    cudaMemcpy( species, host_species.data(), host_species.size() * sizeof(int), cudaMemcpyHostToDevice);

    std::vector<float> host_float_weights(host_weights.size(), 0.0);
    for (size_t ii = 0; ii < host_weights.size(); ii++) {
        host_float_weights[ii] = static_cast<float>(host_weights[ii]);
    }
    cudaMalloc(&weights, host_weights.size() * sizeof(float));
    cudaMemcpy( weights, host_float_weights.data(), host_weights.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    // Allocate memory on the GPU for the fitness array
    // The size is 2 * num_lattices * sizeof(float)
    cudaMalloc(&fitness,  2 * num_lattices * sizeof(float));
    // Allocate memory on the GPU for the lattices array
    // The size is 2 * num_lattices * num_atoms * sizeof(int)
    cudaMalloc(&lattices, 2 * num_lattices * num_atoms * sizeof(int));
    // Set new_fitness to point to the second half of the fitness array
    new_fitness  = fitness  + num_lattices;
    // Set new_lattices to point to the second half of the lattices array
    new_lattices = lattices + num_lattices * num_atoms;

    cudaMalloc(&flat_nbor, host_flat_nbor.size() * sizeof(int));
    cudaMemcpy( flat_nbor, host_flat_nbor.data(), host_flat_nbor.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&flat_nbor_idx, host_flat_nbor_idx.size() * sizeof(int));
    cudaMemcpy( flat_nbor_idx, host_flat_nbor_idx.data(), host_flat_nbor_idx.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&coefficients, num_lattices * num_coefficients * sizeof(float));

    // Allocate device memory for target SRO
    cudaMalloc(&target_sro, host_flat_target_sro.size() * sizeof(float));
    cudaMemcpy(target_sro, host_flat_target_sro.data(), host_flat_target_sro.size() * sizeof(float), cudaMemcpyHostToDevice);

    generate_random_lattices(
        /* inputs  */ num_lattices, num_types, num_atoms, num_shells, species, weights, flat_nbor, flat_nbor_idx, target_sro, coefficients, 
        /* outputs */ fitness, lattices);

    std::vector<float> host_fitness(num_lattices, 0);
    cudaMemcpy(host_fitness.data(), fitness, sizeof(float), cudaMemcpyDeviceToHost);

    // Global Search Loop
    for (int ii = 0; ii < num_iters; ii++) {
        // Perpuatation: Generate new lattices randomly
        generate_random_lattices(
            /* inputs  */ num_lattices, num_types, num_atoms, num_shells, species, weights, flat_nbor, flat_nbor_idx, target_sro, coefficients, 
            /* outputs */ new_fitness, new_lattices);
        // Local Search  I: Perform local parallel Monte Carlo optimization
        local_parallel_monte_carlo(
            /* inputs  */ num_lattices, num_atoms, num_types, num_shells, num_tasks, search_depth, (float)threshold, flat_nbor, flat_nbor_idx, species, weights, target_sro,
            /* outputs */ new_fitness, new_lattices);
        // Ranking: Calculate the best lattices and update the fitness values
        calculate_best_lattices(num_lattices, num_atoms, lattices, new_lattices, fitness, new_fitness);
        // Local Search II: Perform local parallel Monte Carlo optimization
        local_parallel_monte_carlo(
            /* inputs  */ num_lattices, num_atoms, num_types, num_shells, num_tasks, search_depth, (float)threshold, flat_nbor, flat_nbor_idx, species, weights, target_sro,
            /* outputs */ fitness, lattices);
        cudaMemcpy(host_fitness.data(), fitness, sizeof(float), cudaMemcpyDeviceToHost);
        auto now = std::chrono::system_clock::now();
        auto now_time_t = std::chrono::system_clock::to_time_t(now);
        std::tm* now_tm = std::localtime(&now_time_t);
        std::cout << std::put_time(now_tm, "%Y-%m-%d %H:%M:%S") << " - PyHEA - INFO - " << "Iter " << ii << " with best fitness:      " << host_fitness[0] << std::endl;
    }

    int * temp_lattices = new int[num_lattices * num_atoms];
    std::vector<std::vector<int>> final_lattices(num_lattices, std::vector<int>(num_atoms, 0));
    std::vector<float> temp_fitness(num_lattices, 0.0);
    std::vector<double> final_fitness(num_lattices, 0.0);
    cudaMemcpy(temp_lattices, lattices, num_lattices * num_atoms * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(temp_fitness.data(), fitness, num_lattices * sizeof(float), cudaMemcpyDeviceToHost);
    for (int ii = 0; ii < num_lattices; ii++) {
        for (int jj = 0; jj < num_atoms; jj++) {
            final_lattices[ii][jj] = temp_lattices[ii * num_atoms + jj];
        }
        final_fitness[ii] = temp_fitness[ii];
    }
    delete [] temp_lattices;

    cudaFree(species);
    cudaFree(weights);

    cudaFree(fitness);
    cudaFree(lattices);

    cudaFree(flat_nbor);
    cudaFree(flat_nbor_idx);

    cudaFree(target_sro);
    cudaFree(coefficients);

    return std::make_tuple(final_lattices, final_fitness);
}
    
} // namespace cpu
} // namespace accelerate