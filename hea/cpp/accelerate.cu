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

bool cuda_available() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    return device_count > 0;
}

#define THREADS_PER_BLOCK 256

unsigned long long get_microseconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}

/**
 * @brief Calculate the fitness of lattices based on Short-Range Order (SRO) coefficients.
 *
 * This kernel function computes the fitness of different lattices by evaluating the 
 * Short-Range Order (SRO) coefficients. Each block processes a different lattice.
 *
 * @tparam Integer Integer type, typically int.
 * @tparam Real Floating-point type, typically float or double.
 *
 * @param num_atoms [in] Number of atoms in each lattice.
 * @param num_types [in] Number of different atom types.
 * @param num_shells [in] Number of neighbor shells to consider.
 * @param neighbor_list [in] Array of neighbor indices for each atom.
 * @param neighbor_list_indices [in] Array of indices pointing to the start of each atom's neighbor list.
 * @param species [in] Array of species types for each atom.
 * @param weights [in] Array of weights for each shell.
 * @param sro [in] Array of SRO values for each shell.
 * @param coefficients [out] Array to store the computed SRO coefficients.
 * @param fitness [out] Array to store the computed fitness values for each lattice.
 * @param lattices [in] Array of lattices, each containing num_atoms atoms.
 *
 * The function performs the following steps:
 * 1. Initialize shared memory for SRO coefficients.
 * 2. For each atom in the lattice, compute the SRO coefficients by iterating over its neighbors.
 * 3. Use atomic operations to update shared memory with the computed values.
 * 4. Copy the shared SRO coefficients to global memory.
 * 5. Compute the fitness value for the lattice based on the SRO coefficients and weights.
 *
 * The fitness value is calculated using the formula:
 * \f[
 * \text{fitness} = \sqrt{\sum_{k=0}^{\text{num\_shells}-1} \text{weights}[k] \times \text{error}_k}
 * \f]
 * where \f$\text{error}_k\f$ is the error for shell \f$k\f$, computed as:
 * \f[
 * \text{error}_k = \sum_{i=0}^{\text{num\_types}-1} \sum_{j=0}^{\text{num\_types}-1} (i \neq j) \left( \text{shared\_gamma}[i \times \text{num\_types} \times \text{num\_shells} + j \times \text{num\_shells} + k] - \frac{\text{shared\_alpha}[i \times \text{num\_types} \times \text{num\_shells} + j \times \text{num\_shells} + k]}{\text{shared\_count}[i \times \text{num\_types} \times \text{num\_shells} + j \times \text{num\_shells} + k]} \right)^2
 * \f]
 */
template <typename Integer, typename Real>
__global__ void calculate_fitness_of_lattices(
        Integer num_atoms, Integer num_types, Integer num_shells,
        const Integer* neighbor_list, const Integer* neighbor_list_indices, 
        const Integer* species, const Real* weights, 
        const Real* sro, Real* coefficients, 
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
        Real target_sro = 0.0;
        Real fitness_shell = 0.0;
        for (Integer ii = 0; ii < num_types; ++ii) {
            for (Integer jj = 0; jj < num_types; ++jj) {
                Integer index = ii * num_types * num_shells + jj * num_shells + shell;
                Real sro = 1 - shared_gamma[index] / ((static_cast<Real>(species[ii]) / static_cast<Real>(num_atoms)) * (static_cast<Real>(species[jj]) / static_cast<Real>(num_atoms)));
                fitness_shell += (sro - target_sro) * (sro - target_sro);
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
 * @param[in,out] rng_states Pointer to an array of CURAND states.
 * @param[in] seed The seed for random number generation.
 */
__global__ void init_curand(curandState* rng_states, unsigned long long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, id, 0, &rng_states[id]);
}

/**
 * @brief Kernel function to generate normal lattices based on species distribution.
 *
 * This CUDA kernel generates normal lattices for a given number of atoms. Each lattice
 * is determined by the species distribution provided. The kernel assigns lattice values
 * based on the species array, where each species type is represented by an integer.
 *
 * @tparam Integer The integer type used for indexing and lattice values.
 * @param num_atoms [in] The total number of atoms in each lattice.
 * @param species [in] An array representing the distribution of species. The first element
 *                indicates the number of atoms of species 0, the second element indicates
 *                the number of atoms of species 1, and so on. Dimension: [number of species]
 * @param lattices [out] An output array where the generated lattices will be stored. Each block
 *                 generates one lattice, and the lattices are stored consecutively in this array.
 *                 Dimension: [number of blocks * num_atoms]
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

template <typename Integer, typename Real,
          typename LOW_Integer, typename LOW_Real,
          int NUM_TYPES, int NUM_SHELLS>
__device__ __forceinline__ 
void calculate_coefficients(
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
__device__ __forceinline__ 
Real calculate_fitness(
    const Integer& atom1, const Integer& atom2,
    const Integer& num_atoms,
    const Integer* species, const Real* weights,
    const Integer* neighbor_list, const Integer* neighbor_list_indices,
    const Real* sro, Real* coefficients,
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
        Real target_sro = 0.0;
        Real fitness_shell = 0.0;
        for (Integer ii = 0; ii < NUM_TYPES; ++ii) {
            for (Integer jj = 0; jj < NUM_TYPES; ++jj) {
                Integer index = ii * NUM_TYPES * NUM_SHELLS + jj * NUM_SHELLS + shell;
                Real sro = 1 - gamma[index] / ((static_cast<Real>(species[ii]) / static_cast<Real>(num_atoms)) * (static_cast<Real>(species[jj]) / static_cast<Real>(num_atoms)));
                fitness_shell += (sro - target_sro) * (sro - target_sro);
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
__device__ __forceinline__ 
Real calculate_fitness_incremental(
    const Integer& atom1, const Integer& atom2,
    const Integer& num_atoms,
    const Integer* species, const Real* weights,
    const Integer* neighbor_list, const Integer* neighbor_list_indices,
    const Real* sro, Real* coefficients,
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
        Real target_sro = 0.0;
        Real fitness_shell = 0.0;
        for (Integer ii = 0; ii < NUM_TYPES; ++ii) {
            for (Integer jj = 0; jj < NUM_TYPES; ++jj) {
                Integer index = ii * NUM_TYPES * NUM_SHELLS + jj * NUM_SHELLS + shell;
                Real sro = 1 - gamma[index] / ((static_cast<Real>(species[ii]) / static_cast<Real>(num_atoms)) * (static_cast<Real>(species[jj]) / static_cast<Real>(num_atoms)));
                fitness_shell += (sro - target_sro) * (sro - target_sro);
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
__global__ void parallel_monte_carlo(
    const Real threshold, const Integer search_depth, const unsigned long long random_seed,
    const Integer num_atoms,
    const Integer* species, const Real* weights,
    const Integer* neighbor_list, const Integer* neighbor_list_indices,
    const Real* sro,
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
            atom1, atom2, num_atoms, species, weights, neighbor_list, neighbor_list_indices, sro, coefficients, shared_lattice);
        // shared_fitness[tid] = calculate_fitness<Integer, Real, LOW_Integer, LOW_Real, NUM_TYPES, NUM_SHELLS>(
            // atom1, atom2, num_atoms, species, weights, neighbor_list, neighbor_list_indices, sro, coefficients, shared_lattice);

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

template <typename Integer, typename Real>
void generate_random_lattices(
        Integer num_lattices, 
        Integer num_types, Integer num_atoms, Integer num_shells,
        const Integer* species, const Real* weights,
        const Integer* neighbor_list, const Integer* neighbor_list_indices,
        const Real* sro, Real* coefficients,
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
        /* outputs */ sro, coefficients, fitness, lattices);
}

/**
 * @brief Optimizes atomic configurations using a Monte Carlo method to minimize fitness.
 *
 * This function performs a Monte Carlo optimization on a set of atomic configurations, 
 * adjusting their arrangements to minimize a fitness metric related to short-range order.
 *
 * @param lattices A reference to a 2D vector containing the current atomic configurations.
 * @param fitness A reference to a vector containing the fitness values associated with each configuration.
 * @param neighbor_list A constant reference to a 3D vector containing neighboring atom information 
 *        for each shell.
 * @param species A constant reference to a vector representing the types of elements present.
 * @param weights A constant reference to a vector of weights for the short-range order calculations.
 *
 * @return A tuple containing the updated lattices and fitness values after optimization.
 */
template <typename Integer, typename Real>
void local_parallel_monte_carlo(
    const Integer& num_lattices, const Integer& num_atoms, 
    const Integer& num_types, const Integer& num_shells, const Integer& num_tasks,
    const Integer& search_depth, const Real& threshold,    
    const Integer* neighbor_list, const Integer* neighbor_list_indices,
    const Integer* species, const Real* weights,
    const Real* sro,
    Real* fitness, Integer* lattices)
{
    unsigned long long seed = get_microseconds();
    const int num_coefficients = num_types * num_types * num_shells;
    const size_t shared_usage = 2 * num_tasks * sizeof(Integer) 
        + num_atoms * sizeof(uint16_t)
        + num_tasks * sizeof(Real)
        + num_coefficients * sizeof(Real);
    
    if (     num_types == 3 && num_shells == 3) {
        parallel_monte_carlo<Integer, Real, uint16_t, Real, 3, 3><<<num_lattices, num_tasks, shared_usage>>>(
            /* inputs  */ threshold, search_depth, seed, num_atoms, species, weights, neighbor_list, neighbor_list_indices, 
            /* outputs */ sro, fitness, lattices);
    }
    else if (num_types == 4 && num_shells == 3) {
        parallel_monte_carlo<Integer, Real, uint8_t, Real, 4, 3><<<num_lattices, num_tasks, shared_usage>>>(
            /* inputs  */ threshold, search_depth, seed, num_atoms, species, weights, neighbor_list, neighbor_list_indices, 
            /* outputs */ sro, fitness, lattices);
    } 
    else if (num_types == 5 && num_shells == 3) {
        parallel_monte_carlo<Integer, Real, uint8_t, Real, 5, 3><<<num_lattices, num_tasks, shared_usage>>>(
            /* inputs  */ threshold, search_depth, seed, num_atoms, species, weights, neighbor_list, neighbor_list_indices, 
            /* outputs */ sro, fitness, lattices);
    } 
    else if (num_types == 6 && num_shells == 3) {
        parallel_monte_carlo<Integer, Real, uint8_t, Real, 6, 3><<<num_lattices, num_tasks, shared_usage>>>(
            /* inputs  */ threshold, search_depth, seed, num_atoms, species, weights, neighbor_list, neighbor_list_indices, 
            /* outputs */ sro, fitness, lattices);
    }
    else if (num_types == 4 && num_shells == 2) {
        parallel_monte_carlo<Integer, Real, uint8_t, Real, 4, 2><<<num_lattices, num_tasks, shared_usage>>>(
            /* inputs  */ threshold, search_depth, seed, num_atoms, species, weights, neighbor_list, neighbor_list_indices, 
            /* outputs */ sro, fitness, lattices);
    } 
    else if (num_types == 5 && num_shells == 2) {
        parallel_monte_carlo<Integer, Real, uint8_t, Real, 5, 2><<<num_lattices, num_tasks, shared_usage>>>(
            /* inputs  */ threshold, search_depth, seed, num_atoms, species, weights, neighbor_list, neighbor_list_indices, 
            /* outputs */ sro, fitness, lattices);
    } 
    else if (num_types == 6 && num_shells == 2) {
        parallel_monte_carlo<Integer, Real, uint8_t, Real, 6, 2><<<num_lattices, num_tasks, shared_usage>>>(
            /* inputs  */ threshold, search_depth, seed, num_atoms, species, weights, neighbor_list, neighbor_list_indices, 
            /* outputs */ sro, fitness, lattices);
    } 
    else {
        std::cerr << "GPU implementation is not available for the given number of types " << num_types 
                  << " and shells " << num_shells << std::endl;
        exit(1);
    } 
}

template <typename Integer, typename Real>
void calculate_best_lattices(Integer num_lattices, Integer num_atoms, Integer* lattices, Integer* new_lattices, Real* fitness, Real* new_fitness) {
    sort_lattices_by_fitness(lattices, fitness, 2 * num_lattices, num_atoms);
}

/**
 * @brief Executes a local parallel heuristic search (HCS) algorithm.
 *
 * This function performs a local parallel heuristic search to optimize a set of lattices.
 * It generates initial random lattices, evaluates their fitness, and iteratively improves
 * them using a Monte Carlo method until a stopping criterion is met.
 *
 * @param num_lattices The number of networks or lattices to generate.
 * @param num_iters The maximum number of iterations to perform.
 * @param num_tasks The num_tasks identifier for the Monte Carlo method.
 * @param search_depth The search_depth parameter for the Monte Carlo method.
 * @param threshold The threshold value for the fitness score to stop the iterations.
 * @param neighbor_list A 3D vector representing the neighborhood relationships.
 * @param species A vector representing the elements or nodes in the lattices.
 * @param weights A vector representing the weights associated with the elements.
 * @return A tuple containing the optimized lattices and their corresponding fitness scores.
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
    for (size_t ii = 0; ii < host_target_sro.size(); ii++) {
        for (size_t jj = 0; jj < host_target_sro[ii].size(); jj++) {
            std::cout << host_target_sro[ii][jj] << " ";
        }
        std::cout << std::endl;
    }
    // Flatten neighbor_list for easier copying to device
    int idx = 0;
    const int num_shells = host_weights.size();
    const int num_types  = host_species.size();
    const int num_atoms  = std::accumulate(host_species.begin(), host_species.end(), 0);
    const int num_coefficients = num_types * num_types * num_shells * 3;
    std::vector<int> host_flat_nbor;
    std::vector<int> host_flat_nbor_idx;
    std::vector<int> h_sro(num_shells, 0);
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

    int *species = nullptr, *flat_nbor = nullptr, *flat_nbor_idx = nullptr, *lattices = nullptr, *new_lattices = nullptr;
    float *weights = nullptr, *fitness = nullptr, *new_fitness = nullptr, *sro = nullptr, *coefficients = nullptr;

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
    
    cudaMalloc(&sro, num_shells * sizeof(float));
    cudaMemcpy( sro, h_sro.data(), h_sro.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&coefficients, num_lattices * num_coefficients * sizeof(float));

    generate_random_lattices(
        /* inputs  */ num_lattices, num_types, num_atoms, num_shells, species, weights, flat_nbor, flat_nbor_idx, sro, coefficients, 
        /* outputs */ fitness, lattices);

    std::vector<float> host_fitness(num_lattices, 0);
    cudaMemcpy(host_fitness.data(), fitness, sizeof(float), cudaMemcpyDeviceToHost);

    // Global Search Loop
    for (int ii = 0; ii < num_iters; ii++) {
        // Perpuatation: Generate new lattices randomly
        generate_random_lattices(
            /* inputs  */ num_lattices, num_types, num_atoms, num_shells, species, weights, flat_nbor, flat_nbor_idx, sro, coefficients, 
            /* outputs */ new_fitness, new_lattices);
        // Local Search  I: Perform local parallel Monte Carlo optimization
        local_parallel_monte_carlo(
            /* inputs  */ num_lattices, num_atoms, num_types, num_shells, num_tasks, search_depth, (float)threshold, flat_nbor, flat_nbor_idx, species, weights, sro,
            /* outputs */ new_fitness, new_lattices);
        // Ranking: Calculate the best lattices and update the fitness values
        calculate_best_lattices(num_lattices, num_atoms, lattices, new_lattices, fitness, new_fitness);
        // Local Search II: Perform local parallel Monte Carlo optimization
        local_parallel_monte_carlo(
            /* inputs  */ num_lattices, num_atoms, num_types, num_shells, num_tasks, search_depth, (float)threshold, flat_nbor, flat_nbor_idx, species, weights, sro,
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

    cudaFree(sro);
    cudaFree(coefficients);

    return std::make_tuple(final_lattices, final_fitness);
}
    
} // namespace cpu
} // namespace accelerate