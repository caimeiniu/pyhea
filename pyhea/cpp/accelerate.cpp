#include "accelerate.h"

#include <mpi.h>
#include <ctime>
#include <chrono>

// for std::vector
#include <vector>
#include <random>

// for std::copy
#include <iomanip>
#include <iostream>
// for sort and iota functions
#include <numeric>
#include <algorithm>

namespace accelerate {
namespace cpu {

/**
 * @brief Sorts the lattices based on fitness values.
 *
 * This function sorts the lattices based on the fitness values associated with them,
 * ensuring that the best lattices (lowest fitness values) are placed at the beginning 
 * of the lattices array. It updates both the fitness values and the lattices to 
 * reflect the sorted order.
 *
 * @tparam Integer Type used for lattice representation (e.g., int).
 * @tparam Real Type used for fitness values (e.g., double).
 *
 * @param[in, out] fitness A reference to a vector of fitness values (dimension: [num_solutions]),
 *                         where each element corresponds to a lattice's fitness.
 * @param[in, out] lattices A reference to a 2D vector of lattices (dimension: [num_solutions][num_variables]),
 *                           where each inner vector represents a lattice comprised of integers.
 */
template <typename Integer, typename Real>
void sort_lattices_by_fitness(std::vector<Real>& fitness, std::vector<std::vector<Integer>>& lattices) {
    // Get the number of lattices
    const size_t num_solutions = lattices.size();
    
    // Create an index vector to hold the original indices of lattices
    std::vector<Integer> indices(num_solutions);
    std::iota(indices.begin(), indices.end(), 0);  // Fill indices with 0, 1, ..., num_solutions - 1

    // Sort indices based on the corresponding fitness values
    std::sort(indices.begin(), indices.end(), [&fitness](Integer a, Integer b) {
        return fitness[a] < fitness[b];  // Ascending order
    });

    // Prepare vectors to hold sorted fitness values and lattices
    std::vector<Real> sorted_fitness(num_solutions);                   // Sorted fitness values
    std::vector<std::vector<Integer>> sorted_solutions(num_solutions); // Sorted lattices

    // Rearranging lattices and fitness based on sorted indices
    for (size_t idx = 0; idx < num_solutions; ++idx) {
        sorted_fitness[idx] = fitness[indices[idx]];             // Copy sorted fitness
        sorted_solutions[idx] = lattices[indices[idx]];         // Copy sorted lattices
    }

    // Update original vectors with sorted values
    fitness = std::move(sorted_fitness);
    lattices = std::move(sorted_solutions);
}

/**
 * @brief Calculates the fitness values of given lattices based on short-range order (SRO).
 *
 * This function evaluates a set of lattices by analyzing the pairwise atomic interactions
 * within defined shells. The fitness value represents how well a lattice meets the desired
 * short-range order characteristics.
 *
 * @tparam Integer Type used for indexing (e.g., int).
 * @tparam Real Type used for floating-point values (e.g., double).
 *
 * @param[out] fitness A vector of fitness values, dimension: [num_solutions].
 *                     Each element represents the fitness of the corresponding lattice,
 *                     calculated as the square root of the weighted error.
 * @param[in] weight A vector of weights, dimension: [num_shells].
 *                   Each element corresponds to the importance of errors from each shell in the total fitness calculation.
 * @param[in] species A vector representing the species (types of atoms), dimension: [num_types].
 *                    Each element indicates a unique type of atom.
 * @param[in] lattices A 2D vector of lattices, dimension: [num_solutions][num_atoms].
 *                      Each inner vector contains the indices of atoms that form a specific lattice.
 * @param[in] neighbor_list A 3D vector of neighbor lists, dimension: [num_atoms][num_shells][...].
 *                          Each element lists the neighboring atoms for a given atom in a specific shell.
 *
 * The algorithm performs the following steps:
 * 1. Initialize matrices `alpha`, `gamma`, and `count` for pairwise interactions.
 *    - `alpha[atom1][atom2][shell]` accumulates the calculated values for interactions between `atom1` and `atom2` in the specified shell.
 *    - `gamma[atom1][atom2][shell]` counts the occurrences of each pair in the specified shell.
 *    - `count[atom1][atom2][shell]` tracks how many times the pair interaction has been considered.
 *
 * 2. For each lattice, iterate through all atoms and their neighbors:
 *    - For each shell, compute:
 *      - \( \text{gamma}[i][j][k] \) which counts the number of neighbors of type `j` around atom `i`.
 *      - \( \text{alpha}[i][j][k] \) which is calculated as:
 *        \[
 *        \text{alpha}[i][j][k] += \text{species}[i] \times (1 - \text{sro}[k]) \times \text{shell} \times \left(\frac{\text{species}[j]}{\text{num_atoms}}\right)
 *        \]
 *      - Increment the `count` for valid pairs.
 *
 * 3. Calculate the error for each shell:
 *    - For each shell, compute the error as:
 *      \[
 *      \text{error\_kk} = \sum_{i \neq j} \left( \text{gamma}[i][j][k] - \frac{\text{alpha}[i][j][k]}{\text{count}[i][j][k]} \right)^2
 *      \]
 *    - The total error is then:
 *      \[
 *      \text{error} = \sum_{k} \text{weight}[k] \times \text{error\_kk}
 *      \]
 *
 * 4. Store the calculated fitness value as:
 *    \[
 *    \text{fitness}[accumulate] = \sqrt{\text{error}}
 *    \]
 */
template <typename Integer, typename Real>
void calculate_fitness_of_solutions(
        std::vector<Real>& fitness,
        const std::vector<std::vector<Integer>>& lattices,
        const std::vector<Real>& sro,
        const std::vector<Real>& weight,
        const std::vector<Integer>& species,
        const std::vector<std::vector<std::vector<Integer>>>& neighbor_list,
        const std::vector<std::vector<Real>>& target_sro) 
{
    // Get the number of atom types, atoms, shells, and lattices
    const Integer num_types     = species.size(); // Number of atom types
    const Integer num_atoms     = lattices[0].size(); // Number of atoms in each lattice
    const Integer num_shells    = weight.size(); // Number of shells
    const Integer num_solutions  = lattices.size(); // Number of lattices

    using vec1 = std::vector<Real>; // Alias for 1D vector
    using vec2 = std::vector<std::vector<Real>>; // Alias for 2D vector
    using vec3 = std::vector<std::vector<std::vector<Real>>>; // Alias for 3D vector

    // Initialize matrices
    vec3 gamma(num_types, vec2(num_types, vec1(num_shells, 0.0))); // Dimension: [num_types][num_types][num_shells]

    Integer accumulate = 0; // Counter for fitness values

    // Iterate through each lattice
    for (auto lattice : lattices) {
        // Initialize gamma matrices for this lattice
        for (Integer ii = 0; ii < num_types * num_types * num_shells; ++ii) {
            auto xx =  ii / (num_types * num_shells); // Current atom type index
            auto yy = (ii /  num_shells) % num_types; // Current atom type index
            auto zz =  ii %  num_shells; // Current shell index
            gamma[xx][yy][zz] = 0.0;
        }

        // Compute gamma distribution
        for (Integer ii = 0; ii < num_atoms; ++ii) { // Loop over each atom in the lattice
            Integer atom = lattice[ii]; // Current atom index
            for (Integer jj = 0; jj < num_shells; ++jj) { // Loop over each shell
                Integer shell = neighbor_list[ii][jj].size(); // Number of neighbors in this shell
                for (Integer kk = 0; kk < shell; ++kk) { // Loop over each neighbor
                    Integer neighbor = lattice[neighbor_list[ii][jj][kk]]; // Neighbor atom index
                    gamma[atom][neighbor][jj] += 1 / (static_cast<Real>(shell) * num_atoms); // Increment gamma count for this pair
                }
            }
        }

        // Calculate the error for each shell
        Real error = 0.0; // Initialize total error
        for (Integer kk = 0; kk < num_shells; ++kk) { // Loop over each shell
            Real error_kk = 0.0; // Initialize shell-specific error
            for (Integer ii = 0; ii < num_types; ++ii) { // Loop over each atom type
                for (Integer jj = 0; jj < num_types; ++jj) { // Loop over each atom type
                    Real sro = 1 - gamma[ii][jj][kk] / ((static_cast<Real>(species[ii]) / static_cast<Real>(num_atoms)) * (static_cast<Real>(species[jj]) / static_cast<Real>(num_atoms)));
                    Real shell_sro = target_sro[kk][ii * num_types + jj];
                    error_kk += std::pow(sro - shell_sro, 2); // Accumulate squared error
                }
            }
            error += weight[kk] * error_kk; // Accumulate weighted error for the shell
        }
        fitness[accumulate++] = std::sqrt(error);  // Store the calculated fitness value
    }
}

/**
 * @brief Generates random lattices for a set of atomic configurations and calculates their fitness.
 *
 * This function creates random atomic configurations based on the specified species and their counts.
 * Each configuration (lattice) is a permutation of the available atoms, and the fitness of each lattice
 * is calculated using the provided `calculate_fitness_of_solutions` function.
 *
 * @tparam Integer Type used for indexing (e.g., int).
 * @tparam Real Type used for floating-point values (e.g., double).
 *
 * @param[out] fitness A vector of fitness values, dimension: [num_solutions].
 *                     Each element represents the fitness of the corresponding generated lattice.
 * @param[out] lattices A 2D vector of lattices, dimension: [num_solutions][num_atoms].
 *                       Each inner vector contains a random permutation of atom indices for a specific lattice.
 * @param[in] weight A vector of weights, dimension: [num_shells].
 *                   Each element corresponds to the importance of errors from each shell in the total fitness calculation.
 * @param[in] species A vector representing the species (types of atoms) and their counts, dimension: [num_types].
 *                    Each element indicates the number of atoms of that type.
 * @param[in] neighbor_list A 3D vector of neighbor lists, dimension: [num_atoms][num_shells][...].
 *                          Each element lists the neighboring atoms for a given atom in a specific shell.
 *
 * The algorithm performs the following steps:
 * 1. Initialize a vector to hold all possible atom indices based on species counts:
 *    - For each atom type, insert its index `type` multiple times into `all_atoms` according to its count.
 *      \[
 *      \text{all\_atoms} = \{ \text{type} \text{ (count times)} \} \quad \forall \text{type}
 *      \]
 *
 * 2. Randomly shuffle the `all_atoms` vector to generate unique configurations for each lattice:
 *    - Utilize a random number generator to shuffle the indices.
 *    - The shuffling process ensures each lattice has a random permutation of the available atoms.
 *
 * 3. Copy the shuffled atom indices into the `lattices` vector for each lattice.
 *
 * 4. Calculate the fitness values for the generated lattices using the `calculate_fitness_of_solutions` function.
 */
template <typename Integer, typename Real>
void generate_random_lattices(
    std::vector<Real>& fitness,
    std::vector<std::vector<Integer>>& lattices,
    const std::vector<Real>& sro,
    const std::vector<Real>& weight,
    const std::vector<Integer>& species,
    const std::vector<std::vector<std::vector<Integer>>>& neighbor_list,
    const std::vector<std::vector<Real>>& target_sro)
{
    const int num_types     = species.size(); // Number of atom types
    const int num_solutions = lattices.size(); // Number of lattices to generate

    // Create a vector to hold all possible atoms
    std::vector<Integer> all_atoms; // Vector to store atom indices
    Integer num_atoms = std::accumulate(species.begin(), species.end(), 0); // Total number of atoms

    // Populate the all_atoms vector with indices based on species counts
    for (Integer type = 0; type < num_types; ++type) {
        all_atoms.insert(all_atoms.end(), species[type], type); // Add 'count' instances of 'atom type'
    }

    // Random Number Generator initialization
    std::random_device rd; // Non-deterministic random number generator
    std::mt19937 gen(rd()); // Mersenne Twister generator seeded with random device

    // Generate random lattices
    for (Integer ii = 0; ii < num_solutions; ii++) {
        std::shuffle(all_atoms.begin(), all_atoms.end(), gen); // Shuffle the atoms randomly
        std::copy(all_atoms.begin(), all_atoms.end(), lattices[ii].begin()); // Copy shuffled atoms to lattices
    }

    // Calculate fitness values for the generated lattices
    calculate_fitness_of_solutions(fitness, lattices, sro, weight, species, neighbor_list, target_sro);
}


/**
 * @brief Copies the values from the right-hand side 3D vector to the left-hand side 3D vector.
 *
 * This function performs an species-wise copy of the contents of the right-hand side
 * vector to the left-hand side vector, ensuring that both vectors have the same dimensions.
 *
 * @param[out] lhs A reference to the left-hand side 3D vector to be modified.
 * @param[in]  rhs A constant reference to the right-hand side 3D vector to copy from.
 */
template <typename Real>
void copy(std::vector<std::vector<std::vector<Real>>>& lhs, 
           const std::vector<std::vector<std::vector<Real>>>& rhs) {
    for (size_t i = 0; i < lhs.size(); ++i) {
        for (size_t j = 0; j < lhs[i].size(); ++j) {
            lhs[i][j] = rhs[i][j];  // Directly copy inner vectors
        }
    }
}

/**
 * @brief Calculates the best lattices by combining existing lattices with new lattices.
 *
 * This function takes a set of existing lattices and new lattices, evaluates their fitness,
 * and retains the best lattices based on fitness values. The best lattices replace the existing ones.
 *
 * @param[in,out] lattices A 2D vector of existing lattices, dimension: [num_solutions][num_atoms].
 *                          Each inner vector contains indices representing atomic configurations.
 * @param[in,out] new_solutions A 2D vector of new lattices, dimension: [num_new_solutions][num_atoms].
 *                               Each inner vector contains indices representing atomic configurations.
 * @param[in,out] fitness A vector of fitness values for existing lattices, dimension: [num_solutions].
 *                        Each element indicates how well a lattice meets the desired criteria.
 * @param[in] new_fitness A vector of fitness values for new lattices, dimension: [num_new_solutions].
 *                        Each element represents the fitness of a new lattice.
 *
 * The algorithm performs the following steps:
 * 1. Initialize the number of existing lattices.
 * 2. Combine existing fitness values and new fitness values into a single vector:
 *    \[
 *    \text{result\_fitness} = \text{fitness} \cup \text{new\_fitness}
 *    \]
 *
 * 3. Combine existing lattices and new lattices into a single vector:
 *    \[
 *    \text{result\_solutions} = \text{lattices} \cup \text{new\_solutions}
 *    \]
 *
 * 4. Sort the combined lattices based on their fitness values using the `sort_lattices_by_fitness` function.
 *
 * 5. Replace the original fitness and lattices with the top `num_solutions` from the sorted results.
 */
template <typename Integer, typename Real>
void calculate_best_lattices(
    std::vector<std::vector<Integer>>& lattices,
    std::vector<std::vector<Integer>>& new_solutions,
    std::vector<Real>& fitness,
    std::vector<Real>& new_fitness) 
{
    Integer num_solutions = lattices.size(); // Number of existing lattices

    // Create vectors to hold combined fitness and lattices
    std::vector<Real> result_fitness = fitness; // Initialize with existing fitness values
    std::vector<std::vector<Integer>> result_solutions = lattices; // Initialize with existing lattices

    // Combine existing and new fitness values
    result_fitness.insert(result_fitness.end(), new_fitness.begin(), new_fitness.end());
    
    // Combine existing and new lattices
    result_solutions.insert(result_solutions.end(), new_solutions.begin(), new_solutions.end());

    // Sort the combined lattices based on their fitness
    sort_lattices_by_fitness(result_fitness, result_solutions);
    
    // Update the original fitness and lattices with the best `num_solutions`
    std::copy(result_fitness.begin(), result_fitness.begin() + num_solutions, fitness.begin());
    std::copy(result_solutions.begin(), result_solutions.begin() + num_solutions, lattices.begin());
}

/**
 * @brief Calculates the Short Range Order (SRO) coefficients based on atomic configurations.
 *
 * This function updates the `gamma`, `alpha`, and `count` matrices that represent the pair distribution
 * and average pair counts among different types of atoms in specified shells.
 *
 * @param[in,out] gamma A 3D vector representing the pair distribution counts, dimension: [num_types][num_types][num_shells].
 *                      Each element indicates the number of pairs of a specific type in a given shell.
 * @param[in,out] alpha A 3D vector for weighted pair counts, dimension: [num_types][num_types][num_shells].
 *                      Each element represents a weighted value based on the species and SRO.
 * @param[in,out] count A 3D vector to track the number of pairs considered, dimension: [num_types][num_types][num_shells].
 *                      Each element indicates how many pairs have been counted for the respective types in the shell.
 * @param[in] num_atoms The total number of atoms, representing the size of the atomic configuration.
 * @param[in] num_types The total number of atom types.
 * @param[in] num_shells The total number of shells to consider in the calculations.
 * @param[in] sro A vector of SRO target values for each shell, dimension: [num_shells].
 * @param[in] lattice A vector representing the current atomic configuration, dimension: [num_atoms].
 *                      Contains indices of atoms in the lattice.
 * @param[in] species A vector representing the species of each atom, dimension: [num_types].
 *                     Each element indicates the type of the corresponding atom.
 * @param[in] neighbor_list A 3D vector representing the neighboring atoms for each atom in each shell,
 *                          dimension: [num_atoms][num_shells][max_neighbors]. Each element contains indices of neighboring atoms.
 *
 * The function performs the following steps:
 * 1. Initialize `gamma`, `alpha`, and `count` matrices to zero.
 * 2. Loop over each atom in the lattice and its respective shells.
 * 3. For each shell, count the number of neighbors and update `gamma`, `alpha`, and `count`:
 *    \[
 *    \text{gamma}[cent][neighbor][jj] += 1
 *    \]
 *    \[
 *    \text{alpha}[cent][neighbor][jj] += \text{species}[cent] \times (1 - \text{sro}[jj]) \times \text{shell\_num} \times \left(\frac{\text{species}[neighbor]}{\text{num_atoms}}\right)
 *    \]
 *    \[
 *    \text{count}[cent][neighbor][jj] += 1
 *    \]
 * 4. This process ensures that cross-type pairs are counted while self-pairs are ignored.
 */
template <typename Integer, typename Real>
void calculate_sro_coefficient(
    std::vector<std::vector<std::vector<Real>>>& gamma,
    const Integer& num_atoms,
    const Integer& num_types,
    const Integer& num_shells,
    const std::vector<Integer>& lattice,  // Use const reference for input
    const std::vector<Integer>& species,
    const std::vector<std::vector<std::vector<Integer>>>& neighbor_list
) {
    // Initialize gamma, alpha, and count to zero for each type pair and shell
    for (Integer ii = 0; ii < num_types * num_types * num_shells; ++ii) {
        auto zz =  ii %  num_shells; // Current shell index
        auto yy = (ii /  num_shells) % num_types; // Current atom type index (y)
        auto xx =  ii / (num_types   * num_shells); // Current atom type index (x)
        gamma[xx][yy][zz] = 0.0;
    }

    // Compute pair distribution and average pair counts
    for (Integer ii = 0; ii < num_atoms; ++ii) {
        Integer atom = lattice[ii]; // Central atom index
        for (Integer jj = 0; jj < num_shells; ++jj) {
            Integer shell_size = neighbor_list[ii][jj].size(); // Number of neighbors in the shell
            for (Integer kk = 0; kk < shell_size; ++kk) {
                Integer neighbor = lattice[neighbor_list[ii][jj][kk]]; // Surrounding atom index
                gamma[atom][neighbor][jj] += 1 / (static_cast<Real>(shell_size) * num_atoms); // Increment gamma count for this pair
            }
        }
    }
}

template <typename Integer, typename Real>
Real calculate_fitness(
    const Integer& atom1, const Integer& atom2,
    std::vector<std::vector<std::vector<Real>>>& alpha,
    std::vector<std::vector<std::vector<Real>>>& gamma,
    std::vector<std::vector<std::vector<Real>>>& count,
    const Integer& num_atoms,
    const Integer& num_types,
    const Integer& num_shells,
    const std::vector<Real>& sro,
    std::vector<Integer>& lattice,  // Use const reference for input
    const std::vector<Integer>& species,
    const std::vector<std::vector<std::vector<Integer>>>& neighbor_list,
    const std::vector<Real>& weight
) {
    // Initialize gamma, alpha, and count to zero for each type pair and shell
    for (Integer ii = 0; ii < num_types * num_types * num_shells; ++ii) {
        auto zz =  ii %  num_shells; // Current shell index
        auto yy = (ii /  num_shells) % num_types; // Current atom type index (y)
        auto xx =  ii / (num_types   * num_shells); // Current atom type index (x)
        alpha[xx][yy][zz] = 0.0;
        gamma[xx][yy][zz] = 0.0;
        count[xx][yy][zz] = 0.0;
    }

    std::swap(lattice[atom1], lattice[atom2]);

    // Compute pair distribution and average pair counts
    for (Integer ii = 0; ii < num_atoms; ++ii) {
        Integer atom = lattice[ii]; // Central atom index
        for (Integer jj = 0; jj < num_shells; ++jj) {
            Integer shell_size = neighbor_list[ii][jj].size(); // Number of neighbors in the shell
            for (Integer kk = 0; kk < shell_size; ++kk) {
                Integer neighbor = lattice[neighbor_list[ii][jj][kk]]; // Surrounding atom index
                gamma[atom][neighbor][jj] += 1; // Increment the gamma count
                if (atom != neighbor) {  // Ensure we are counting cross-type pairs only
                    Real var = species[atom] * (1 - sro[jj]) * shell_size * (species[neighbor] / static_cast<Real>(num_atoms));
                    alpha[atom][neighbor][jj] += var; // Update alpha with weighted value
                    count[atom][neighbor][jj] += 1; // Increment the count for this pair
                }
            }
        }
    }

    Real error = 0.0; // Initialize total error
    for (Integer kk = 0; kk < num_shells; ++kk) { // Loop over each shell
        Real error_kk = 0.0; // Initialize shell-specific error
        for (Integer ii = 0; ii < num_types; ++ii) { // Loop over each atom type
            for (Integer jj = 0; jj < num_types; ++jj) { // Loop over each atom type
                if (ii != jj) { // Ensure we are comparing different atom types
                    // Calculate error for this pair in the shell
                    Real var = (count[ii][jj][kk] == 0) ? 0.0 : alpha[ii][jj][kk] / count[ii][jj][kk];
                    error_kk += std::pow(gamma[ii][jj][kk] - var, 2); // Accumulate squared error
                }
            }
        }
        error += weight[kk] * error_kk; // Accumulate weighted error for the shell
    }
    return std::sqrt(error);
}

/**
 * @brief Calculates the incremental change in fitness when two atoms are swapped.
 *
 * This function updates the `gamma`, `alpha`, and `count` matrices to reflect the changes
 * resulting from swapping two atoms in a specified atomic configuration.
 *
 * @param[in] atom1 The index of the first atom to be swapped.
 * @param[in] atom2 The index of the second atom to be swapped.
 * @param[in] num_atoms The total number of atoms in the system.
 * @param[in] num_types The total number of distinct atom types.
 * @param[in] num_shells The number of shells to consider for neighbor counts.
 * @param[in] species A vector representing the species of each atom, dimension: [num_types].
 *                    Each element indicates the type of the corresponding atom.
 * @param[in] neighbor_list A 3D vector representing the neighboring atoms for each atom in each shell,
 *                          dimension: [num_atoms][num_shells][max_neighbors]. Each element contains indices of neighboring atoms.
 * @param[in] sro A vector of SRO target values for each shell, dimension: [num_shells].
 * @param[in] weight A vector of weights for each shell, dimension: [num_shells].
 * @param[in,out] lattice A vector representing the current atomic configuration, dimension: [num_atoms].
 *                         Contains indices of atoms in the lattice.
 * @param[in,out] alpha A 3D vector for weighted pair counts, dimension: [num_types][num_types][num_shells].
 *                      Each element represents a weighted value based on the species and SRO.
 * @param[in,out] gamma A 3D vector representing the pair distribution counts, dimension: [num_types][num_types][num_shells].
 *                      Each element indicates the number of pairs of a specific type in a given shell.
 * @param[in,out] count A 3D vector to track the number of pairs considered, dimension: [num_types][num_types][num_shells].
 *                      Each element indicates how many pairs have been counted for the respective types in the shell.
 *
 * The function performs the following steps:
 * 1. Decrement counts in `gamma`, `alpha`, and `count` for both atoms' original neighbors.
 * 2. Swap the two atoms in the `lattice`.
 * 3. Increment counts in `gamma`, `alpha`, and `count` for both atoms' new neighbors.
 * 4. Calculate the incremental error for the affected shells using the updated matrices:
 *    \[
 *    \text{err} = \sqrt{\sum_{k=0}^{num\_shells} w_k \sum_{i=0}^{num\_types} \sum_{j=0}^{num\_types} (\text{gamma}[i][j][k] - \frac{\text{alpha}[i][j][k]}{\text{count}[i][j][k]})^2}
 *    \]
 * 5. Return the calculated error.
 */
template <typename Integer, typename Real>
Real calculate_fitness_incremental(
    std::vector<Integer>& lattice,
    std::vector<std::vector<std::vector<Real>>>& gamma,
    const Integer& atom1,
    const Integer& atom2,
    const Integer& num_atoms,
    const Integer& num_types,
    const Integer& num_shells,
    const std::vector<Integer>& species,
    const std::vector<std::vector<std::vector<Integer>>>& neighbor_list,
    const std::vector<std::vector<Real>>& target_sro,
    const std::vector<Real>& weight
) {
    // Decrement counts for atom1 and atom2
    for (Integer shell = 0; shell < num_shells; ++shell) {
        Integer atom = lattice[atom1];
        Integer atom_after_swap = lattice[atom2];
        Integer shell_size = neighbor_list[atom1][shell].size();
        for (Integer kk = 0; kk < shell_size; ++kk) {
            Integer neighbor = lattice[neighbor_list[atom1][shell][kk]];
            Integer neighbor_after_swap = neighbor;
            if (neighbor_list[atom1][shell][kk] == atom2) {
                neighbor_after_swap = lattice[atom1];
            } else if (neighbor_list[atom1][shell][kk] == atom1) {
                neighbor_after_swap = lattice[atom2];
            }

            gamma[atom][neighbor][shell] -= 1.0 / (shell_size * num_atoms);
            gamma[neighbor][atom][shell] -= 1.0 / (shell_size * num_atoms);

            gamma[atom_after_swap][neighbor_after_swap][shell] += 1.0 / (shell_size * num_atoms);
            gamma[neighbor_after_swap][atom_after_swap][shell] += 1.0 / (shell_size * num_atoms);
        }

        atom = lattice[atom2];
        atom_after_swap = lattice[atom1];
        for (Integer kk = 0; kk < shell_size; ++kk) {
            Integer neighbor = lattice[neighbor_list[atom2][shell][kk]];
            Integer neighbor_after_swap = neighbor;
            if (neighbor_list[atom2][shell][kk] == atom1) {
                neighbor_after_swap = lattice[atom2];
            } else if (neighbor_list[atom2][shell][kk] == atom2) {
                neighbor_after_swap = lattice[atom1];
            }

            gamma[atom][neighbor][shell] -= 1.0 / (shell_size * num_atoms);
            gamma[neighbor][atom][shell] -= 1.0 / (shell_size * num_atoms);
            
            gamma[atom_after_swap][neighbor_after_swap][shell] += 1.0 / (shell_size * num_atoms);
            gamma[neighbor_after_swap][atom_after_swap][shell] += 1.0 / (shell_size * num_atoms);
        }
    }

    // Swap the two atoms
    std::swap(lattice[atom1], lattice[atom2]);

    // Calculate the incremental error for affected shells
    Real error = 0.0;
    for (Integer kk = 0; kk < num_shells; ++kk) {
        Real weight_kk = weight[kk];
        Real error_kk = 0.0;
        for (Integer ii = 0; ii < num_types; ++ii) {
            for (Integer jj = 0; jj < num_types; ++jj) {
                Real sro = 1 - gamma[ii][jj][kk] / ((static_cast<Real>(species[ii]) / static_cast<Real>(num_atoms)) * (static_cast<Real>(species[jj]) / static_cast<Real>(num_atoms)));
                Real shell_sro = target_sro[kk][ii * num_types + jj];
                error_kk += (sro - shell_sro) * (sro - shell_sro);
            }
        }
        error += weight_kk * error_kk;
    }
    return std::sqrt(error);
}

/**
 * @brief Performs a local parallel Monte Carlo optimization on a set of lattices.
 *
 * This function iteratively refines multiple lattices to a problem by employing
 * the Monte Carlo method. It evaluates the fitness of lattices and attempts
 * to improve them by swapping pairs of atoms, while maintaining statistical
 * properties defined by the input parameters.
 *
 * @tparam Integer Type used for integer values (e.g., int).
 * @tparam Real Type used for floating-point values (e.g., double).
 *
 * @param[out] lattices A 2D vector containing the current lattices, where each
 *                       inner vector represents a lattice consisting of atom indices.
 *                       This is modified in place to store the final lattices.
 * @param[out] fitness A 1D vector of fitness values corresponding to each lattice.
 *                     This is modified in place to reflect updated fitness values.
 * @param[in] task The number of tasks (iterations) to perform during the optimization.
 * @param[in] depth The depth of convergence for the Monte Carlo method; controls
 *                  how many iterations will be allowed for convergence.
 * @param[in] neighbor_list A 3D vector containing the neighbors of each atom,
 *                          organized by atom index and shell index. Dimensions:
 *                          [num_atoms][num_atoms][num_shells].
 * @param[in] species A 1D vector specifying the type of each atom in the system.
 *                    Dimension: [num_types].
 * @param[in] sro A 1D vector representing short-range order parameters for the system.
 *                Dimension: [num_shells].
 * @param[in] weight A 1D vector of weights for each shell, used in fitness evaluation.
 *                   Dimension: [num_shells].
 */
template <typename Integer, typename Real>
void local_parallel_monte_carlo(
    std::vector<std::vector<Integer>>& lattices,
    std::vector<Real>& fitness,
    const Integer& task,
    const Integer& depth,
    const std::vector<std::vector<std::vector<Integer>>>& neighbor_list,
    const std::vector<Integer>& species,
    const std::vector<std::vector<Real>>& target_sro,
    const std::vector<Real>& weight
) {
    Real threshold = 0.001; // Threshold for fitness comparison

    const Integer num_atoms  = lattices[0].size();
    const Integer num_types  = species.size();
    const Integer num_shells = weight.size();

    using vec1 = std::vector<Real>; // Alias for 1D vector
    using vec2 = std::vector<std::vector<Real>>; // Alias for 2D vector
    using vec3 = std::vector<std::vector<std::vector<Real>>>; // Alias for 3D vector

    vec3 gamma     (num_types, vec2(num_types, vec1(num_shells, 0.0))); // Dimension: [num_types][num_types][num_shells]
    vec3 prev_gamma(num_types, vec2(num_types, vec1(num_shells, 0.0)));
    vec3 best_gamma(num_types, vec2(num_types, vec1(num_shells, 0.0)));

    for (size_t ii = 0; ii < lattices.size(); ++ii) {
        // std::cout << "ii = " << ii << std::endl;

        std::vector<Integer> prev = lattices[ii];
        std::vector<Integer> curr = lattices[ii];
        std::vector<Integer> best = lattices[ii];
        Real prev_fit = fitness[ii];

        Integer accumulate = 0;

        calculate_sro_coefficient(
            gamma, 
            num_atoms, num_types, num_shells, curr, species, neighbor_list);

        copy(prev_gamma, gamma);

        // mc until all threads are converged at least 10 times
        while (accumulate < depth) {
            Integer depth = 0;
            Real best_fit = 0;
            // sample with 128 tasks
            for (Integer jj = 0; jj < task; jj++) {
                curr = prev;
                Integer atom1 = std::rand() % curr.size();
                Integer atom2 = std::rand() % curr.size();

                while (curr[atom1] == curr[atom2]) {
                    atom1 = std::rand() % curr.size();
                    atom2 = std::rand() % curr.size();
                }

                copy(gamma, prev_gamma);
                // Real new_fit = calc_grs(curr, neighbor_list, species, weight, sro, alpha, gamma, count, num_atoms, num_types, num_shells, atom1, atom2);
                Real curr_fit = calculate_fitness_incremental(
                    curr, gamma,
                    atom1, atom2, num_atoms, num_types, num_shells, species, neighbor_list, target_sro, weight);
                
                if (jj == 0) {
                    best = curr;
                    best_fit = curr_fit;
                    copy(best_gamma, gamma);
                }
                else {
                    if (curr_fit < best_fit) {
                        best = curr;
                        best_fit = curr_fit;
                        copy(best_gamma, gamma);
                    }
                }
            }
            Real new_fit = best_fit;
            // Accept the new lattice if it copy better or within the threshold
            if (best_fit < prev_fit) {
                prev = best;
                fitness[ii] = best_fit;
                prev_fit = best_fit;
                copy(prev_gamma, best_gamma);
                accumulate = 0;
            }
            else {
                curr = prev;
                accumulate += 1;
                copy(gamma, prev_gamma);
            }
        }
        lattices[ii] = curr;
    }
    sort_lattices_by_fitness(fitness, lattices);
}

/**
 * @brief Executes a local parallel heuristic search (HCS) algorithm.
 *
 * This function performs a local parallel heuristic search to optimize a set of lattices.
 * It generates initial random lattices, evaluates their fitness, and iteratively improves
 * them using a Monte Carlo method until a stopping criterion is met.
 *
 * @param num_solutions The number of networks or lattices to generate.
 * @param step The maximum number of iterations to perform.
 * @param task The task identifier for the Monte Carlo method.
 * @param depth The depth parameter for the Monte Carlo method.
 * @param threshold The threshold value for the fitness score to stop the iterations.
 * @param neighbor_list A 3D vector representing the neighborhood relationships.
 * @param species A vector representing the elements or nodes in the lattices.
 * @param weight A vector representing the weights associated with the elements.
 * @return A tuple containing the optimized lattices and their corresponding fitness scores.
 */
std::tuple<std::vector<std::vector<int>>, std::vector<double>> run_local_parallel_hcs(
        const int input_num_solutions,
        const int step,
        const int task,
        const int depth,
        const double threshold,
        const std::vector<std::vector<std::vector<int>>>& neighbor_list,
        const std::vector<int>& species,
        const std::vector<double>& weight,
        const std::vector<std::vector<double>>& target_sro) 
{   
    int mpi_rank, mpi_size;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    int num_solutions = input_num_solutions / mpi_size;

    const int num_shells = weight.size();
    const int num_atoms  = std::accumulate(species.begin(), species.end(), 0);
    std::vector<std::vector<int>> lattices(num_solutions, std::vector<int>(num_atoms, 0));
    std::vector<std::vector<int>> new_solutions(num_solutions, std::vector<int>(num_atoms, 0));
    std::vector<double> sro(num_shells, 0.0);
    std::vector<double> fitness(num_solutions, 0.0);
    std::vector<double> new_fitness(num_solutions, 0.0);
    generate_random_lattices(fitness, lattices, sro, weight, species, neighbor_list, target_sro);

    // Global Search Loop
    double best_fitness = 100.0;
    for (int ii = 0; ii < step; ii++) {
        // Perpuatation: Generate new lattices randomly
        generate_random_lattices(new_fitness, new_solutions, sro, weight, species, neighbor_list, target_sro);
        // Local Search  I: Perform local parallel Monte Carlo optimization
        local_parallel_monte_carlo(new_solutions, new_fitness, task, depth, neighbor_list, species, target_sro, weight);
        // Ranking: Calculate the best lattices and update the fitness values
        calculate_best_lattices(lattices, new_solutions, fitness, new_fitness);
        // Local Search II: Perform local parallel Monte Carlo optimization
        local_parallel_monte_carlo(lattices, fitness, task, depth, neighbor_list, species, target_sro, weight);
        // Print the current iteration and best fitness

        best_fitness = fitness[0];
        MPI_Allreduce(MPI_IN_PLACE, &best_fitness, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

        if (mpi_rank == 0) {
            auto now = std::chrono::system_clock::now();
            auto now_time_t = std::chrono::system_clock::to_time_t(now);
            std::tm* now_tm = std::localtime(&now_time_t);
            std::cout << std::put_time(now_tm, "%Y-%m-%d %H:%M:%S") << " - PyHEA - INFO - " << "Iter " << ii << " with best fitness:      " << best_fitness << std::endl;
        }
    }
    return std::make_tuple(lattices, fitness);
}
    
} // namespace cpu
} // namespace accelerate