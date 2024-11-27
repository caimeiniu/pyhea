#ifndef ACCELERATE_H
#define ACCELERATE_H


#include <vector>
#include <iomanip>

constexpr int BCC_NUM_NEIGHBORS_SHELL1 = 8;
constexpr int BCC_NUM_NEIGHBORS_SHELL2 = 6;
constexpr int BCC_NUM_NEIGHBORS_SHELL3 = 12;
constexpr int BCC_TOTAL_NEIGHBORS_PER_ATOM = BCC_NUM_NEIGHBORS_SHELL1 + BCC_NUM_NEIGHBORS_SHELL2 + BCC_NUM_NEIGHBORS_SHELL3;

constexpr int FCC_NUM_NEIGHBORS_SHELL1 = 12;
constexpr int FCC_NUM_NEIGHBORS_SHELL2 = 6;
constexpr int FCC_NUM_NEIGHBORS_SHELL3 = 12;
constexpr int FCC_TOTAL_NEIGHBORS_PER_ATOM = FCC_NUM_NEIGHBORS_SHELL1 + FCC_NUM_NEIGHBORS_SHELL2 + FCC_NUM_NEIGHBORS_SHELL3;

namespace lattice {
namespace cpu {

/**
 * @brief Executes a local parallel heuristic search (HCS) algorithm.
 *
 * This function performs a local parallel heuristic search to optimize a set of lattices.
 * It generates initial random lattices, evaluates their fitness, and iteratively improves
 * them using a Monte Carlo method until a stopping criterion is met.
 *
 * @param num_lattices The number of networks or lattices to generate.
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
        const int num_lattices,
        const int step,
        const int task,
        const int depth,
        const double threshold,
        const std::vector<std::vector<std::vector<int>>>& neighbor_list,
        const std::vector<int>& species,
        const std::vector<double>& weight);
    
} // namespace cpu
} // namespace accelerate

namespace accelerate {
namespace gpu {
/**
 * @brief Executes a local parallel heuristic search (HCS) algorithm.
 *
 * This function performs a local parallel heuristic search to optimize a set of solutions.
 * It generates initial random solutions, evaluates their fitness, and iteratively improves
 * them using a Monte Carlo method until a stopping criterion is met.
 *
 * @param num_solutions The number of networks or solutions to generate.
 * @param step The maximum number of iterations to perform.
 * @param task The task identifier for the Monte Carlo method.
 * @param depth The depth parameter for the Monte Carlo method.
 * @param threshold The threshold value for the fitness score to stop the iterations.
 * @param neighbor_list A 3D vector representing the neighborhood relationships.
 * @param species A vector representing the elements or nodes in the solutions.
 * @param weight A vector representing the weights associated with the elements.
 * @return A tuple containing the optimized solutions and their corresponding fitness scores.
 */
std::tuple<std::vector<std::vector<int>>, std::vector<double>> run_local_parallel_hcs_cuda(
        const int num_solutions,
        const int step,
        const int task,
        const int depth,
        const double threshold,
        const std::vector<std::vector<std::vector<int>>>& neighbor_list,
        const std::vector<int>& species,
        const std::vector<double>& weight);

} // namespace gpu
} // namespace accelerate

#endif // ACCELERATE_H