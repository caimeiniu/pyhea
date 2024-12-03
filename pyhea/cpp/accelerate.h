#ifndef ACCELERATE_H
#define ACCELERATE_H

#include <vector>
#include <iomanip>

namespace accelerate {
namespace cpu {
/**
 * @brief Executes a local parallel heuristic search (HCS) algorithm on CPU.
 *
 * @param num_lattices Number of networks or lattices to generate
 * @param step Maximum number of iterations
 * @param task Task identifier for Monte Carlo method
 * @param depth Depth parameter for Monte Carlo method
 * @param threshold Threshold value for fitness score
 * @param neighbor_list Neighborhood relationships (3D vector)
 * @param species Elements or nodes in lattices
 * @param weight Weights associated with elements
 * @return std::tuple<std::vector<std::vector<int>>, std::vector<double>> 
 *         Optimized lattices and their fitness scores
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

#ifdef USE_CUDA
namespace gpu {
/**
 * @brief Executes a local parallel heuristic search (HCS) algorithm on GPU.
 *
 * @param num_solutions Number of networks or solutions to generate
 * @param step Maximum number of iterations
 * @param task Task identifier for Monte Carlo method
 * @param depth Depth parameter for Monte Carlo method
 * @param threshold Threshold value for fitness score
 * @param neighbor_list Neighborhood relationships (3D vector)
 * @param species Elements or nodes in solutions
 * @param weight Weights associated with elements
 * @return std::tuple<std::vector<std::vector<int>>, std::vector<double>>
 *         Optimized solutions and their fitness scores
 */
std::tuple<std::vector<std::vector<int>>, std::vector<double>> run_local_parallel_hcs_cuda(
    const int num_solutions,
    const int step,
    const int task,
    const int depth,
    const double threshold,
    const std::vector<std::vector<std::vector<int>>>& neighbor_list,
    const std::vector<int>& species,
    const std::vector<double>& weight,
    const std::vector<std::vector<double>>& target_sro);

/**
 * @brief Check if CUDA is available at runtime
 * @return bool True if CUDA is available
 */
bool cuda_available();
} // namespace gpu
#endif // USE_CUDA

} // namespace accelerate

#endif // ACCELERATE_H