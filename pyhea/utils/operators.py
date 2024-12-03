import numpy as np

def selective_sort(nests, fitness):
    """
    @brief Sorts the nests and fitness arrays based on the fitness values.
    
    This function sorts the nests and fitness arrays in ascending order of fitness values.
    It uses numpy for efficient sorting.
    
    @param nests A list of lists where each sublist represents a nest.
    @param fitness A list of fitness values corresponding to each nest.
    
    @return A tuple containing two lists:
        - sorted_nests: The nests sorted based on fitness values.
        - sorted_fitness: The fitness values sorted in ascending order.
    
    @details
    The function performs the following steps:
    1. Convert the input lists to numpy arrays.
    2. Use numpy's argsort function to get the indices that would sort the fitness array.
    3. Use these indices to sort both the nests and fitness arrays.
    
    Formula:
    \f[
    \text{sorted\_indices} = \text{np.argsort}(\text{fitness})
    \]
    \f[
    \text{sorted\_nests} = \text{nests}[\text{sorted\_indices}]
    \]
    \f[
    \text{sorted\_fitness} = \text{fitness}[\text{sorted\_indices}]
    \]
    """
    # assert len(nests) == len(fitness), "The number of nests and fitness values should be the same."
    # Convert lists to numpy arrays
    nests = np.array(nests)
    fitness = np.array(fitness)
    
    # Get sorted indices based on fitness
    sorted_indices = np.argsort(fitness)
    
    # Sort nests and fitness using the sorted indices
    sorted_nests = nests[sorted_indices]
    sorted_fitness = fitness[sorted_indices]
    
    return sorted_nests.tolist(), sorted_fitness.tolist()


# rename the nests to some more meaningful name