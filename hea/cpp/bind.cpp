#include <pybind11/stl.h>     // for binding STL containers
#include <pybind11/pybind11.h>

#include "accelerate.h"     // Include the header file where `calc_fits` is declared

PYBIND11_MODULE(accelerate, m) {
    m.def("run_local_parallel_hcs", &accelerate::cpu::run_local_parallel_hcs, 
        pybind11::arg("nnet"),
        pybind11::arg("step"),
        pybind11::arg("task"),
        pybind11::arg("depth"),
        pybind11::arg("thr"),
        pybind11::arg("nbor"),
        pybind11::arg("element"),
        pybind11::arg("weight"),
        R"pbdoc(
            Run the local parallel hybrid cuckoo search algorithm.

            Parameters
            ----------
            nnet : int
                The number of nests to generate.
            step : int
                The total number of atoms in each nest.
            task : int
                A vector specifying the maximum number of each atom type allowed.
            depth : int
                The total number of atoms to be distributed in each nest.
            thr : float
                The threshold for fitness comparison.
            nbor : list of list of int
                A list of lists representing the neighbors of each atom for each shell.
            element : list of int
                A list indicating the number of each type of atom.
            weight : list of float
                A list of weights used to calculate the weighted error.
            comm : MPI_Comm
                The MPI communicator.

            Returns
            -------
            tuple
                A tuple containing the lattice structures and their fitness values.
        )pbdoc"
    );

    m.def("run_local_parallel_hcs_cuda", &accelerate::gpu::run_local_parallel_hcs_cuda, 
        pybind11::arg("nnet"),
        pybind11::arg("step"),
        pybind11::arg("task"),
        pybind11::arg("depth"),
        pybind11::arg("thr"),
        pybind11::arg("nbor"),
        pybind11::arg("element"),
        pybind11::arg("weight"),
        R"pbdoc(
            Run the local parallel hybrid cuckoo search algorithm.

            Parameters
            ----------
            nnet : int
                The number of nests to generate.
            step : int
                The total number of atoms in each nest.
            task : int
                A vector specifying the maximum number of each atom type allowed.
            depth : int
                The total number of atoms to be distributed in each nest.
            thr : float
                The threshold for fitness comparison.
            nbor : list of list of int
                A list of lists representing the neighbors of each atom for each shell.
            element : list of int
                A list indicating the number of each type of atom.
            weight : list of float
                A list of weights used to calculate the weighted error.

            Returns
            -------
            tuple
                A tuple containing the lattice structures and their fitness values.
        )pbdoc"
    );

    m.def("cuda_available", &accelerate::gpu::cuda_available, 
        R"pbdoc(
            Run the local parallel hybrid cuckoo search algorithm.

            Parameters
            ----------
            nnet : int
                The number of nests to generate.
            step : int
                The total number of atoms in each nest.
            task : int
                A vector specifying the maximum number of each atom type allowed.
            depth : int
                The total number of atoms to be distributed in each nest.
            thr : float
                The threshold for fitness comparison.
            nbor : list of list of int
                A list of lists representing the neighbors of each atom for each shell.
            element : list of int
                A list indicating the number of each type of atom.
            weight : list of float
                A list of weights used to calculate the weighted error.
            comm : MPI_Comm
                The MPI communicator.

            Returns
            -------
            tuple
                A tuple containing the lattice structures and their fitness values.
        )pbdoc"
    );
}
