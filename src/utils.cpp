/*
 * Created by Lorenzo Paganelli (acse-lp320, paagamelo on GitHub).
 */
#include "utils.h"
#include <mpi.h>
#include <iostream>

/// TODO: out of date.
void print_help()
{
    std::cerr << "Usage: mpirun -n X ./benchmark [use_shm n_partners n_reps "
                 "n_iterations message_size verbose]\n"
                 "For further info see the documentation on top of "
                 "benchmark.cpp\n";
}

void print_partners_setup(int rank, int n_partners, const int* partners,
                          const int* partners_map, bool use_shm)
{
    std::cout << "rank " << rank << " has partners: [ ";
    for (int i = 0; i < n_partners; i++)
    {
        std::cout << partners[i];
        if (use_shm && partners_map[i] != MPI_UNDEFINED) std::cout << "*";
        std::cout << " ";
    }
    std::cout << "]\n";
    std::cout.flush();

    if (rank == 0 && use_shm) std::cout << "* = residing on the same node.\n";
}
