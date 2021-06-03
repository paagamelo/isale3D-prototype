#include "utils.h"
#include <mpi.h>
#include <iostream>
#include <algorithm>
#include <numeric>

void print_help()
{
    std::cerr << "Usage: mpirun -n X ./benchmark [use_shm n_partners "
              << "n_iterations message_size verbose]\n"
              << "For further info see the documentation on top of "
              << "benchmark.cpp\n";
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

void print_report(const double* dts, int n_processes)
{
    double min_elapsed = *std::min_element(dts, dts + n_processes);
    double max_elapsed = *std::max_element(dts, dts + n_processes);
    double sum = std::accumulate(dts, dts + n_processes, 0.0);
    double avg_elapsed = sum / n_processes;
    std::cout << "--------------------------------\n"
              << "Report\n"
              << "minimum time elapsed: " << min_elapsed << " s\n"
              << "maximum time elapsed: " << max_elapsed << " s\n"
              << "average time elapsed: " << avg_elapsed << " s\n";
}

void to_file(const std::string& fname, const double* data, int n_elements)
{
    // Writing a binary file means we don't lose any precision. The only
    // drawback is that we have to pay attention to the byte order (signedness,
    // endianness and size of a C double) when reading data at post-processing
    // time, but numpy helps a lot.
    FILE* fout = fopen(fname.c_str(), "wb");
    if (fout == nullptr)
    {
        std::cerr << "Can't open " << fname << "\n";
        return;
    }
    int written = fwrite(data, sizeof(double), n_elements, fout);
    if (written != n_elements)
    {
        std::cerr << "Couldn't write to " << fname << "\n";
        return;
    }
    fclose(fout);
}
