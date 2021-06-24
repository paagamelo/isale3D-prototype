/*
 * Created by Lorenzo Paganelli (acse-lp320, paagamelo on GitHub).
 */
/**
 * TODO: document.
 *
 * To compile: mkdir build && cd build && cmake .. && make benchmark
 * Or: make (for systems without cmake >= 3.10 installed)
 *
 * Usage: mpirun -n P ./benchmark [kernel_id n_reps n_iterations output_fname]
 * P = number of processes to run.
 * kernel_id = which kernel to time. 0 = GatherBcast, 1 = Allreduce,
 * 2 = AllreduceSend (see exchange_dt.h).
 * n_reps = number of times the benchmark is repeated to find the minimum
 * average elapsed time (see timer.h).
 * n_iterations = number of times the kernel is run to find the average
 * elapsed time (see timer.h).
 * output_fname = name of the output file.
 *
 * Bibliography:
 * [1] https://cvw.cac.cornell.edu/mpionesided/onesidedef
 * [2] https://intel.ly/3fBCeZd
 * [3] https://link.springer.com/chapter/10.1007/978-3-540-75416-9_38
 * [4] https://www.mpi-forum.org/docs/mpi-3.0/mpi30-report.pdf
 * [5] https://dl.acm.org/doi/10.5555/648136.748782
 * [6] https://www.mcs.anl.gov/research/projects/mpi/mpptest/
 */
#include "timer.cpp" // due to templates
#include "utils.h"
#include "exchange_dt.h"
#include <mpi.h>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <limits>

// Number of arguments the software takes.
#define N_ARGS 4 /// 7

int main(int argc, char *argv[])
{
    // Rank of this process and total number of processes.
    int rank, n_processes;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_processes);

    // Read configuration parameters.
    /// TODO: use dashed options and getopt().
    if (argc != 1 && argc != N_ARGS + 1)
    {
        if (rank == 0) print_help();
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    int kernel_id = std::stoi(argv[1]);
    int n_reps = std::stoi(argv[2]);
    int n_iterations = std::stoi(argv[3]);
    std::string output_fname = argv[4];

    Computation* comp;
    switch (kernel_id)
    {
        case 0: comp = new GatherBcast(rank, n_processes); break;
        case 1: comp = new AllReduce(rank, n_processes); break;
        case 2: comp = new AllReduceSend(rank, n_processes); break;
        default:
        {
            if (rank == 0) print_help();
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    double dt = time(rank, n_reps, n_iterations, *comp);
    if (rank == 0)
    {
        std::ofstream file;
        file.open(output_fname + ".txt", std::ios_base::app);
        if (file.fail()) std::cerr << "Can't open output file\n";
        else
        {
            file.precision(std::numeric_limits<double>::max_digits10);
            file << dt << "\n";
            file.close();
        }
        std::cout.precision(std::numeric_limits<double>::max_digits10);
        std::cout << dt << "\n";
    }

    delete comp;
    MPI_Finalize();
    return 0;
}
