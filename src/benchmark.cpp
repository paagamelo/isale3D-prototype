/*
 * Created by Lorenzo Paganelli (acse-lp320, paagamelo on GitHub).
 */
/**
 * TODO: document.
 *
 * To compile with cmake: mkdir build && cd build && cmake .. && make benchmark
 * To compile without cmake: make
 *
 * Usage: mpirun -n P ./benchmark [kernel n_reps n_iterations outname]
 * P = number of processes to run.
 * kernel = name of the kernel to run (see exchange_dt.h and kernels.h).
 * n_reps = number of times the benchmark is repeated to find the minimum
 * average elapsed time (see timer.h).
 * n_iterations = number of times the kernel is run to find the average
 * elapsed time (see timer.h).
 * outname = name of the output file.
 * Extra arguments needed if kernel == PointToPoint (to be provided in the
 * following order):
 * message_size = size of messages in bytes.
 * n_partners = number of partners for each process (either 2, 4 or 8).
 * Extra arguments needed if kernel == Shm (to be provided in the following
 * order):
 * message_size = see above.
 * n_partners = see above.
 * lock_each_iteration = whether to lock/unlock the shared window on a
 * per *iteration* basis (either 0 or 1: 0 meaning the window will be locked and
 * timed on a per *repetition* basis, 1 on a per *iteration* basis).
 *
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
#include "kernels.h"
#include "exchange_dt.h"

#include <mpi.h>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <limits>

void arg_error(int rank)
{
    if (rank == 0) std::cerr << "Inconsistent arguments. See benchmark.cpp\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
}

int main(int argc, char *argv[])
{
    // Rank of this process and total number of processes.
    int rank, n_processes;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_processes);

    // Read configuration parameters.
    /// TODO: consider using dashed options and getopt().
    if (argc < 5) arg_error(rank);

    std::string kernel = argv[1];
    int n_reps = std::stoi(argv[2]);
    int n_iterations = std::stoi(argv[3]);
    std::string outname = argv[4];
    Computation* comp;

    if (kernel == "GatherBcast")
        comp = new GatherBcast(rank, n_processes);
    else if (kernel == "AllReduce")
        comp = new AllReduce(rank, n_processes);
    else
    {
        if (argc < 7) arg_error(rank);

        int message_size = std::stoi(argv[5]);
        int n_partners = std::stoi(argv[6]);

        if (kernel == "PointToPoint")
            comp = new PointToPoint(rank, n_processes, n_partners, message_size);
        else if (kernel == "Shm")
        {
            if (argc < 8) arg_error(rank);
            bool lock_each_iteration = std::stoi(argv[7]) != 0;
            comp = new Shm(rank, n_processes, n_partners, message_size,
                           lock_each_iteration);
        }
    }

    double dt = time(rank, n_reps, n_iterations, *comp);
    if (rank == 0)
    {
        std::ofstream file;
        file.open(outname + ".txt", std::ios_base::app);
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
