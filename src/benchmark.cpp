/**
 * TODO: document.
 *
 * To compile: mkdir build && cd build && cmake .. && make benchmark
 *
 * Usage: mpirun -n P ./benchmark [use_shm lock_each_iteration n_partners n_reps n_iterations message_size verbose]
 * P = number of processes to run.
 * use_shm = whether to use shared memory communication between processes on
 * the same node (either 0 or 1, 0 being false).
 * lock_each_iteration = whether to lock/unlock the shared window on a
 * per iteration basis (either 0 or 1, 0 meaning the window will be locked and
 * timed on a per repetition basis). Meaningful only if use_shm = 1.
 * n_partners = number of partners for each process (either 2, 4 or 8).
 * n_reps = number of times the benchmark is run to find the minimum
 * average elapsed time.
 * n_iterations = number of times halos are exchanged to find the average
 * elapsed time.
 * message_size = size of messages in bytes.
 * verbose = whether to print extra info during execution (either 0 or 1).
 * Alternatively, the software can be run with no command line argument. In this
 * case, a "config.txt" file is assumed to be located in the directory of
 * execution. Such file should contain the arguments described above in its
 * first line, as if they were specified via command line.
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
#include "utils.h"
#include <mpi.h>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <limits>

// Number of arguments the software takes.
#define N_ARGS 7

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
    int args[N_ARGS];
    if (argc == 1) // Arguments to be read from a configuration file.
    {
        // Having all the processes trying to read the same file may not be a
        // good idea. Process 0 will read it and broadcast its content.
        if (rank == 0)
        {
            std::string fname = "config.txt";
            std::ifstream file;
            file.open(fname);
            if (file.fail())
            {
                std::cerr << "Can't open " << fname << "\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
            std::string line;
            getline(file, line);
            file.close();
            std::istringstream iss(line);
            for (int& arg : args) iss >> arg;
        }
        MPI_Bcast(args, N_ARGS, MPI_INT, 0, MPI_COMM_WORLD);
    }
    else // Arguments specified from the command line.
    {
        for (int i = 0; i < N_ARGS; i++)
            // Convert arguments to int. argv[1] is the name of the executable,
            // so arguments are shifted by one.
            args[i] = std::stoi(argv[i + 1]);
    }
    bool use_shm, lock_each_iteration, verbose;
    int n_partners, n_reps, n_iterations, message_size;
    use_shm = args[0] != 0;
    lock_each_iteration = args[1] != 0;
    n_partners = args[2];
    n_reps = args[3];
    n_iterations = args[4];
    message_size = args[5];
    verbose = args[6] != 0;

    Computation* comp;
    if (use_shm)
        comp = new Shm(rank, n_processes, n_partners, message_size,
                       lock_each_iteration);
    else
        comp = new PointToPoint(rank, n_processes, n_partners, message_size);

    double dt = time(rank, n_reps, n_iterations, *comp);
    if (rank == 0)
    {
        std::string fname = "out_" +
                            std::to_string(use_shm) + "_" +
                            std::to_string(lock_each_iteration) + "_" +
                            std::to_string(n_partners) + "_" +
                            std::to_string(n_reps) + "_" +
                            std::to_string(n_iterations) + "_" +
                            std::to_string(message_size) + ".txt";
        std::ofstream file;
        file.open(fname);
        if (file.fail()) std::cerr << "Can't open output file\n";
        else
        {
            /// TODO: append to file maybe?
            file.precision(std::numeric_limits<double>::max_digits10);
            file << dt << "\n";
            file.close();
        }
        if (verbose)
        {
            std::cout.precision(std::numeric_limits<double>::max_digits10);
            std::cout << dt << "\n";
        }
    }

    delete comp;
    MPI_Finalize();
    return 0;
}
