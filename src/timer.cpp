/*
 * Created by Lorenzo Paganelli (acse-lp320, paagamelo on GitHub).
 */
#include "timer.h"
#include <mpi.h>
#include <algorithm>

template<typename F, typename G, typename H>
double time(int rank, int n_reps, int n_iterations, F kernel, G begin, H end)
{
    // Each process will cache the average elapsed time at each repetition.
    auto* dts = new double[n_reps];

    for (int rep = 0; rep < n_reps; rep++)
    {
        double start = MPI_Wtime();

        begin(rep);
        for (int it = 0; it < n_iterations; it++)
            kernel(rep, it);
        end(rep);

        double stop = MPI_Wtime();
        // Average elapsed time on this process.
        double elapsed = (stop - start) / n_iterations;
        dts[rep] = elapsed;
    }

    // Process 0 will gather the maximum elapsed time across different processes
    // at each repetition.
    double* max_dts = nullptr;
    if (rank == 0) max_dts = new double[n_reps];

    MPI_Reduce(dts, max_dts, n_reps, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double min = -1;
    if (rank == 0)
        // Across different repetitions, we select the minimum of the biggest
        // execution times, as it's a reproducible measure.
        min = *std::min_element(max_dts, max_dts + n_reps);
    delete[] dts;
    if (rank == 0) delete[] max_dts;
    return min;
}

double time(int rank, int n_reps, int n_iterations, Computation& comp)
{
    return time(rank, n_reps, n_iterations,
                [&](int rep, int it) { comp.kernel(rep, it); },
                [&](int rep) { comp.begin(rep); },
                [&](int rep) { comp.end(rep); });
}
