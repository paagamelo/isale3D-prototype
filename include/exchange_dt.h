/*
 * Created by Lorenzo Paganelli (acse-lp320, paagamelo on GitHub).
 */
/**
 * Contains kernels emulating iSALE timestep exchange.
 */
#ifndef ISALE3D_PROTOTYPE_EXCHANGE_DT_H
#define ISALE3D_PROTOTYPE_EXCHANGE_DT_H

#include <random>
#include <algorithm>

/**
 * Abstract class containing common attributes and functionalities.
 */
struct ExchangeDt : Computation
{
    // rank of this process.
    int rank;
    // total number of processes.
    int n_processes;
    // limiting timestep to be exchanged.
    double dt = -1;

    ExchangeDt(int rank, int n_processes)
    {
        this->rank = rank;
        this->n_processes = n_processes;
    }

    /**
     * Randomly initialise dt.
     */
    void rand_init(int seed)
    {
        std::mt19937 gen(seed);
        // real distribution in [0, 1).
        std::uniform_real_distribution<double> real_distribution(0.0, 1.0);
        dt = real_distribution(gen);
    }

    inline void begin(int) override { }
    inline void end(int) override { }
};

/**
 * Uses a gather followed by a broadcast.
 */
struct GatherBcast : ExchangeDt
{
    GatherBcast(int rank, int n_processes) : ExchangeDt(rank, n_processes)
    {
        // initialise dt once only, with a seed which differs between processes.
        rand_init(31 * rank);
    }

    inline void kernel(int, int) override
    {
        double* dts = nullptr;
        if (rank == 0) dts = new double[n_processes];

        MPI_Gather(&dt, 1, MPI_DOUBLE, dts, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        double min = -1;
        if (rank == 0) min = *std::min_element(dts, dts + n_processes);
        MPI_Bcast(&min, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) delete[] dts;
    }
};

struct AllReduce : ExchangeDt
{
    AllReduce(int rank, int n_processes) : ExchangeDt(rank, n_processes)
    {
        // initialise dt once only, with a seed which differs between processes.
        rand_init(31 * rank);
    }

    inline void kernel(int, int) override
    {
        double min = -1;
        MPI_Allreduce(&dt, &min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    }
};

#endif // ISALE3D_PROTOTYPE_EXCHANGE_DT_H
