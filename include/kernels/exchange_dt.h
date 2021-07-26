/*
 * Created by Lorenzo Paganelli (acse-lp320, paagamelo on GitHub).
 *
 * Bibliography:
 * [1] https://cvw.cac.cornell.edu/mpionesided/onesidedef
 * [2] https://intel.ly/3fBCeZd
 * [3] https://link.springer.com/chapter/10.1007/978-3-540-75416-9_38
 * [4] https://www.mpi-forum.org/docs/mpi-3.0/mpi30-report.pdf
 * [5] https://dl.acm.org/doi/10.5555/648136.748782
 * [6] https://www.mcs.anl.gov/research/projects/mpi/mpptest/
 * [7] https://cvw.cac.cornell.edu/MPIP2P/percomm
 */
/**
 * Contains kernels emulating iSALE3D timestep exchange. At each iteration,
 * iSALE3D processes compute their local limiting timestep and its location
 * (the 3D index of the cell it comes from). This info is then exchanged and
 * the global limiting timestep is selected and used. For further info, see the
 * iSALE manual.
 */
/*
 * Kernels are inlined to (try to) reduce the number of function calls, so as to
 * mitigate the overhead deriving from passing the kernel as argument to the
 * `time` function (see comments in timer.h).
 * Inlining is complex, but on average produces faster code. See C++ core
 * guidelines for more:
 * - https://bit.ly/2MQZypM
 * - https://bit.ly/3oL8F91
 * Note inlining requires functions definitions to be placed in header files,
 * this is why kernels are not placed in a dedicated .cpp file.
 *
 * Part of the comment above is taken from my ACSE-5 coursework:
 * https://github.com/acse-2020/group-project-most-vexing-parse
 */
#ifndef ISALE3D_PROTOTYPE_EXCHANGE_DT_H
#define ISALE3D_PROTOTYPE_EXCHANGE_DT_H

#include <random>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <limits>

// The maximum value an x-, y- or z-coordinate can assume.
#define MAX_COORD 100

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
    // index of the cell containing the limiting timestep.
    int x = -1, y = -1, z = -1;

    ExchangeDt(int rank, int n_processes)
    {
        this->rank = rank;
        this->n_processes = n_processes;
    }

    /**
     * Randomly initialise dt, x, y and z.
     */
    void rand_init(int seed)
    {
        std::mt19937 gen(seed);
        // real distribution in [0, 1).
        std::uniform_real_distribution<double> real_distribution(0.0, 1.0);
        // int distribution in [0, MAX_COORD].
        std::uniform_int_distribution<int> int_distribution(0, MAX_COORD);
        dt = real_distribution(gen);
        x = int_distribution(gen);
        y = int_distribution(gen);
        z = int_distribution(gen);
#ifdef DEBUG_MODE
        std::cout << "rank " << rank << " " << dt << " at " << x << ", " << y
                  << ", " << z << "\n";
        std::cout.flush();
#endif
    }

    inline void begin(int) override { }
    inline void end(int) override { }
};

/**
 * Uses a MPI_Gather followed by a MPI_Bcast. This replicates iSALE3D behavior
 * on 23/06/21.
 */
struct GatherBcast : ExchangeDt
{
    GatherBcast(int rank, int n_processes) : ExchangeDt(rank, n_processes)
    {
        // initialise data with a seed which differs between processes.
        rand_init(31 * rank);
    }

    inline void kernel(int, int) override
    {
        double tmp[4];
        tmp[0] = dt;
        tmp[1] = x;
        tmp[2] = y;
        tmp[3] = z;

        double* tmps = nullptr;
        if (rank == 0) tmps = new double[4  * n_processes];

        MPI_Gather(tmp, 4, MPI_DOUBLE, tmps, 4, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        double min_dt = -1;
        if (rank == 0)
        {
            min_dt = tmps[0];
            for (int i = 1; i < n_processes; i++)
                if (tmps[4 * i] < min_dt) min_dt = tmps[4 * i];
        }
        MPI_Bcast(&min_dt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#ifdef DEBUG_MODE
        std::cout << "rank " << rank << " min dt " << min_dt << "\n";
#endif
        if (rank == 0) delete[] tmps;
    }
};

/**
 * Helper struct for packing and unpacking 3D integer indexes into doubles.
 */
struct Packer
{
    // See the constructor.
    double dscale;
    double dscale2;
    long long int iscale;

    /**
      * Creates a packer given the maximum value a packable integer can assume.
      * The minimum value is always 0.
      */
    Packer(int max)
    {
        // The scale is an integer value used to pack/unpack 3 integers into a
        // single natural number. Given three integers x, y, z, they are packed
        // using the following formula:
        // packed = x + y * scale + z * scale * scale
        // Multiplying an integer by the scale has the effect of shifting it,
        // so the three integers are stored in separate groups of bits.
        // In order for this to work, the scale must be strictly bigger than
        // any value a packable integer can assume.
        int scale = max + 1;
        dscale = (double)scale;
        dscale2 = dscale * dscale;
        iscale = (long long int)scale;
        // The result of the formula above is a (big) natural number, and we are
        // going to store it into a double. In order to correctly unpack a
        // packed value, we need to make sure it is stored exactly. The biggest
        // integer such that all smaller integers can be stored exactly in a
        // IEEE 754 double is 2^53 (this follows from the fact that IEEE 754
        // doubles have 52+1 bits of mantissa). The biggest packed value is:
        // max + max * scale + max * scale * scale
        // If this is smaller than 2^53, we can correctly pack and unpack any
        // value in [0, max].
        static_assert(std::numeric_limits<double>::is_iec559, "IEEE 754 required");
        assert(max + max * dscale + max * dscale2 < pow(2, 53));
    }

    inline double pack(int x, int y, int z) const
    {
        return x + y * dscale + z * dscale2;
    }

    inline void unpack(double packed, int coords[3]) const
    {
        // When unpacking our value, we need to use exact-precision integer
        // division and modulo. However, our packed value can take up to 53
        // bits, so we need long long ints (64 bits).
        long long int ipacked = (long long int)packed;
        // we carry out exact operations with long long int precision.
        long long int c1 = ipacked % iscale;
        long long int c2 = (ipacked / iscale) % iscale;
        long long int c3 = ipacked / (iscale * iscale);
        // and then convert the results to int (we know they can fit an int as
        // they were integers originally).
        coords[0] = (int)c1;
        coords[1] = (int)c2;
        coords[2] = (int)c3;
    }
};

/**
 * Uses a MPI_Allreduce with MPI_MINLOC operator (see [4] section 5.9.4). The
 * x, y, z location of the local limiting timestep is packed in a single value
 * and exchanged in the MPI_Allreduce. All processes receive both the minimum
 * timestep and its location.
 */
struct AllReduce : ExchangeDt
{
    // See [4] section 5.9.4.
    struct
    {
        double dt;
        int index;
    } sbuf, rbuf;
    Packer packer = Packer(MAX_COORD);

    AllReduce(int rank, int n_processes) : ExchangeDt(rank, n_processes)
    {
        // initialise data with a seed which differs between processes.
        rand_init(31 * rank);
        // When using the MPI_MINLOC operator, there is a difference between
        // C++ and Fortran (note iSALE is written in Fortran):
        // - in fortran, indices and values have to be coerced to the same type
        // (i.e., double).
        // - In C++, indices can only be integers. See [4] 5.9.4.
        // This means that in fortran we would pack our 3D index into a double,
        // whereas in C++ we would pack it into an integer. Since we want to
        // replicate the computation we would carry out in fortran, we will pack
        // 3D indices into doubles here. However, we will cast the packed values
        // to int and effectively send integers rather than doubles. This is the
        // best we can do.
        // Since we are using integers under the hood, we need to make sure we
        // can correctly pack/unpack 3D indices in a single int (for further
        // info, see the comments on Packer's constructor).
        auto max_packed = MAX_COORD * (1 + packer.dscale + packer.dscale2);
        auto int_max = std::numeric_limits<int>::max();
        assert(max_packed <= int_max);
    }

    inline void kernel(int, int) override
    {
        sbuf.dt = dt;
        // See the comments in the constructor.
        sbuf.index = (int)packer.pack(x, y, z);

        MPI_Allreduce(&sbuf, &rbuf, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);

        int coords[3];
        packer.unpack(rbuf.index, coords);
#ifdef DEBUG_MODE
        std::cout << "rank " << rank << " min dt " << rbuf.dt << " at "
                  << coords[0] << ", " << coords[1] << ", " << coords[2] << "\n";
#endif
    }
};

#endif // ISALE3D_PROTOTYPE_EXCHANGE_DT_H
