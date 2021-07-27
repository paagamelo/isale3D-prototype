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
 *
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

#include "shm.h"

/**
 * A halo exchange computation using point-wise persistent communications (see,
 * e.g., [7]).
 */
struct PeristentComm : PointToPoint
{
    PeristentComm(int rank, int n_processes, int n_partners, int message_size) :
    PointToPoint(rank, n_processes, n_partners, message_size)
    {
        // pointer to the first request.
        auto* rq = reqs;
        for (int i = 0; i < n_partners; i++)
        {
            MPI_Recv_init(rbuf[i], n_elements, MPI_DOUBLE, partners[i], 0,
                           MPI_COMM_WORLD, rq++);
            MPI_Send_init(sbuf, n_elements, MPI_DOUBLE, partners[i], 0,
                           MPI_COMM_WORLD, rq++);
        }
    }

    ~PeristentComm() override
    {
        for (int i = 0; i < 2 * n_partners; i++)
            MPI_Request_free(&reqs[i]);
    }

    inline void kernel(int, int it) override
    {
#ifdef DEBUG_MODE
        fill_sbuf(it);
#endif
        MPI_Startall(2 * n_partners, reqs);
        MPI_Waitall (2 * n_partners, reqs, MPI_STATUSES_IGNORE);
#ifdef DEBUG_MODE
        for (int i = 0; i < n_partners; i++)
            check_rbuf(i, it);
#endif
    }
};


#ifndef ISALE3D_PROTOTYPE_PERSISTENT_COMMUNICATION_H
#define ISALE3D_PROTOTYPE_PERSISTENT_COMMUNICATION_H

#endif //ISALE3D_PROTOTYPE_PERSISTENT_COMMUNICATION_H
