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
 * Contains kernels simulating halos exchange between processes, using different
 * techniques (e.g., point-wise communications, MPI-3 SHM). Kernels in this
 * file are mainly adapted from [2] and [3].
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
#ifndef ISALE3D_PROTOTYPE_SHM_H
#define ISALE3D_PROTOTYPE_SHM_H

#include "timer.h"
#include "setup.h"
#include <mpi.h>
#include <iostream>
#include <cassert>
#include <cstring>

/**
 * An abstract computation featuring data exchanges between neighboring MPI
 * processes (partners). A one-dimensional array of doubles is exchanged.
 */
struct HalosExchange : Computation
{
    // rank of this process.
    int rank;
    // partners[i] will contain the global rank of the i-th partner.
    int* partners;
    // number of partners.
    int n_partners;
    // sending buffer (to be initialised by subclasses).
    double* sbuf = nullptr;
    // receiving buffers: rbuf[i] is the receive buffer for partner i.
    double** rbuf;
    // number of elements (doubles) to exchange.
    int n_elements;
    // array of requests (to be initialised by subclasses).
    MPI_Request* reqs = nullptr;

    // Users can specify the message size rather than the number of values
    // to exchange.
    HalosExchange(int rank, int n_processes, int n_partners, int message_size)
    {
        this->rank = rank;
        this->n_partners = n_partners;
        this->partners = new int[n_partners];
        int failed = find_partners(rank, n_processes, n_partners, partners);
        if (failed) MPI_Abort(MPI_COMM_WORLD, 1);
        int sod = sizeof(double);
        if (message_size % sod != 0)
        {
            if (rank == 0)
                std::cerr << "Message size must be a multiple of " << sod << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        this->n_elements = message_size / sod;
        this->rbuf = new double*[n_partners];
        for (int i = 0; i < n_partners; i++)
            rbuf[i] = new double[n_elements];
    }

    ~HalosExchange() override
    {
        delete[] partners;
        for (int i = 0; i < n_partners; i++)
            delete[] rbuf[i];
        delete[] rbuf;
    }

    /**
      * Fills sbuf with incremental values. The starting value differs across
      * processes and iterations. At present, it is computed as:
      * this process' rank + the number of iterations carried out so far (it).
      */
    // The rationale here is to fill sbuf with values whose correctness can be
    // easily checked by other processes (they just need to know the global rank
    // of this process and the iteration number in order to check whether they
    // received correct data).
    virtual void fill_sbuf(int it)
    {
        for (int i = 0; i < n_elements; i++)
            sbuf[i] = rank + it + i;
    }

    /**
     * Assuming partner i filled its sbuf with fill_sbuf, checks the correctness
     * of the received data (i.e. rbuf[i]).
     */
    virtual void check_rbuf(int i, int it)
    {
        for (int j = 0; j < n_elements; j++)
            assert(rbuf[i][j] == partners[i] + it + j);
    }

    /**
     * Sets up a pair of non-blocking send and receive with partner i.
     */
    virtual void sendrecv(int i, MPI_Request* send_rq, MPI_Request* recv_rq,
                          int tag)
    {
        MPI_Irecv(rbuf[i], n_elements, MPI_DOUBLE, partners[i], tag,
                  MPI_COMM_WORLD, recv_rq);
        MPI_Isend(sbuf, n_elements, MPI_DOUBLE, partners[i], tag,
                  MPI_COMM_WORLD, send_rq);
    }
};

/**
 * A halo exchange computation using point-wise communications.
 */
struct PointToPoint : HalosExchange
{
    PointToPoint(int rank, int n_processes, int n_partners, int message_size) :
    HalosExchange(rank, n_processes, n_partners, message_size)
    {
        sbuf = new double[n_elements];
        // We will have two communications for each partner (a send and a
        // receive).
        reqs = new MPI_Request[2 * n_partners];
    }

    ~PointToPoint() override
    {
        delete[] sbuf;
        delete[] reqs;
    }

    inline void begin(int) override
    {
        // If we're not in debug mode, we fill the sending buffer once only.
        fill_sbuf(0);
    };
    inline void end(int) override { };

    inline void kernel(int, int it) override
    {
#ifdef DEBUG_MODE
        fill_sbuf(it);
#endif
        // pointer to the first request.
        auto* rq = reqs;
        for (int i = 0; i < n_partners; i++)
        {
            // As long as there is a waitall between iterations, we do not need
            // to use incremental tags across iterations.
            sendrecv(i, rq, rq + 1, 0);
            rq += 2;
        }

        MPI_Waitall(2 * n_partners, reqs, MPI_STATUSES_IGNORE);
#ifdef DEBUG_MODE
        for (int i = 0; i < n_partners; i++)
            check_rbuf(i, it);
#endif
    }
};

/**
 * A halo exchange computation using MPI-3 SHM communication between processes
 * on the same node. For further info see [2].
 */
struct Shm : HalosExchange
{
    // size of the messages exchanged in bytes.
    int message_size;
    // shared memory communicator (for processes on the same node).
    MPI_Comm shmcomm{};
    // RMA window.
    MPI_Win win{};
    // partners_map[i] will contain the shmcomm rank of the i-th partner, or
    // MPI_UNDEFINED in case such partner resides on a different node.
    int* partners_map;
    // partners_ptrs[i] will contain the base pointer in this process' address
    // space of the shared segment belonging to partner i.
    int** partners_ptrs;
    // number of partners residing different nodes (inter-node partners).
    int n_inter_partners;
    // If true, the lock/unlock synchronization primitives will be called on
    // a per-iteration basis (i.e. inside the kernel function). Otherwise,
    // they are called at each repetition (i.e. inside the begin/end setup
    // functions).
    bool lock_each_iteration;

    Shm(int rank, int n_processes, int n_partners, int message_size,
        bool lock_each_iteration) : HalosExchange(rank, n_processes, n_partners,
                                                  message_size)
    {
        this->message_size = message_size;
        this->lock_each_iteration = lock_each_iteration;
        // MPI_Comm_split_type will split the processes into disjointed subgroups
        // located on the same node. A new (shared memory) communicator is created
        // for each subgroup and returned in shmcomm.
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                            MPI_INFO_NULL, &shmcomm);
        // MPI_Win_allocate_shared will create a window object with shared memory.
        // On each process, it allocates memory of message_size bytes that is shared
        // among all processes in shmcomm. The allocated memory is contiguous
        // across process ranks, which means that the first address in the memory
        // segment of process i is consecutive with the last address in the memory
        // segment of process i−1.
        MPI_Win_allocate_shared(message_size, sizeof(double), MPI_INFO_NULL,
                                shmcomm, &sbuf, &win);
        // We need to distinguish between processes located on the same node and
        // processes belonging to other nodes. Let us first create a global and
        // a shared MPI group.
        MPI_Group world_group, shared_group;
        MPI_Comm_group(MPI_COMM_WORLD, &world_group);
        MPI_Comm_group(shmcomm, &shared_group);
        // We can now map global ranks to shmcomm ranks via MPI_Group_translate_ranks.
        partners_map = new int[n_partners];
        MPI_Group_translate_ranks(world_group, n_partners, partners,
                                  shared_group, partners_map);
        // In order to exploit shared memory, we need to find out the local address
        // of each shared memory segment. MPI_Win_shared_query will return the base
        // pointer in this process' address space of the shared segment belonging
        // to a given target rank.
        partners_ptrs = new int*[n_partners];
        // MPI_Win_shared_query returns the local size and local displacement
        // size too, but we will ignore them.
        MPI_Aint ignore_size;
        int ignore_disp_unit;
        for (int i = 0; i < n_partners; i++)
        {
            partners_ptrs[i] = nullptr;
            if (partners_map[i] != MPI_UNDEFINED)
                MPI_Win_shared_query(win, partners_map[i], &ignore_size,
                                     &ignore_disp_unit, &partners_ptrs[i]);
        }
        // Let us compute the number of inter-node partners.
        n_inter_partners = 0;
        for (int i = 0; i < n_partners; i++)
            if (partners_map[i] == MPI_UNDEFINED) n_inter_partners++;
        if (n_inter_partners > 0)
            reqs = new MPI_Request[2 * n_inter_partners];
    }

    ~Shm() override
    {
        MPI_Win_free(&win);
        delete[] partners_map;
        delete[] partners_ptrs;
        if (n_inter_partners > 0) delete[] reqs;
    }

    // Note that RMA communications must be protected somehow. According to [2],
    // passive target synchronization is the most performance-efficient
    // technique. We can define a RMA access epoch with MPI_Win_lock_all,
    // MPI_Win_unlock_all. Between the two calls, remote memory operations are
    // allowed to occur.
    /// (!) According to [3] and [4], some implementations may require memory
    /// to be allocated with MPI_Alloc_mem in order to be used for passive
    /// target RMA. However, this is not mentioned in [2], and [4] reports that
    /// this restriction is due to a complication which is absent in case of
    /// shared memory. We will thus ignore it.

    inline void begin(int) override
    {
        // We will assert MPI_MODE_NOCHECK as no process will try to acquire
        // a conflicting lock on this window.
        if (!lock_each_iteration) MPI_Win_lock_all(MPI_MODE_NOCHECK, win);
    }

    inline void end(int) override
    {
        if (!lock_each_iteration) MPI_Win_unlock_all(win);
    }

    inline void kernel(int, int it) override
    {
        // Data exchange happens in two steps:
        // (1) each process writes its data in its shared memory segment.
        // (2) each process exchanges data with its partners. It will read its
        // partner's shared segment in case it resides on the same node, or use
        // a pair of non-blocking send/receive otherwise.

        if (lock_each_iteration) MPI_Win_lock_all(MPI_MODE_NOCHECK, win);

        // This kernel essentially behaves like `PointToPoint`, except it uses
        // MPI-3 SHM when possible. The two kernels are designed to be compared
        // against each other. Thus, one thing which is worth noting is that
        // `PointToPoint` fills and checks the send/recv buffers on a per
        // iteration basis in debug mode only. This kernel, on the other hand,
        // fills the sending buffer at each iteration in any case. This is
        // because filling the shared window with data is a fundamental part of
        // one-sided communications (see, e.g., [1]), which we need to take into
        // account when timing the kernel, in order to have a fair comparison.

        // Direct write to shared memory segment.
        fill_sbuf(it);

        // MPI_Win_sync will sync the private and public copies of the window (see
        // memory models in [4]). It is a memory barrier, needed to make sure
        // the memory write above is publicly visible.
        MPI_Win_sync(win);
        // A time barrier: we don't want to read other processes' memory
        // segments while they are still writing to it.
        MPI_Request barrier_req;
        MPI_Ibarrier(shmcomm, &barrier_req);

        // While we wait for processes to reach the barrier above, we set up
        // inter-node communications.
        auto* rq = reqs;
        for (int i = 0; i < n_partners; i++)
            if (partners_map[i] == MPI_UNDEFINED)
            {
                // As long as there is a waitall between iterations, we do not
                // need to use incremental tags across iterations.
                sendrecv(i, rq, rq + 1, 0);
                rq += 2;
            }

        MPI_Wait(&barrier_req, MPI_STATUS_IGNORE);
        // According to [2], an additional memory sync may be needed on some
        // platforms.
        MPI_Win_sync(win);

        // We will now deal with intra-node partners.
        for (int i = 0; i < n_partners; i++)
            if (partners_map[i] != MPI_UNDEFINED)
                // Memory is shared so we can directly read from it.
                memcpy(rbuf[i], partners_ptrs[i], message_size);

        // This time barrier ensures no process overwrites its shared segment
        // while another process is still reading it.
        MPI_Ibarrier(shmcomm, &barrier_req);

        // No more shared memory operations below.
        if (lock_each_iteration) MPI_Win_unlock_all(win);

#ifdef DEBUG_MODE
        // Intra-node communications are completed at this point (just memory
        // copies). So we can check the correctness of the data we just read.
        for (int i = 0; i < n_partners; i++)
            if (partners_map[i] != MPI_UNDEFINED)
                check_rbuf(i, it);
#endif

        // Now we make sure inter-node communications finished as well.
        if (n_inter_partners > 0)
            MPI_Waitall(2 * n_inter_partners, reqs, MPI_STATUS_IGNORE);

#ifdef DEBUG_MODE
        for (int i = 0; i < n_partners; i++)
            if (partners_map[i] == MPI_UNDEFINED)
                check_rbuf(i, it);
#endif

        MPI_Wait(&barrier_req, MPI_STATUS_IGNORE);
    }
};

#endif // ISALE3D_PROTOTYPE_SHM_H
