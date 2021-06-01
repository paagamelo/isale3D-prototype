/**
 * TODO: document.
 * 
 * Bibliography:
 * [1] https://cvw.cac.cornell.edu/mpionesided/onesidedef
 * [2] https://intel.ly/3fBCeZd
 * [3] https://link.springer.com/chapter/10.1007/978-3-540-75416-9_38
 * [4] https://www.mpi-forum.org/docs/mpi-3.0/mpi30-report.pdf
 */
#include <iostream>
#include <mpi.h>

// Some syntactic sugar.
#define COMM_WORLD MPI_COMM_WORLD
#define TAG 0

/// TODO: document.
#define VERBOSE true

/// TODO: document.
// Adapted from [2] and [3].
int main(int argc, char *argv[])
{
    // rank of this process and total number of processes.
    int rank, n_processes;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(COMM_WORLD, &rank);
    MPI_Comm_size(COMM_WORLD, &n_processes);

    // MPI_Comm_split_type will split the processes into disjointed subgroups
    // located on the same node. A new (shared memory) communicator is created
    // for each subgroup and returned in shmcomm.
    MPI_Comm shmcomm;
    MPI_Comm_split_type(COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, 
                        &shmcomm);

    // MPI_Win_allocate_shared will create a window object with shared memory. 
    // On each process, it allocates memory of mem_size bytes that is shared 
    // among all processes in shmcomm. The allocated memory is contiguous
    // across process ranks, which means that the first address in the memory 
    // segment of process i is consecutive with the last address in the memory 
    // segment of process i−1.
    /// TODO: [future improvements] according to [4], setting the info key 
    /// "alloc_shared_noncontig" to true, the implementation can allocate the 
    /// memory requested by each process in a location that is close to this 
    /// process, improving performance.
    MPI_Win win;
    // pointer to this process' memory segment.
    int* mem;
    // number of elements (integers) to exchange.
    /// TODO: read this from user
    int n_elements = 1;
    int soi = sizeof(int);
    // size of "mem" in bytes.
    /// TODO: read this from user
    int mem_size = n_elements * soi;
    MPI_Win_allocate_shared(mem_size, soi, MPI_INFO_NULL, shmcomm, &mem, &win);

    // We need to distinguish between processes located on the same node and
    // processes belonging to other nodes. To do so, let us first create a
    // global and a shared MPI groups.
    MPI_Group world_group, shared_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group); 
    MPI_Comm_group(shmcomm, &shared_group);
    // We can now map global ranks to shmcomm ranks using 
    // MPI_Group_translate_ranks.
    /// TODO: read this from user
    int n_partners = 2;
    // partners[i] will contain the global rank of the i-th partner.
    int* partners = new int[n_partners];
    /// TODO: consider 4, 8 partners too
    partners[0] = (rank - 1 + n_processes) % n_processes;
    partners[1] = (rank + 1) % n_processes;
    // partners_map[i] will contain the shmcomm rank of the i-th partner, or
    // MPI_UNDEFINED in case such partner resides on a different node.
    int* partners_map = new int[n_partners];
    MPI_Group_translate_ranks(world_group, n_partners, partners, shared_group, 
                              partners_map);

    // Let us compute the number of partners on the same node and the number
    // of partners on different nodes.
    // Number of intra-node partners (on the same node).
    int n_intra_partners = 0;
    // Number of inter-node partners (on different nodes).
    int n_inter_partners = 0;
    for (int i = 0; i < n_partners; i++)
    {
        if (partners_map[i] != MPI_UNDEFINED)
            n_intra_partners++;
        else
            n_inter_partners++;
    }

    if (VERBOSE)
    {
        std::cout << "rank " << rank << " has neighbors: [ ";
        for (int i = 0; i < n_partners; i++)
        {
            std::cout << partners[i];
            if (partners_map[i] != MPI_UNDEFINED) std::cout << "*";
            std::cout << " ";
        }
        std::cout << "]\n";
        std::cout.flush();

        if (rank == 0) std::cout << "* = residing on the same node.\n";
    }

    // In order to exploit shared memory, we need to find out the local address
    // of each shared memory segment. MPI_Win_shared_query will return the base 
    // pointer in this process' address space of the shared segment belonging 
    // to a given target rank. partners_ptrs[i] will contain that address for
    // process i.
    int** partners_ptrs = new int*[n_partners];
    // MPI_Win_shared_query returns the local size and local displacement size
    // too, but we will ignore them.
    MPI_Aint ignore_size;
    int ignore_disp_unit;
    for (int i = 0; i < n_partners; i++)
    {
        partners_ptrs[i] = nullptr;
        if (partners_map[i] != MPI_UNDEFINED)
            MPI_Win_shared_query(win, partners_map[i], &ignore_size,
                                 &ignore_disp_unit, &partners_ptrs[i]);
    }

    // We can now start exchaning halos. We will do that in two steps:
    // (1) each process writes its data in its shared memory segment.
    // (2) each process exchanges data with its partners. It will read its 
    // partner's shared segment in case it resides on the same node, or use a 
    // pair of non-blocking send and receive otherwise.
    // There will be a memory and a time barrier between the two steps. This is
    // adapted from [2].

    // Let us start the communication. According to [2], passive target
    // synchronization is the most performance-efficient. We can define a RMA
    // access epoch with MPI_Win_lock_all, MPI_Win_unlock_all. Between the two
    // calls, remote memory operations are allowed to occur. We will assert
    // MPI_MODE_NOCHECK as no process will try to acquire a conflicting lock
    // on this window.
    /// (!) According to [3] and [4], some implementations may require memory 
    /// to be allocated with MPI_Alloc_mem in order to be used for passive 
    /// target RMA. However, this is not mentioned in [2], and [4] reports that
    /// this restriction is due to a complication which is absent in case of
    /// shared memory. We will thus ignore it.
    /// TODO: [future improvements] according to [2], using individual locks
    /// per process can improve performance.
    MPI_Win_lock_all(MPI_MODE_NOCHECK, win);

    // For the time being, we simply write each process' rank in its shared 
    // segment (via a direct write).
    /// TODO: do something more sophisticated.
    mem[0] = rank;
    
    // MPI_Win_sync will sync the private and public copies of the window (see
    // memory models in [4]). It is a memory barrier.
    /// TODO: [future improvements] according to [4], in case of RMA unified 
    /// memory model, an update of a location in a private window in process 
    /// memory becomes visible without additional RMA calls.
    MPI_Win_sync(win);
    // MPI_Barrier is a time barrier.
    MPI_Barrier(shmcomm);
    // According to [2], an additional memory sync may be needed on some
    // platforms.
    MPI_Win_sync(win);

    /// TODO: move to setup.
    int** rbuf = new int*[n_partners];
    for (int i = 0; i < n_partners; i++)
        rbuf[i] = new int[n_elements];

    /// TODO: move to setup.
    /// TODO: why the 2?
    auto* reqs = new MPI_Request[2 * n_inter_partners];
    auto* rq = reqs;
    
    for (int i = 0; i < n_partners; i++) 
    {
        // intra-node communication (same node).
        if (partners_map[i] != MPI_UNDEFINED) 
            // Memory is shared so we can load it directly.
            memcpy(rbuf[i], partners_ptrs[i], mem_size);
        // inter-node communication (different nodes).
        else 
        {
            MPI_Irecv(rbuf[i], n_elements, MPI_INT, i, TAG, COMM_WORLD, rq++);
            MPI_Isend(&rank, n_elements, MPI_INT, i, TAG, COMM_WORLD, rq++);
        }
    }

    /// TODO: comment.
    if(n_inter_partners > 0)
        MPI_Waitall(2 * n_inter_partners, reqs, MPI_STATUS_IGNORE);

    /// TODO: comment.
    MPI_Win_unlock_all(win);
    MPI_Win_free(&win);

    if (VERBOSE)
    {
        std::cout << "rank " << rank << " has read: [ ";
        for (int i = 0; i < n_partners; i++)
            std::cout << rbuf[i][0] << " ";
        std::cout << "]\n";
        std::cout.flush();
    }

    /// TODO: deletes missing.
    
    MPI_Finalize();
    return 0;
}
