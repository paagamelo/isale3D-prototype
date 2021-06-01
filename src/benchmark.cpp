/**
 * TODO: document.
 *
 * To compile: mkdir build && cd build && cmake .. && make benchmark
 *
 * Usage: mpirun -n X ./benchmark [n_partners n_iterations message_size]
 * X = number of processes to run.
 * n_partners = number of partners for each process (either 2, 4 or 8).
 * n_iterations = number of iterations to run.
 * message_size = size of the messages in bytes.
 *
 * Bibliography:
 * [1] https://cvw.cac.cornell.edu/mpionesided/onesidedef
 * [2] https://intel.ly/3fBCeZd
 * [3] https://link.springer.com/chapter/10.1007/978-3-540-75416-9_38
 * [4] https://www.mpi-forum.org/docs/mpi-3.0/mpi30-report.pdf
 */
#include <mpi.h>
#include <iostream>
#include <algorithm>
#include <numeric>

// Some syntactic sugar.
#define TAG 0
#define COMM_WORLD MPI_COMM_WORLD
#define PERIODIC_NEXT(i, size) (((i) + 1) % (size))
#define PERIODIC_PREV(i, size) (((i) - 1 + (size)) % (size))
/// TODO: consider taking it from the command line
#define VERBOSE true

// Prints a help message explaining how to use the software.
void print_help()
{
    std::cerr << "Usage: mpirun -n X ./benchmark [n_partners n_iterations "
              << "message_size]\n";
}

// Finds the global ranks of this process' partners. Ranks are stored in the
// given array.
int find_partners(int rank, int n_processes, int* partners, int n_partners)
{
    // At the moment, only 2, 4 or 8 partners are supported.
    /// TODO: [future improvements] support arbitrary numbers of partners.
    if (n_partners != 2 && n_partners != 4 && n_partners != 8)
    {
        if (rank == 0)
            std::cerr << "Number of partners must be either 2, 4 or 8\n";
        return 1;
    }
    /// TODO: [future improvements] only periodic boundaries are supported atm.
    /// Consider supporting other types of boundary conditions.
    // In case of two partners, they are found in a one-dimensional fashion.
    if (n_partners == 2)
    {
        partners[0] = PERIODIC_PREV(rank, n_processes);
        partners[1] = PERIODIC_NEXT(rank, n_processes);
        return 0;
    }
    // In case of 4 or 8 partners, a two-dimensional decomposition is adopted.
    // A processes grid is created via MPI_Dims_create (this will create a grid
    // that is as square as possible) and partners are located in such grid.
    int dims[2] = {0, 0};
    MPI_Dims_create(n_processes, 2, dims);
    int nrows = dims[0];
    int ncols = dims[1];
    // Let us locate this process inside the grid: (n, m) is the (row, column)
    // index of this process in the grid, and is such that id = n * ncols + m.
    // In other words, we are distributing processes on the grid in a row major
    // ordered fashion.
    int n = rank / ncols;
    int m = rank % ncols;
    // We can now compute the global rank of our top, bottom, left and right
    // neighbors (partners).
    partners[0] = PERIODIC_PREV(n, nrows) * ncols + m;
    partners[1] = PERIODIC_NEXT(n, nrows) * ncols + m;
    partners[2] = n * ncols + PERIODIC_PREV(m, ncols);
    partners[3] = n * ncols + PERIODIC_NEXT(m, ncols);
    // In case of eight partners, we want corners too.
    if (n_partners == 8)
    {
        partners[4] = PERIODIC_PREV(n, nrows) * ncols + PERIODIC_PREV(m, ncols);
        partners[5] = PERIODIC_PREV(n, nrows) * ncols + PERIODIC_NEXT(m, ncols);
        partners[6] = PERIODIC_NEXT(n, nrows) * ncols + PERIODIC_PREV(m, ncols);
        partners[7] = PERIODIC_NEXT(n, nrows) * ncols + PERIODIC_NEXT(m, ncols);
    }
    return 0;
}

int main(int argc, char *argv[])
{
    // Rank of this process and total number of processes.
    int rank, n_processes;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(COMM_WORLD, &rank);
    MPI_Comm_size(COMM_WORLD, &n_processes);

    // Read configuration parameters.
    if (argc != 4)
    {
        if (rank == 0) print_help();
        MPI_Abort(COMM_WORLD, 1);
        return 1;
    }
    /// TODO: [future improvements] if the user does not input integers, atoi
    /// will fail badly. Consider checking and printing a clear error message in
    /// that case.
    int n_partners = atoi(argv[1]);
    int n_iterations = atoi(argv[2]);
    // Each process will have a buffer containing mem_size bytes of data that
    // will be sent to its partners.
    /// TODO: [future improvements] note each process sends exactly the same
    /// data to each of its neighbors. In simulation code this is not the case.
    /// It is worth considering whether sending different data (and using
    /// multiple buffers) can result in a more realistic benchmark.
    int mem_size = atoi(argv[3]);

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
    // Pointer to this process' memory segment (we will manipulate integers).
    int* mem;
    int soi = sizeof(int);
    if (mem_size % soi != 0)
    {
        if (rank == 0)
            std::cerr << "Message size must be a multiple of " << soi << "\n";
        MPI_Abort(COMM_WORLD, 1);
        return 1;
    }
    // Number of integers to exchange.
    int n_elements = mem_size / soi;
    MPI_Win_allocate_shared(mem_size, soi, MPI_INFO_NULL, shmcomm, &mem, &win);

    // Let us now find the global ranks of this process' partners: partners[i]
    // will contain the global rank of the i-th partner.
    int* partners = new int[n_partners];
    int flag = find_partners(rank, n_processes, partners, n_partners);
    if (flag)
    {
        MPI_Abort(COMM_WORLD, flag);
        return flag;
    }

    // We need to distinguish between processes located on the same node and
    // processes belonging to other nodes. To do so, let us first create a
    // global and a shared MPI group.
    MPI_Group world_group, shared_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group); 
    MPI_Comm_group(shmcomm, &shared_group);
    // We can now map global ranks to shmcomm ranks using 
    // MPI_Group_translate_ranks: partners_map[i] will contain the shmcomm rank
    // of the i-th partner, or MPI_UNDEFINED in case such partner resides on a
    // different node.
    int* partners_map = new int[n_partners];
    MPI_Group_translate_ranks(world_group, n_partners, partners, shared_group, 
                              partners_map);

    // Let us compute the number of partners on the same node and the number
    // of partners on different nodes.
    // Number of intra-node partners (on the same node).
    /// TODO: is this used?
    int n_intra_partners = 0;
    // Number of inter-node partners (on different nodes).
    int n_inter_partners = 0;
    for (int i = 0; i < n_partners; i++)
    {
        if (partners_map[i] != MPI_UNDEFINED) n_intra_partners++;
        else n_inter_partners++;
    }

    // Print partners setup.
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
        MPI_Barrier(COMM_WORLD);
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

    // Receive buffers (in a simulation code, these may be portions of the
    // actual domain of this process).
    int** rbuf = new int*[n_partners];
    for (int i = 0; i < n_partners; i++)
        rbuf[i] = new int[n_elements];

    // We will have two communications for each inter-node partner (a send and
    // a receive).
    auto* reqs = new MPI_Request[2 * n_inter_partners];

    auto start = MPI_Wtime();

    // We can now start exchanging halos. We will do that in two steps:
    // (1) each process writes its data in its shared memory segment.
    // (2) each process exchanges data with its partners. It will read its 
    // partner's shared segment in case it resides on the same node, or use a 
    // pair of non-blocking send/receive otherwise.
    // There will be a memory and a time barrier between the two steps, as
    // well as between different iterations (this mimics simulation codes).
    for (int n = 0; n < n_iterations; n++)
    {
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
        /// per-process can improve performance.
        MPI_Win_lock_all(MPI_MODE_NOCHECK, win);

        // Just fill mem with increasing integers that differ between
        // iterations.
        for (int i = 0; i < n_elements; i++)
            mem[i] = rank + i + n;

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

        // Pointer to the first request.
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

        // MPI SHM communications are completed at this point, thus we can close
        // our epoch.
        MPI_Win_unlock_all(win);

        // Ler us make sure inter-node communications finished as well.
        if (n_inter_partners > 0)
            MPI_Waitall(2 * n_inter_partners, reqs, MPI_STATUS_IGNORE);

        // Check correctness of received/read data.
        for (int i = 0; i < n_partners; i++)
            for (int j = 0; j < n_elements; j++)
                if (rbuf[i][j] != partners[i] + j + n)
                {
                    std::cerr << "iteration " << n <<", rank " << rank
                              << " found incorrect value in receive buffer for"
                              << " partner " << partners[i] << ", " << j
                              << "-th element: expected " << partners[i] + j + n
                              << ", found " << rbuf[i][j] << "\n";
                    std::cerr.flush();
                    MPI_Abort(COMM_WORLD, 1);
                    return 1;
                }

        // Before moving to the next iteration, we want processes to sync. This
        // mimics simulation codes, and is also necessary to ensure no process
        // starts updating its shared segment while another process is still
        // reading from it (note MPI_Win_unlock_all only guarantees this
        // process has completed its own read operations).
        MPI_Barrier(COMM_WORLD);
    }

    auto end = MPI_Wtime();
    auto elapsed = end - start;
    double* dts = nullptr;
    if (rank == 0) dts = new double[n_partners];
    MPI_Gather(&elapsed, 1, MPI_DOUBLE, dts, 1, MPI_DOUBLE, 0, COMM_WORLD);
    if (rank == 0)
    {
        double min_elapsed = *std::min_element(dts, dts + n_partners);
        double max_elapsed = *std::max_element(dts, dts + n_partners);
        double sum = std::accumulate(dts, dts + n_partners, 0.0);
        double avg_elapsed = sum / n_partners;
        std::cout << "\nReport\n\n"
                  << "partners: " << n_partners << "\n"
                  << "iterations: " << n_iterations << "\n"
                  << "message size: " << mem_size << "\n"
                  << "--------------------------------" << "\n"
                  << "minimum time elapsed: " << min_elapsed << " s\n"
                  << "maximum time elapsed: " << max_elapsed << " s\n"
                  << "average time elapsed: " << avg_elapsed << " s\n"
                  << "across different processes.\n";
    }

    MPI_Win_free(&win);
    delete[] partners;
    delete[] partners_map;
    delete[] partners_ptrs;
    for (int i = 0; i < n_partners; i++)
        delete[] rbuf[i];
    delete[] rbuf;
    delete[] reqs;
    if (rank == 0) delete[] dts;

    MPI_Finalize();
    return 0;
}
