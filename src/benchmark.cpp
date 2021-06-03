/**
 * TODO: document.
 *
 * To compile: mkdir build && cd build && cmake .. && make benchmark
 *
 * Usage: mpirun -n P ./benchmark [use_shm n_partners n_iterations message_size verbose]
 * P = number of processes to run.
 * use_shm = whether to use shared memory communication between processes on
 * the same node (either 0 or 1, 0 being false).
 * n_partners = number of partners for each process (either 2, 4 or 8).
 * n_iterations = number of iterations to run.
 * message_size = size of messages in bytes.
 * verbose = whether to print extra info during execution (either 0 or 1).
 * Alternatively, the software can be run with no command line argument. In this
 * case, a "config.txt" file is assumed to be located in the directory of
 * execution. Such file should contain the parameters described above in its
 * first line, as if they were specified via command line.
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
#include <fstream>
#include <sstream>
#include "utils.h"

// Some syntactic sugar.
#define TAG 0
#define COMM_WORLD MPI_COMM_WORLD
#define PERIODIC_NEXT(i, size) (((i) + 1) % (size))
#define PERIODIC_PREV(i, size) (((i) - 1 + (size)) % (size))

// Finds the global ranks of this process' partners. Ranks are stored in the
// given array. Returns 1 in case of failure, 0 in case of success.
int find_partners(int rank, int n_processes, int n_partners, int* partners)
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
    if (argc != 1 && argc != 6)
    {
        if (rank == 0) print_help();
        MPI_Abort(COMM_WORLD, 1);
        return 1;
    }
    bool use_shm, verbose;
    int n_partners, n_iterations;
    // mem_size is the message size: each process will have a buffer containing
    // mem_size bytes of data that will be sent to its partners.
    /// TODO: [future improvements] note each process sends exactly the same
    /// data to each of its neighbors. In simulation code this is not the case.
    /// It is worth considering whether sending different data (and using
    /// multiple buffers) can result in a more realistic benchmark.
    int mem_size;
    if (argc == 1)
    {
        // In case the software is run with no command line arguments, those
        // arguments are assumed to be located in the first line of a
        // "config.txt" file. Having all the processes trying to read that
        // same file may not be a good idea. Process 0 will read it and
        // broadcast its content.
        int config_params[5];
        if (rank == 0)
        {
            /// TODO: move somewhere else.
            std::ifstream file;
            file.open("config.txt");
            if (file.fail()) {
                std::cerr << "Can't open config.txt\n";
                MPI_Abort(COMM_WORLD, 1);
                return 1;
            }
            std::string line;
            getline(file, line);
            file.close();
            std::istringstream iss(line);
            for (int& param : config_params)
                iss >> param;
        }
        MPI_Bcast(config_params, 5, MPI_INT, 0, COMM_WORLD);
        use_shm = config_params[0] != 0;
        n_partners = config_params[1];
        n_iterations = config_params[2];
        mem_size = config_params[3];
        verbose = config_params[4] != 0;
    }
    if (argc == 6) // configuration parameters specified from the command line.
    {
        /// TODO: collect this code and the piece of code above somehow.
        use_shm = std::stoi(argv[1]) != 0;
        n_partners = std::stoi(argv[2]);
        n_iterations = std::stoi(argv[3]);
        mem_size = std::stoi(argv[4]);
        verbose = std::stoi(argv[5]) != 0;
    }

    // MPI_Comm_split_type will split the processes into disjointed subgroups
    // located on the same node. A new (shared memory) communicator is created
    // for each subgroup and returned in shmcomm.
    MPI_Comm shmcomm;
    if (use_shm)
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
    if (use_shm)
        MPI_Win_allocate_shared(mem_size, soi, MPI_INFO_NULL, shmcomm, &mem, &win);
    else
        mem = new int[n_elements];

    // Let us now find the global ranks of this process' partners: partners[i]
    // will contain the global rank of the i-th partner.
    int* partners = new int[n_partners];
    int flag = find_partners(rank, n_processes, n_partners, partners);
    if (flag)
    {
        MPI_Abort(COMM_WORLD, flag);
        return flag;
    }

    // We need to distinguish between processes located on the same node and
    // processes belonging to other nodes.
    int* partners_map;
    if (use_shm)
    {
        // Let us first create a global and a shared MPI group.
        MPI_Group world_group, shared_group;
        MPI_Comm_group(MPI_COMM_WORLD, &world_group);
        MPI_Comm_group(shmcomm, &shared_group);
        // We can now map global ranks to shmcomm ranks using
        // MPI_Group_translate_ranks: partners_map[i] will contain the shmcomm
        // rank of the i-th partner, or MPI_UNDEFINED in case such partner
        // resides on a different node.
        partners_map = new int[n_partners];
        MPI_Group_translate_ranks(world_group, n_partners, partners,
                                  shared_group, partners_map);
    }

    // Let us compute the number of partners residing different nodes
    // (inter-node partners).
    int n_inter_partners = n_partners;
    if (use_shm)
    {
        n_inter_partners = 0;
        for (int i = 0; i < n_partners; i++)
            if (partners_map[i] == MPI_UNDEFINED)
                n_inter_partners++;
    }

    if (verbose)
        print_partners_setup(rank, n_partners, partners, partners_map, use_shm);

    // In order to exploit shared memory, we need to find out the local address
    // of each shared memory segment. MPI_Win_shared_query will return the base
    // pointer in this process' address space of the shared segment belonging
    // to a given target rank. partners_ptrs[i] will contain that address for
    // process i.
    int** partners_ptrs;
    if (use_shm)
    {
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
    }

    // Receive buffers (in a simulation code, these may be portions of the
    // actual domain of this process).
    int** rbuf = new int*[n_partners];
    for (int i = 0; i < n_partners; i++)
        rbuf[i] = new int[n_elements];

    // We will have two communications for each inter-node partner (a send and
    // a receive).
    MPI_Request* reqs;
    if (n_inter_partners > 0)
        reqs = new MPI_Request[2 * n_inter_partners];

    /// TODO: some more precise measure?
    auto start = MPI_Wtime();

    // We can now start exchanging halos. We will do that in two steps:
    // (1) each process writes its data in its shared memory segment.
    // (2) each process exchanges data with its partners. It will read its
    // partner's shared segment in case it resides on the same node, or use a
    // pair of non-blocking send/receive otherwise.
    // Note that RMA communications must be protected somehow. According to [2],
    // passive target synchronization is the most performance-efficient
    // technique. We can define a RMA access epoch with MPI_Win_lock_all,
    // MPI_Win_unlock_all. Between the two calls, remote memory operations are
    // allowed to occur.
    // Note we lock and unlock the window outside the main loop (as in [2]).
    // There is no need to lock it at each iteration (unless we want to include
    // synchronization overhead in our benchmark on a per-iteration basis).
    // Finally, we will assert MPI_MODE_NOCHECK as no process will try to
    // acquire a conflicting lock on this window.
    /// (!) According to [3] and [4], some implementations may require memory
    /// to be allocated with MPI_Alloc_mem in order to be used for passive
    /// target RMA. However, this is not mentioned in [2], and [4] reports that
    /// this restriction is due to a complication which is absent in case of
    /// shared memory. We will thus ignore it.
    /// TODO: [future improvements] according to [2], using individual locks
    /// per-process can improve performance.
    if (use_shm) MPI_Win_lock_all(MPI_MODE_NOCHECK, win);

    for (int n = 0; n < n_iterations; n++)
    {
        // Just fill mem with increasing integers that differ between
        // iterations.
        for (int i = 0; i < n_elements; i++)
            mem[i] = rank + i + n;

        /// TODO: could start sending stuff to inter-node partners.

        // MPI_Win_sync will sync the private and public copies of the window (see
        // memory models in [4]). It is a memory barrier.
        /// TODO: comment better.
        /// TODO: [future improvements] according to [4], in case of RMA unified
        /// memory model, an update of a location in a private window in process
        /// memory becomes visible without additional RMA calls.
        if (use_shm) MPI_Win_sync(win);
        // MPI_Barrier is a time barrier.
        if (use_shm) MPI_Barrier(shmcomm);
        // According to [2], an additional memory sync may be needed on some
        // platforms.
        if (use_shm) MPI_Win_sync(win);

        /// TODO: dealing with inter-node partners beforehand would be
        /// more efficient.

        // Pointer to the first request.
        auto* rq = reqs;
        for (int i = 0; i < n_partners; i++)
        {
            // intra-node communication (same node).
            if (use_shm && partners_map[i] != MPI_UNDEFINED)
                // Memory is shared so we can load it directly.
                memcpy(rbuf[i], partners_ptrs[i], mem_size);
            // inter-node communication (different nodes).
            else
            {
                // As long as we have a waitall between iterations, we don't
                // need to use incremental tags.
                MPI_Irecv(rbuf[i], n_elements, MPI_INT, partners[i], TAG,
                          COMM_WORLD, rq++);
                MPI_Isend(mem, n_elements, MPI_INT, partners[i], TAG,
                          COMM_WORLD, rq++);
            }
        }

        // Intra-node communications are completed at this point (just memory
        // copies). Let us make sure inter-node communications finished as well.
        // This mimics simulation codes, where we wait for exchanged data before
        // manipulating it.
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

        // This time barrier between iterations is needed for correctness: it
        // ensures no process overwrites its shared segment while another
        // process is still reading it.
        if (use_shm) MPI_Barrier(shmcomm);
    }

    // Close our epoch.
    if (use_shm) MPI_Win_unlock_all(win);

    auto end = MPI_Wtime();
    auto elapsed = end - start;
    // Process 0 will gather the elapsed times on each process.
    double* dts = nullptr;
    if (rank == 0) dts = new double[n_processes];
    MPI_Gather(&elapsed, 1, MPI_DOUBLE, dts, 1, MPI_DOUBLE, 0, COMM_WORLD);
    if (rank == 0)
    {
        std::string fname = "out_" +
                            std::to_string(use_shm) + "_" +
                            std::to_string(n_partners) + "_" +
                            std::to_string(n_iterations) + "_" +
                            std::to_string(mem_size) + ".bin";
        to_file(fname, dts, n_processes);
        if (verbose) print_report(dts, n_processes);
    }

    if (use_shm) MPI_Win_free(&win);
    else delete[] mem;
    delete[] partners;
    if (use_shm) delete[] partners_map;
    if (use_shm) delete[] partners_ptrs;
    for (int i = 0; i < n_partners; i++)
        delete[] rbuf[i];
    delete[] rbuf;
    if (n_inter_partners > 0) delete[] reqs;
    if (rank == 0) delete[] dts;

    MPI_Finalize();
    return 0;
}
