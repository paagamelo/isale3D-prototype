#include "setup.h"
#include <iostream>
#include <mpi.h>

#define PERIODIC_NEXT(i, size) (((i) + 1) % (size))
#define PERIODIC_PREV(i, size) (((i) - 1 + (size)) % (size))

int find_partners(int rank, int n_processes, int n_partners, int* partners)
{
    /// TODO: [future improvements] support more of partners.
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
