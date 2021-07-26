/*
 * Created by Lorenzo Paganelli (acse-lp320, paagamelo on GitHub).
 */
#ifndef ISALE3D_PROTOTYPE_SETUP_H
#define ISALE3D_PROTOTYPE_SETUP_H

/**
 * Finds the global ranks of this process' partners. At the moment, only 2, 4
 * or 8 partners are supported. In case of two partners, they are found in a
 * one-dimensional fashion (i.e., they are the processes with global rank one
 * higher and one lower of this process' rank). In case of 4 or 8 partners,
 * a two-dimensional decomposition is adopted: processes are mapped to a 2D
 * grid and partners are located in such grid. They are the top, bottom, left
 * and right neighbors in case of 4 partners, whereas the four corner neighbors
 * are included in case of 8 partners. Periodic boundaries are assumed in any
 * case.
 *
 * @param rank - rank of this process.
 * @param n_processes - total number of processes.
 * @param n_partners - number of partners (either 2, 4 or 8).
 * @param partners - global ranks of this process' partners are stored in this
 * array. The caller is responsible for allocating enough space.
 * @return 0 in case of success, 1 in case of failure.
 */
int find_partners(int rank, int n_processes, int n_partners, int* partners);

#endif // ISALE3D_PROTOTYPE_SETUP_H
