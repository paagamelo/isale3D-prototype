/*
 * Created by Lorenzo Paganelli (acse-lp320, paagamelo on GitHub).
 */
#ifndef ISALE3D_PROTOTYPE_UTILS_H
#define ISALE3D_PROTOTYPE_UTILS_H

#include <string>

// Prints a help message explaining how to use the software.
/// TODO: document.
void print_help();

// Prints the partners setup for each process.
/// TODO: document.
void print_partners_setup(int rank, int n_partners, const int* partners,
                          const int* partners_map, bool use_shm);

#endif // HALO_EXCHANGE_BENCHMARK_UTILS_H
