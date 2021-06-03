#ifndef HALO_EXCHANGE_BENCHMARK_UTILS_H
#define HALO_EXCHANGE_BENCHMARK_UTILS_H

#include <string>

// Prints a help message explaining how to use the software.
void print_help();

/// TODO: document.
void print_partners_setup(int rank, int n_partners, const int* partners,
                          const int* partners_map, bool use_shm);

/// TODO: document.
void print_report(const double* dts, int n_processes);

// Writes the given double array to a binary file.
void to_file(const std::string& fname, const double* data, int n_elements);

#endif // HALO_EXCHANGE_BENCHMARK_UTILS_H
