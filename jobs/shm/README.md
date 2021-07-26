## shm results

In this directory you can find the results (output files) of various runs of the
kernels contained in `shm.h`. The code was run on [DiaL](https://www2.le.ac.uk/offices/itservices/ithelp/services/hpc/dirac).
The version of the code used to obtain these results is the one at commit
`950630464a6c30fd20603b53d67be99085f3a06b`.

The `run.pbs` job loads the appropriate modules and runs the code with three different kernels:
- `PointToPoint`;
- `Shm`, with the `lock_each_iteration` flag set to 0;
- `Shm`, with the `lock_each_iteration` flag set to 1.

Each of these three kernels is run with six different message sizes, which are 
typical iSALE3D message sizes.
At each run, the minimum average elapsed time is written to the output file (`shm-benchmark.oJOB_ID`). 
The `run.pbs` job was submitted twice (thus generating two output files): first using intel's compiler
and mpi implementation version 2021.2.0 (`JOB_ID=877674`), then using version 17.0.4 (`JOB_ID=877691`).
