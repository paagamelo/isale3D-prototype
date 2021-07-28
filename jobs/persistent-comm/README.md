## persistent communication results

In this directory you can find the results (output files) of various runs of the
kernels contained in `persistent_communication.h`. The code was run on [DiaL](https://www2.le.ac.uk/offices/itservices/ithelp/services/hpc/dirac).
The version of the code used to obtain these results is the one at commit
`f694cceb5d95f734c70096097c3a18e88a73db35`.

The `run.pbs` job loads the appropriate modules and runs the code with the `PersistentComm` kernel.
The code is run with six different message sizes, which are
typical iSALE3D message sizes.
At each run, the minimum average elapsed time is written to the output file (`peristent-comm-benchmark.oJOB_ID`).
The `run.pbs` job was submitted twice (thus generating two output files): first using intel's compiler
and mpi implementation version 2021.2.0 (`JOB_ID=879223`), then using version 17.0.4 (`JOB_ID=879511`).
