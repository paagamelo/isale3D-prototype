## exchange-dt results

In this directory you can find the results (output files) of various runs of the 
kernels contained in `exchange_dt.h`. The code was run on [DiaL](https://www2.le.ac.uk/offices/itservices/ithelp/services/hpc/dirac).
The version of the code used to obtain these results is the one at commit 
`ff9211e60aa845a140d8c0a02cb2597666e9f42f`.

The `Makefile` submits the `run.pbs` job on an increasing number of nodes in `[1, 7]`. 
The `run.pbs` job loads the appropriate modules (either gcc 6.4 and openmpi 3.0.1, 
or intel's compiler and mpi implementation, both version 2021.2.0), then runs the 
code two times: first with the `GatherBcast` kernel, then with the `AllReduce` kernel. 
At the end of each run, the minimum average elapsed time is written in a text file named 
`scaling-N.txt`, where `N` is the number of nodes the job ran on. If you look in those 
text files, you will see four elapsed times: the first two were produced using gcc and 
openmpi, the second two were produced using intel's products. The first one and the 
third one refer to the `GatherBcast` kernel. The second one and fourth one refer to 
the `AllReduce` kernel.
