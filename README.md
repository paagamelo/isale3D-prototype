## iSALE3D-prototype

This software was developed as part of Lorenzo Paganelli's ACSE-9 project.
The aim of such project is to improve the parallel performance of iSALE3D, a
well-established shock physics code. This software, which emulates some portions of
interest of iSALE3D, was developed to investigate more easily the benefits and 
drawbacks of potential optimisation strategies.

### Repository structure

- source code is located in `src` and `include` directories. 
  `src/benchmark.cpp` contains the entry point of the program and a brief
  description of the code structure. Instructions on how to compile and run the
  code are on top of  `src/benchmark.cpp`.
- `test.sh` is a simple script that runs the code multiple times with all the available options and checks
  the exit statuses. The script is designed to be used jointly with the `DEBUG_MODE`. When the code
  is compiled in debug mode (i.e., with `-DDEBUG_MODE` flag), extra checks are performed
  to ensure correct execution. If any check fails, `test.sh` will stop and warn the user.
- `jobs` contains the results of various runs of the software on [DiaL](https://www2.le.ac.uk/offices/itservices/ithelp/services/hpc/dirac).

### Enviroments

The code was developed and tested on MacOS 10.15.7 (19H2) with `gcc` 7.5.0 and Open MPI 4.1.1.
It was deployed on CentOS Linux 7 with:
- `gcc` 6.4 and Open MPI 3.0.1
- `mpiicc` (i.e., Intel compiler and MPI implementation) version 2021.2.0
- `mpiicc` version 17.0.4
