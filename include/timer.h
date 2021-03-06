/*
 * Created by Lorenzo Paganelli (acse-lp320, paagamelo on GitHub).
 *
 * Bibliography:
 * [1] https://cvw.cac.cornell.edu/mpionesided/onesidedef
 * [2] https://intel.ly/3fBCeZd
 * [3] https://link.springer.com/chapter/10.1007/978-3-540-75416-9_38
 * [4] https://www.mpi-forum.org/docs/mpi-3.0/mpi30-report.pdf
 * [5] https://dl.acm.org/doi/10.5555/648136.748782
 * [6] https://www.mcs.anl.gov/research/projects/mpi/mpptest/
 * [7] https://cvw.cac.cornell.edu/MPIP2P/percomm
 */
#ifndef ISALE3D_PROTOTYPE_TIMER_H
#define ISALE3D_PROTOTYPE_TIMER_H

/**
 * Times a given MPI kernel. The kernel is run for a given number of iterations
 * and the average elapsed time is computed on each process. The biggest elapsed
 * time is then selected, as suggested in [2]. The rationale is that the biggest
 * execution time is typically the one that guides the execution flow in MPI.
 * This is repeated for a given number of repetitions, and the minimum time
 * across different repetitions is selected and returned. As explained in [5],
 * the minimum between different runs is typically the only reproducible
 * measure, being less sensitive to perturbations.
 * The user can provide two setup functions that will be called at the beginning
 * and at the end of each repetition, and will be included in the overall timing
 * (their contribution will be inverse to the number of iterations). This
 * feature was introduced to reproduce the experiments described in [2].
 * Note the returned value is meaningful on process 0 only.
 *
 * @tparam F - type of kernel function: void(int, int). Takes in the number of
 * repetitions and the number of iterations carried out so far.
 * @tparam G - type of setup functions: void(int). Takes in the number of
 * repetitions carried out so far.
 * @tparam H - same as G.
 * @param rank - rank of this process.
 * @param n_processes - total number of processes.
 * @param n_reps - number of repetitions to run. Theoretically, the higher this
 * is the more reliable the results are, and the more costly the benchmark is in
 * terms of wall clock time. However, high values for n_reps are typically not
 * needed. The default value used in [6] is 30.
 * @param n_iterations - number of iterations to run. This should be high enough
 * that executing the kernel this many times does not take a time quantity which
 * is close to clock precision. Similarly to n_reps, this controls the tradeoff
 * between reliability and cost of the benchmark.
 * @param kernel - kernel to time.
 * @param begin - setup function to call at the beginning of each repetition.
 * @param end - callback function to call at the end of each repetition.
 * @return at each repetition, the biggest average elapsed time across different
 * processes is selected. The minimum of these biggest times is returned.
 * Meaningful on process 0 only.
 */
// Taking in the kernel as argument, as opposed to a hard-coded kernel,
// typically involves an overhead. But allows for a much more flexible code
// architecture.
template<typename F, typename G, typename H>
double time(int rank, int n_reps, int n_iterations, F kernel, G begin, H end);

/**
 * A computation to be timed with functionalities defined in this file
 * (timer.h).
 */
// This is explicitly designed to work jointly with the time function defined
// above. Using objects (as opposed to top-level functions) allows various
// kernels to have minimal, identical signatures. Different parameters needed by
// different kernels are declared as class' attributes.
struct Computation
{
    /*
     * The piece of code below (mentioning and linking C++ core guidelines)
     * is adapted from my ACSE-5 coursework:
     * https://github.com/acse-2020/group-project-most-vexing-parse
     */

    Computation() = default;
    // Since this is a polymorphic class (i.e. a base class), we'll suppress
    // copying and moving as it can lead to slicing. For further info see:
    // - http://bit.ly/3ptNVUb (C++ core guideline C.67: A polymorphic class
    // should suppress copying)
    Computation(const Computation&) = delete;             // copy ctor
    Computation& operator=(const Computation&) = delete;  // copy assignment
    Computation(Computation&&) = delete;                  // move ctor
    Computation& operator=(Computation&&) = delete;       // move operator
    // We'll then have a virtual non-throwing destructor, for further info see:
    // - http://bit.ly/3ptVBps (C++ core guideline C.37: Make destructors
    // noexcept)
    // - http://bit.ly/2NBxebq (C++ core guideline C.35: A base class
    // destructor should be either public and virtual, or protected and
    // non-virtual)
    virtual ~Computation() noexcept = default;

    // See the documentation of of `time` for further info on the functions
    // below.

    virtual void begin(int rep) = 0;
    virtual void kernel(int rep, int it) = 0;
    virtual void end(int rep) = 0;
};

/**
 * See <F, G, H> time(int, int, int, int, F, G, H). This function is a wrapper
 * around that, using a Computation object.
 */
double time(int rank, int n_reps, int n_iterations, Computation& comp);

#endif // ISALE3D_PROTOTYPE_TIMER_H
