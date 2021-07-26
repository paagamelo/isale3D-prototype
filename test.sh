#!/bin/bash

# This script runs all the possible kernels with different arguments and checks
# the exit status of each run.

make clean
make

echo "mpirun --oversubscribe -n 10 ./benchmark GatherBcast 10 100 dummy"
mpirun --oversubscribe -n 10 ./benchmark GatherBcast 10 100 dummy
EXIT_CODE=$?
if [ "$EXIT_CODE" -ne "0" ] ; then
  exit 1
fi
echo "mpirun --oversubscribe -n 10 ./benchmark AllReduce 10 100 dummy"
mpirun --oversubscribe -n 10 ./benchmark AllReduce 10 100 dummy
EXIT_CODE=$?
if [ "$EXIT_CODE" -ne "0" ] ; then
  exit 1
fi

for n_partners in 2 4 8 ; do
  echo "mpirun --oversubscribe -n 9 ./benchmark PointToPoint 10 100 dummy 4096 $n_partners"
  mpirun --oversubscribe -n 9 ./benchmark PointToPoint 10 100 dummy 4096 $n_partners
  EXIT_CODE=$?
  if [ "$EXIT_CODE" -ne "0" ] ; then
      exit 1
  fi
done

for lock_each_iteration in 0 1 ; do
  for n_partners in 2 4 8 ; do
    echo "mpirun --oversubscribe -n 9 ./benchmark Shm 10 100 dummy 4096 $n_partners $lock_each_iteration"
    mpirun --oversubscribe -n 9 ./benchmark Shm 10 100 dummy 4096 $n_partners $lock_each_iteration
    EXIT_CODE=$?
    if [ "$EXIT_CODE" -ne "0" ] ; then
        exit 1
    fi
  done
done

make clean
rm dummy.txt

echo "Test passed"
