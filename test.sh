#!/bin/bash

## TODO: document.

# current directory
DIR=${PWD##*/}
# build directory
BUILD="build"

# head to the build directory if we're not there
if [ "$DIR" != "$BUILD" ] ; then
  # create the build directory if it's not there, and run cmake.
  if [ ! -d "$BUILD" ] ; then
    mkdir $BUILD && cd $BUILD && cmake .. && cd ..
  fi
  cd $BUILD || exit 1
fi

make benchmark

for use_shm in 0 1 ; do
  for n_partners in 2 4 8 ; do
    echo "mpirun -n 9 ./benchmark $use_shm $n_partners 1000 4096 0"
    mpirun --oversubscribe -n 9 ./benchmark $use_shm $n_partners 1000 4096 0
    EXIT_CODE=$?
    if [ "$EXIT_CODE" -ne "0" ] ; then
        exit 1
    fi
  done
done

echo "Test passed"
