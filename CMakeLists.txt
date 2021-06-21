cmake_minimum_required(VERSION 3.10)

project(isale_benchmark)

# set up compiler
set(CMAKE_CPP_COMPILER mpic++)
SET(CPPFLAGS "-Wall -Wpedantic -Wextra -std=c++17 -O3 -Ofast -march=native")
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${CPPFLAGS}")

add_executable(benchmark src/benchmark.cpp src/setup.cpp src/utils.cpp)

target_include_directories(benchmark PRIVATE include)

# set up mpi requirement
find_package(MPI REQUIRED)

target_include_directories(benchmark PUBLIC ${MPI_INCLUDE_PATH})
target_link_libraries(benchmark ${MPI_LIBRARIES})

if(MPI_COMPILE_FLAGS)
  set_target_properties(benchmark PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(benchmark PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()
