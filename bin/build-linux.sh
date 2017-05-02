#! /bin/sh

# Exit on error
set -ev

# Environment variables
export CXXFLAGS="-mno-avx"
if [ "$CXX" = "g++" ]; then
    export CC=/usr/bin/gcc-$GCC_VERSION
    export CXX=/usr/bin/g++-$GCC_VERSION
else
    export CC=/usr/bin/clang-3.8
    export CXX=/usr/bin/clang++-3.8
fi

export MPI_HOME=$HOME/mpich
export MPICC=$MPI_HOME/bin/mpicc
export MPICXX=$MPI_HOME/bin/mpicxx
export LD_LIBRARY_PATH=/usr/lib/lapack:/usr/lib/libblas:$LD_LIBRARY_PATH

# Options for Elemental
export F77=gfortran-5
ElemOpts="-DCMAKE_Fortran_COMPILER=$F77 -DCMAKE_BUILD_TYPE=Debug -DMATH_LIBS='-lapack -lblas'"


# Configure TiledArray
mkdir _build
mkdir _install
cd _build
# TA_ERROR="throw" is the recommended way to configure TA for running unit tests
cmake .. -DCMAKE_INSTALL_PREFIX=../_install -DCMAKE_CXX_COMPILER=$CXX \
  -DCMAKE_C_COMPILER=$CC -DMPI_CXX_COMPILER=$MPICXX -DMPI_C_COMPILER=$MPICC \
  -DTA_BUILD_UNITTEST=ON -DCMAKE_BUILD_TYPE=Debug -DTA_ERROR="throw" \
  -DENABLE_ELEMENTAL=ON -Wno-dev -DMAD_ELEMENTAL_OPTIONS="$ElemOpts"

# Build all libraries, examples, and applications
make -j2 all VERBOSE=1
make install
make -j2 ta_test VERBOSE=1
cd tests
export MAD_NUM_THREADS=2
./ta_test --show_progress
