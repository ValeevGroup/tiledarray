#! /bin/sh

# Exit on error
set -ev

# Environment variables
if [ "$CXX" = "g++" ]; then
    export CC=/usr/bin/gcc-$GCC_VERSION
    export CXX=/usr/bin/g++-$GCC_VERSION
    export CXXFLAGS="-mno-avx -fext-numeric-literals"
    export F77=gfortran-$GCC_VERSION
else
    export CC=/usr/bin/clang-$LLVM_VERSION
    export CXX=/usr/bin/clang++-$LLVM_VERSION
    export CXXFLAGS="-mno-avx"
    export F77=gfortran-$GCC_VERSION
fi

export MPI_HOME=$HOME/mpich
export MPICC=$MPI_HOME/bin/mpicc
export MPICXX=$MPI_HOME/bin/mpicxx
export LD_LIBRARY_PATH=/usr/lib/lapack:/usr/lib/libblas:$LD_LIBRARY_PATH

# Configure TiledArray
mkdir _build
mkdir _install
cd _build
# TA_ERROR="throw" is the recommended way to configure TA for running unit tests
cmake .. -DCMAKE_INSTALL_PREFIX=../_install -DCMAKE_CXX_COMPILER=$CXX \
  -DCMAKE_C_COMPILER=$CC -DMPI_CXX_COMPILER=$MPICXX -DMPI_C_COMPILER=$MPICC \
  -DTA_BUILD_UNITTEST=ON -DCMAKE_BUILD_TYPE=Debug -DTA_ERROR="throw" \
  -DENABLE_ELEMENTAL=ON -Wno-dev \
  -DMADNESS_CMAKE_EXTRA_ARGS="-Wno-dev;-DELEMENTAL_CMAKE_BUILD_TYPE=Debug;-DELEMENTAL_MATH_LIBS='-L/usr/lib/libblas -L/usr/lib/lapack -lblas -llapack';-DELEMENTAL_CMAKE_EXTRA_ARGS=-DCMAKE_Fortran_COMPILER=$F77"

# Build all libraries, examples, and applications
make -j2 all VERBOSE=1
make install
make -j2 ta_test VERBOSE=1
cd tests
export MAD_NUM_THREADS=2
./ta_test --show_progress
cd ..
make evd
