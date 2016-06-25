#! /bin/sh

# Exit on error
set -ev

# Environment variables
export CXXFLAGS="-mno-avx"
if [ "$CXX" = "g++" ]; then
    export CC=/usr/bin/gcc-$GCC_VERSION
    export CXX=/usr/bin/g++-$GCC_VERSION
else
    export CC=/usr/bin/clang-3.7
    export CXX=/usr/bin/clang++-3.7
fi
export MPICC=$HOME/mpich/bin/mpicc
export MPICXX=$HOME/mpich/bin/mpicxx
export LD_LIBRARY_PATH=/usr/lib/lapack:/usr/lib/libblas:$LD_LIBRARY_PATH

# Configure TiledArray
mkdir _build
mkdir _install
cd _build
cmake .. -DCMAKE_INSTALL_PREFIX=../_install -DCMAKE_CXX_COMPILER=$CXX -DCMAKE_C_COMPILER=$CC \
      -DMPI_CXX_COMPILER=$MPICXX -DMPI_C_COMPILER=$MPICC -DTA_BUILD_UNITTEST=ON \
      -DCMAKE_BUILD_TYPE=Debug -DENABLE_ELEMENTAL=ON

# Build all libraries, examples, and applications
make -j2 all VERBOSE=1
make install
make -j2 ta_test VERBOSE=1
cd tests
export MAD_NUM_THREADS=2
./ta_test --show_progress
