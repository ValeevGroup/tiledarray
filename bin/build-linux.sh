#! /bin/sh

# Exit on error
set -ev

# Environment variables
export CXXFLAGS="-mno-avx"
if [ "$CXX" = "g++" ]; then
    export CC=/usr/bin/gcc-$GCC_VERSION
    export CXX=/usr/bin/g++-$GCC_VERSION
fi
export MPICH_CC=$CC
export MPICH_CXX=$CXX
export MPICC=/usr/bin/mpicc.mpich2
export MPICXX=/usr/bin/mpicxx.mpich2
export LD_LIBRARY_PATH=/usr/lib/lapack:/usr/lib/libblas:$LD_LIBRARY_PATH

# Configure TiledArray
mkdir _build
mkdir _install
cd _build
cmake -DCMAKE_INSTALL_PREFIX=../_install -DTA_BUILD_UNITTEST=ON -DCMAKE_BUILD_TYPE=Debug ..

# Build all libraries, examples, and applications
make -j2 all VERBOSE=1
make install
make -j2 ta_test VERBOSE=1
cd tests
export MAD_NUM_THREADS=2
./ta_test --show_progress
