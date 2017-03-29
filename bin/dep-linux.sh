#! /bin/sh

# Exit on error
set -ev

# Install packages

# always use gcc to compile MPICH, there are unexplained issues with clang (e.g. MPI_Barrier aborts)
export CC=/usr/bin/gcc-$GCC_VERSION
export CXX=/usr/bin/g++-$GCC_VERSION

# Print compiler information
$CC --version
$CXX --version

# log the CMake version (need 3+)
cmake --version

# Install MPICH
if [ ! -d "${HOME}/mpich" ]; then
    wget --no-check-certificate -q http://www.mpich.org/static/downloads/3.2/mpich-3.2.tar.gz
    tar -xzf mpich-3.2.tar.gz
    cd mpich-3.2
    ./configure CC="ccache $CC" CXX="ccache $CXX" --disable-fortran --disable-romio --prefix=${HOME}/mpich
    make -j2
    make install
    ${HOME}/mpich/bin/mpichversion
    ${HOME}/mpich/bin/mpicc -show
    ${HOME}/mpich/bin/mpicxx -show
else
    echo "MPICH installed..."
    find ${HOME}/mpich -name mpiexec
    find ${HOME}/mpich -name mpicc
    find ${HOME}/mpich -name mpicxx
fi
