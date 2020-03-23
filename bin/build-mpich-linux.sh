#! /bin/sh

# Exit on error
set -ev

# Install packages

# always use gcc to compile MPICH, there are unexplained issues with clang (e.g. MPI_Barrier aborts)
export CC=/usr/bin/gcc-$GCC_VERSION
export CXX=/usr/bin/g++-$GCC_VERSION
export FC=/usr/bin/gfortran-$GCC_VERSION

# Print compiler information
$CC --version
$CXX --version
$FC --version

# log the CMake version (need 3+)
cmake --version

# Install MPICH unless previous install is cached ... must manually wipe cache on version bump or toolchain update
export INSTALL_DIR=${INSTALL_PREFIX}/mpich
if [ ! -d "${INSTALL_DIR}" ]; then
    cd ${BUILD_PREFIX}
    export MPICH_VERSION=3.3
    wget --no-check-certificate -q http://www.mpich.org/static/downloads/${MPICH_VERSION}/mpich-${MPICH_VERSION}.tar.gz
    tar -xzf mpich-${MPICH_VERSION}.tar.gz
    cd mpich-${MPICH_VERSION}
    ./configure FC=$FC CC=$CC CXX=$CXX --prefix=${INSTALL_DIR}
    make -j2
    make install
    ${INSTALL_DIR}/bin/mpichversion
    ${INSTALL_DIR}/bin/mpicc -show
    ${INSTALL_DIR}/bin/mpicxx -show
    ${INSTALL_DIR}/bin/mpifort -show
else
    echo "MPICH installed..."
    find ${INSTALL_DIR} -name mpiexec
    find ${INSTALL_DIR} -name mpicc
    find ${INSTALL_DIR} -name mpicxx
    find ${INSTALL_DIR} -name mpifort
fi
