#! /bin/sh

# Exit on error
set -ev

# always use gcc, just like mpich ... ?
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
export INSTALL_DIR=${INSTALL_PREFIX}/scalapack
if [ ! -d "${INSTALL_DIR}" ]; then

    # Make sure MPI is built
    ${INSTALL_PREFIX}/mpich/bin/mpichversion
    ${INSTALL_PREFIX}/mpich/bin/mpicc -show
    ${INSTALL_PREFIX}/mpich/bin/mpicxx -show
    ${INSTALL_PREFIX}/mpich/bin/mpif90 -show

    cd ${BUILD_PREFIX}
    git clone https://github.com/Reference-ScaLAPACK/scalapack.git
    cd scalapack
    git checkout 0efeeb6d2ec9faf0f2fd6108de5eda60773cdcf9 # checked revision
    cmake -H. -Bbuild_scalapack \
      -DCMAKE_C_COMPILER=$CC \
      -DCMAKE_Fortran_COMPILER=$FC \
      -DMPI_C_COMPILER=${INSTALL_PREFIX}/mpich/bin/mpicc \
      -DMPI_Fortran_COMPILER=${INSTALL_PREFIX}/mpich/bin/mpif90 \
      -DCMAKE_TOOLCHAIN_FILE="${TRAVIS_BUILD_DIR}/cmake/toolchains/travis.cmake" \
      -DCMAKE_PREFIX_PATH=${INSTALL_DIR} \
      -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
      -DBUILD_SHARED_LIBS=${BUILD_SHARED}

    cmake --build build_scalapack -j2
    cmake --build build_scalapack --target install
    find ${INSTALL_DIR} -name libscalapack.so
else
    echo "ScaLAPACK installed..."
    find ${INSTALL_DIR} -name libscalapack.so
fi
