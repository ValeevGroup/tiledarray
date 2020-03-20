#! /bin/sh

# Exit on error
set -ev

# always use gcc to compile MPICH, there are unexplained issues with clang (e.g. MPI_Barrier aborts)
export CC=/usr/bin/gcc-$GCC_VERSION
export CXX=/usr/bin/g++-$GCC_VERSION

# Print compiler information
$CC --version
$CXX --version

# log the CMake version (need 3+)
cmake --version

# Install MPICH unless previous install is cached ... must manually wipe cache on version bump or toolchain update
export INSTALL_DIR=${INSTALL_PREFIX}/scalapack
if [ ! -d "${INSTALL_DIR}" ]; then

    # Make sure MPI is built
    ${INSTALL_DIR}/bin/mpichversion
    ${INSTALL_DIR}/bin/mpicc -show
    ${INSTALL_DIR}/bin/mpicxx -show
    ${INSTALL_DIR}/bin/mpifort -show
    


    cd ${BUILD_PREFIX}
    git clone https://github.com/Reference-ScaLAPACK/scalapack.git
    cd scalapack
    git checkout 0efeeb6d2ec9faf0f2fd6108de5eda60773cdcf9 # checked revision
    cmake -H. -Bbuild_scalapack \
      -DCMAKE_C_COMPILER=mpicc \
      -DCMAKE_TOOLCHAIN_FILE="${TRAVIS_BUILD_DIR}/cmake/toolchains/travis.cmake" \
      -DCMAKE_PREFIX_PATH=${INSTALL_DIR} \
      -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}


    make -j 2 -C build_scalapack
    make -C build_scalapack install
    find ${INSTALL_DIR} -name libscalapack.so
else
    echo "ScaLAPACK installed..."
    find ${INSTALL_DIR} -name libscalapack.so
fi
