#! /bin/sh

# Exit on error
set -ev

# Install packages

# Environment variables
if [ "$CXX" = "g++" ]; then
  export CC=/usr/bin/gcc-$GCC_VERSION
  export CXX=/usr/bin/g++-$GCC_VERSION
  export EXTRACXXFLAGS="-mno-avx"
else
  export CC=/usr/bin/clang-$CLANG_VERSION
  export CXX=/usr/bin/clang++-$CLANG_VERSION
  export EXTRACXXFLAGS="-mno-avx  -stdlib=libc++"
fi

# Print compiler information
$CC --version
$CXX --version

# log the CMake version (need 3+)
cmake --version

# Install Eigen3 unless previous install is cached ... must manually wipe cache on version bump or toolchain update
export INSTALL_DIR=${INSTALL_PREFIX}/eigen3
if [ ! -d "${INSTALL_DIR}" ]; then
    cd ${BUILD_PREFIX}
    wget -q https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.bz2
    tar -xjf eigen-3.3.7.tar.bz2
    cd eigen-*
    mkdir build
    cd build
    cmake .. -DCMAKE_CXX_COMPILER=$CXX \
      -DCMAKE_C_COMPILER=$CC \
      -DCMAKE_CXX_FLAGS="${EXTRACXXFLAGS}" \
      -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}
    make install
else
    echo "Eigen3 already installed ..."
fi
