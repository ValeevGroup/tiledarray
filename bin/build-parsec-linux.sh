#! /bin/sh

# Exit on error
set -ev

# Will build MADNESS stand-alone for Debug builds only
if [ "$BUILD_TYPE" = "Debug" ]; then

  # Environment variables
  if [ "$CXX" = "g++" ]; then
    export CC=/usr/bin/gcc-$GCC_VERSION
    export CXX=/usr/bin/g++-$GCC_VERSION
    export EXTRACXXFLAGS="-mno-avx"
    export F77=gfortran-$GCC_VERSION
  else
    export CC=/usr/bin/clang-$CLANG_VERSION
    export CXX=/usr/bin/clang++-$CLANG_VERSION
    export EXTRACXXFLAGS="-mno-avx -stdlib=libc++"
    export F77=gfortran-$GCC_VERSION
  fi

  export MPI_HOME=${INSTALL_PREFIX}/mpich
  export MPICC=$MPI_HOME/bin/mpicc
  export MPICXX=$MPI_HOME/bin/mpicxx
  export LD_LIBRARY_PATH=/usr/lib/lapack:/usr/lib/libblas:$LD_LIBRARY_PATH

  # list the prebuilt prereqs
  ls -l ${INSTALL_PREFIX}

  # where to install PaRSEC (need for testing installed code)
  export INSTALL_DIR=${INSTALL_PREFIX}/parsec

  # extract the tracked tag of PARSEC
  export PARSEC_TAG=440b29367fe98541ebe13f60deac0770f7362e30
  echo "required PARSEC revision = ${PARSEC_TAG}"

  # Should make sure installed PARSEC tag matches the required tag, if not, remove INSTALL_DIR (will cause reinstall)
  #if [ -f "${INSTALL_DIR}/include/parsec/config.h" ]; then
  #  export INSTALLED_PARSEC_TAG=`grep 'define PARSEC_REVISION' ${INSTALL_DIR}/include/parsec/config.h | awk '{print $3}' | sed s/\"//g`
  #  echo "installed PARSEC revision = ${INSTALLED_PARSEC_TAG}"
  #  if [ "${PARSEC_TAG}" != "${INSTALLED_PARSEC_TAG}" ]; then
  #    rm -rf "${INSTALL_DIR}"
  #  fi
  #fi

  if [ ! -d "${INSTALL_DIR}" ]; then

    # make build dir
    cd ${BUILD_PREFIX}
    mkdir -p parsec
    cd parsec

    # check out the tracked tag of PARSEC
    
    git clone https://bitbucket.org/schuchart/parsec.git parsec_src && cd parsec_src && git checkout ${PARSEC_TAG} && cd ..

    cmake parsec_src \
      -DCMAKE_TOOLCHAIN_FILE="${TRAVIS_BUILD_DIR}/cmake/toolchains/travis.cmake" \
      -DCMAKE_CXX_COMPILER=$CXX \
      -DCMAKE_C_COMPILER=$CC \
      -DMPI_CXX_COMPILER=$MPICXX \
      -DMPI_C_COMPILER=$MPICC \
      -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
      -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
      -DCMAKE_CXX_FLAGS="-ftemplate-depth=1024 -Wno-unused-command-line-argument ${EXTRACXXFLAGS}"

    # Build+install PaRSEC
    make -j2 install VERBOSE=1
  fi

fi
