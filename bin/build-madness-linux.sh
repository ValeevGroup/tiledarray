#! /bin/sh

# Exit on error
set -ev

# Will build MADNESS+Elemental stand-alone for Debug builds only
if [ "$BUILD_TYPE" = "Debug" ]; then

  # Environment variables
  if [ "$CXX" = "g++" ]; then
    export CC=/usr/bin/gcc-$GCC_VERSION
    export CXX=/usr/bin/g++-$GCC_VERSION
    # Elemental needs -fext-numeric-literals when ENABLE_ELEMENTAL=ON
    export EXTRACXXFLAGS="-mno-avx -fext-numeric-literals"
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

  # where to install MADNESS+Elemental (need for testing installed code)
  export INSTALL_DIR=${INSTALL_PREFIX}/madness

  # extract the tracked tag of MADNESS
  export MADNESS_TAG=`grep 'set(TA_TRACKED_MADNESS_TAG ' ${TRAVIS_BUILD_DIR}/external/versions.cmake | awk '{print $2}' | sed s/\)//g`
  echo "required MADNESS revision = ${MADNESS_TAG}"

  # make sure installed MADNESS tag matches the required tag, if not, remove INSTALL_DIR (will cause reinstall)
  if [ -f "${INSTALL_DIR}/include/madness/config.h" ]; then
    export INSTALLED_MADNESS_TAG=`grep 'define MADNESS_REVISION' ${INSTALL_DIR}/include/madness/config.h | awk '{print $3}' | sed s/\"//g`
    echo "installed MADNESS revision = ${INSTALLED_MADNESS_TAG}"
    if [ "${MADNESS_TAG}" != "${INSTALLED_MADNESS_TAG}" ]; then
      rm -rf "${INSTALL_DIR}"
    fi
  fi

  if [ ! -d "${INSTALL_DIR}" ]; then

    # make build dir
    cd ${BUILD_PREFIX}
    mkdir -p madness
    cd madness

    # check out the tracked tag of MADNESS
    git clone https://github.com/m-a-d-n-e-s-s/madness madness_src && cd madness_src && git checkout ${MADNESS_TAG} && cd ..

    cmake madness_src \
      -DCMAKE_TOOLCHAIN_FILE="${TRAVIS_BUILD_DIR}/cmake/toolchains/travis.cmake" \
      -DCMAKE_CXX_COMPILER=$CXX \
      -DCMAKE_C_COMPILER=$CC \
      -DMPI_CXX_COMPILER=$MPICXX \
      -DMPI_C_COMPILER=$MPICC \
      -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
      -DBUILD_SHARED_LIBS=${BUILD_SHARED} \
      -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
      -DCMAKE_CXX_FLAGS="-ftemplate-depth=1024 -Wno-unused-command-line-argument ${EXTRACXXFLAGS}" \
      -DENABLE_MPI=ON \
      -DMPI_THREAD=multiple \
      -DENABLE_TBB=OFF \
      -DTBB_ROOT_DIR=/usr \
      -DENABLE_MKL=OFF \
      -DFORTRAN_INTEGER_SIZE=4 \
      -DENABLE_LIBXC=OFF \
      -DENABLE_GPERFTOOLS=OFF \
      -DASSERTION_TYPE=throw \
      -DDISABLE_WORLD_GET_DEFAULT=ON \
      -DENABLE_ELEMENTAL=ON \
      -DELEMENTAL_TAG=de7b5bea1abf5f626b91582f742cf99e2e551bff \
      -DELEMENTAL_CXXFLAGS=-Wno-deprecated-declarations \
      -DELEMENTAL_CMAKE_BUILD_TYPE=$BUILD_TYPE \
      -DELEMENTAL_CMAKE_EXTRA_ARGS=-DCMAKE_Fortran_COMPILER=$F77

    # Build MADworld + LAPACK/BLAS interface + Elemental
    make -j2 install-elemental install-madness-world install-madness-clapack install-madness-common install-madness-config VERBOSE=1
  fi

fi
