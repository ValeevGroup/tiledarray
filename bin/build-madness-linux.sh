#! /bin/sh

# Exit on error
set -ev

# Will build MADNESS+Elemental stand-alone for Debug builds only
if [ "$BUILD_TYPE" = "Debug" ]; then

  # Environment variables
  if [ "$CXX" = "g++" ]; then
    export CC=/usr/bin/gcc-$GCC_VERSION
    export CXX=/usr/bin/g++-$GCC_VERSION
    export EXTRACXXFLAGS="-mno-avx -fext-numeric-literals"
    export F77=gfortran-$GCC_VERSION
  else
    export CC=/usr/bin/clang-5.0
    export CXX=/usr/bin/clang++-5.0
    export EXTRACXXFLAGS="-mno-avx"
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

  if [ ! -d "${INSTALL_DIR}" ]; then

    # make build dir
    cd ${BUILD_PREFIX}
    mkdir -p madness
    cd madness

    git clone https://github.com/m-a-d-n-e-s-s/madness madness_src

    cmake madness_src \
      -DCMAKE_CXX_COMPILER=$CXX \
      -DCMAKE_C_COMPILER=$CC \
      -DMPI_CXX_COMPILER=$MPICXX \
      -DMPI_C_COMPILER=$MPICC \
      -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
      -DBUILD_SHARED_LIBS=OFF \
      -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
      -DCMAKE_CXX_FLAGS="-ftemplate-depth=1024 -Wno-unused-command-line-argument ${EXTRACXXFLAGS}" \
      -DENABLE_MPI=ON \
      -DMPI_THREAD=multiple \
      -DENABLE_TBB=ON \
      -DTBB_ROOT_DIR=/usr \
      -DENABLE_MKL=OFF \
      -DFORTRAN_INTEGER_SIZE=4 \
      -DENABLE_LIBXC=OFF \
      -DLAPACK_LIBRARIES='-L/usr/lib/libblas -L/usr/lib/lapack -lblas -llapack' \
      -DENABLE_GPERFTOOLS=OFF \
      -DASSERTION_TYPE=throw \
      -DDISABLE_WORLD_GET_DEFAULT=ON \
      -DENABLE_ELEMENTAL=ON \
      -DELEMENTAL_TAG=50e24bb92b6b2e30d657decda3a903d967f00f0b \
      -DELEMENTAL_CMAKE_BUILD_TYPE=$BUILD_TYPE \
      -DELEMENTAL_CMAKE_EXTRA_ARGS=-DCMAKE_Fortran_COMPILER=$F77

    # Build all libraries, examples, and applications
    make -j2 install-elemental install-world install-clapack install-common install-config VERBOSE=1
  fi

fi
