#! /bin/sh

# control whether building shared or static libs by the parity of
# GCC_VERSION + CLANG_VERSION + "$BUILD_TYPE" == "Debug" ? 1 : 0
gccv=$GCC_VERSION
clangv=$([ "X$CLANG_VERSION" = "X" ] && echo "0" || echo "$CLANG_VERSION")
debugv=$([ "X$BUILD_TYPE" = "XDebug" ] && echo "1" || echo "0")
sharedv=$(($gccv+$clangv+$debugv))
export BUILD_SHARED=$(($sharedv % 2))

# get the most recent cmake available
if [ ! -d "${INSTALL_PREFIX}/cmake" ]; then
  CMAKE_VERSION=3.17.0
  CMAKE_URL="https://cmake.org/files/v${CMAKE_VERSION%.[0-9]}/cmake-${CMAKE_VERSION}-Linux-x86_64.tar.gz"
  mkdir ${INSTALL_PREFIX}/cmake && wget --no-check-certificate -O - ${CMAKE_URL} | tar --strip-components=1 -xz -C ${INSTALL_PREFIX}/cmake
fi
export PATH=${INSTALL_PREFIX}/cmake/bin:${PATH}
cmake --version

${TRAVIS_BUILD_DIR}/bin/build-mpich-linux.sh
${TRAVIS_BUILD_DIR}/bin/build-madness-linux.sh
${TRAVIS_BUILD_DIR}/bin/build-eigen3-linux.sh

# Exit on error
set -ev

# download latest Doxygen
if [ "$DEPLOY" = "1" ]; then
  DOXYGEN_VERSION=1.8.17
  if [ ! -d ${INSTALL_PREFIX}/doxygen-${DOXYGEN_VERSION} ]; then
    cd ${BUILD_PREFIX} && wget http://doxygen.nl/files/doxygen-${DOXYGEN_VERSION}.linux.bin.tar.gz
    cd ${INSTALL_PREFIX} && tar xzf ${BUILD_PREFIX}/doxygen-${DOXYGEN_VERSION}.linux.bin.tar.gz
  fi
  export PATH=${INSTALL_PREFIX}/doxygen-${DOXYGEN_VERSION}/bin:$PATH
  which doxygen
  doxygen --version
fi

# Environment variables
if [ "$CXX" = "g++" ]; then
    export CC=/usr/bin/gcc-$GCC_VERSION
    export CXX=/usr/bin/g++-$GCC_VERSION
    export EXTRACXXFLAGS="-mno-avx"
    # if linking statically will need fortran libs to detect liblapacke.a in BTAS
    # Elemental also needs fortran
    export F77=gfortran-$GCC_VERSION
else
    export CC=/usr/bin/clang-$CLANG_VERSION
    export CXX=/usr/bin/clang++-$CLANG_VERSION
    export EXTRACXXFLAGS="-mno-avx -stdlib=libc++"
    # if linking statically will need fortran libs to detect liblapacke.a in BTAS
    # Elemental also needs fortran
    export F77=gfortran-$GCC_VERSION
fi

export MPI_HOME=${INSTALL_PREFIX}/mpich
export MPICC=$MPI_HOME/bin/mpicc
export MPICXX=$MPI_HOME/bin/mpicxx
export LD_LIBRARY_PATH=/usr/lib/lapack:/usr/lib/libblas:$LD_LIBRARY_PATH

# list the prebuilt prereqs
ls -l ${INSTALL_PREFIX}

# where to install TA (need for testing installed code)
export INSTALL_DIR=${INSTALL_PREFIX}/TA

# make build dir
cd ${BUILD_PREFIX}
mkdir -p TA
cd TA

# if have old installed copy of TA, make sure that BTAS tag matches the required tag, if not, remove INSTALL_DIR (will cause rebuild of TA)
if [ -f "${INSTALL_DIR}/include/btas/version.h" ]; then
  export INSTALLED_BTAS_TAG=`grep 'define BTAS_REVISION' ${INSTALL_DIR}/include/btas/version.h | awk '{print $3}' | sed s/\"//g`
  echo "installed BTAS revision = ${INSTALLED_BTAS_TAG}"
  # extract the tracked tag of BTAS
  export BTAS_TAG=`grep 'set(TA_TRACKED_BTAS_TAG ' ${TRAVIS_BUILD_DIR}/external/versions.cmake | awk '{print $2}' | sed s/\)//g`
  echo "required BTAS revision = ${BTAS_TAG}"
  if [ "${BTAS_TAG}" != "${INSTALLED_BTAS_TAG}" ]; then
    rm -rf "${INSTALL_DIR}"
  fi
fi

# MADNESS+Elemental are build separately if $BUILD_TYPE=Debug, otherwise built as part of TA
if [ "$BUILD_TYPE" = "Debug" ]; then

  if [ "$COMPUTE_COVERAGE" = "1" ]; then
    export CODECOVCXXFLAGS="-O0 --coverage"
  fi

  cmake ${TRAVIS_BUILD_DIR} \
    -DCMAKE_TOOLCHAIN_FILE="${TRAVIS_BUILD_DIR}/cmake/toolchains/travis.cmake" \
    -DCMAKE_CXX_COMPILER=$CXX \
    -DCMAKE_C_COMPILER=$CC \
    -DCMAKE_Fortran_COMPILER=$F77 \
    -DMPI_CXX_COMPILER=$MPICXX \
    -DMPI_C_COMPILER=$MPICC \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    -DBUILD_SHARED_LIBS=${BUILD_SHARED} \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DCMAKE_CXX_FLAGS="-ftemplate-depth=1024 -Wno-unused-command-line-argument ${EXTRACXXFLAGS} ${CODECOVCXXFLAGS}" \
    -DCMAKE_PREFIX_PATH="${INSTALL_PREFIX}/madness;${INSTALL_PREFIX}/eigen3" \
    -DTA_BUILD_UNITTEST=ON \
    -DTA_ERROR="throw" \
    -DENABLE_ELEMENTAL=ON

else

  # if have old installed copy of TA, make sure that MADNESS tag matches the required tag, if not, remove INSTALL_DIR (will cause rebuild of MADNESS)
  if [ -f "${INSTALL_DIR}/include/madness/config.h" ]; then
    export INSTALLED_MADNESS_TAG=`grep 'define MADNESS_REVISION' ${INSTALL_DIR}/include/madness/config.h | awk '{print $3}' | sed s/\"//g`
    echo "installed MADNESS revision = ${INSTALLED_MADNESS_TAG}"
    # extract the tracked tag of MADNESS
    export MADNESS_TAG=`grep 'set(TA_TRACKED_MADNESS_TAG ' ${TRAVIS_BUILD_DIR}/external/versions.cmake | awk '{print $2}' | sed s/\)//g`
    echo "required MADNESS revision = ${MADNESS_TAG}"
    if [ "${MADNESS_TAG}" != "${INSTALLED_MADNESS_TAG}" ]; then
      rm -rf "${INSTALL_DIR}"
    fi
  fi

  cmake ${TRAVIS_BUILD_DIR} \
    -DCMAKE_TOOLCHAIN_FILE="${TRAVIS_BUILD_DIR}/cmake/toolchains/travis.cmake" \
    -DCMAKE_CXX_COMPILER=$CXX \
    -DCMAKE_C_COMPILER=$CC \
    -DCMAKE_Fortran_COMPILER=$F77 \
    -DMPI_CXX_COMPILER=$MPICXX \
    -DMPI_C_COMPILER=$MPICC \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    -DBUILD_SHARED_LIBS=${BUILD_SHARED} \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DCMAKE_CXX_FLAGS="-ftemplate-depth=1024 -Wno-unused-command-line-argument ${EXTRACXXFLAGS}" \
    -DCMAKE_PREFIX_PATH="${INSTALL_PREFIX}/eigen3" \
    -DTA_BUILD_UNITTEST=ON \
    -DTA_ERROR="throw" \
    -DENABLE_ELEMENTAL=ON -Wno-dev \
    -DMADNESS_CMAKE_EXTRA_ARGS="-Wno-dev;-DELEMENTAL_CMAKE_BUILD_TYPE=$BUILD_TYPE;-DELEMENTAL_CXXFLAGS=-Wno-deprecated-declarations;-DELEMENTAL_MATH_LIBS='-L/usr/lib/libblas -L/usr/lib/lapack -llapack -lblas';-DELEMENTAL_CMAKE_EXTRA_ARGS=-DCMAKE_Fortran_COMPILER=$F77"

fi

# Build all libraries, examples, and applications
make -j2 all VERBOSE=1
make install

# Validate
make -j1 ta_test VERBOSE=1
export MAD_NUM_THREADS=2
# to find El dep shared libs (e.g. libpmrr)
export LD_LIBRARY_PATH=${INSTALL_PREFIX}/TA/lib:${INSTALL_PREFIX}/madness/lib:${LD_LIBRARY_PATH}
make check

# Build examples
make -j2 examples VERBOSE=1

# run evd example manually TODO add run_examples target
# with Debug can only use 1 thread , but since TBB is ON for Debug builds let's just skip it entirely
if [ "$BUILD_TYPE" = "Release" ]; then
  ${MPI_HOME}/bin/mpirun -n 1 examples/elemental/evd 512 64 2
  ${MPI_HOME}/bin/mpirun -n 2 examples/elemental/evd 512 64 2
fi
