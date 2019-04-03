#! /bin/sh

${TRAVIS_BUILD_DIR}/bin/build-mpich-linux.sh
${TRAVIS_BUILD_DIR}/bin/build-madness-linux.sh
${TRAVIS_BUILD_DIR}/bin/build-eigen3-linux.sh

# Exit on error
set -ev

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

# where to install TA (need for testing installed code)
export INSTALL_DIR=${INSTALL_PREFIX}/TA

# make build dir
cd ${BUILD_PREFIX}
mkdir -p TA
cd TA

# MADNESS+Elemental are build separately if $BUILD_TYPE=Debug, otherwise built as part of TA
if [ "$BUILD_TYPE" = "Debug" ]; then

  if [ "$GCC_VERSION" = 5 ]; then
    export CODECOVCXXFLAGS="-O0 --coverage"
  fi

  cmake ${TRAVIS_BUILD_DIR} \
    -DCMAKE_CXX_COMPILER=$CXX \
    -DCMAKE_C_COMPILER=$CC \
    -DMPI_CXX_COMPILER=$MPICXX \
    -DMPI_C_COMPILER=$MPICC \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DCMAKE_CXX_FLAGS="-ftemplate-depth=1024 -Wno-unused-command-line-argument ${EXTRACXXFLAGS} ${CODECOVCXXFLAGS}" \
    -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX} \
    -DTA_BUILD_UNITTEST=ON \
    -DTA_ERROR="throw" \
    -DENABLE_ELEMENTAL=ON \
    -DMADNESS_ROOT_DIR="${INSTALL_PREFIX}/madness"

else

  cmake ${TRAVIS_BUILD_DIR} \
    -DCMAKE_CXX_COMPILER=$CXX \
    -DCMAKE_C_COMPILER=$CC \
    -DMPI_CXX_COMPILER=$MPICXX \
    -DMPI_C_COMPILER=$MPICC \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DCMAKE_CXX_FLAGS="-ftemplate-depth=1024 -Wno-unused-command-line-argument ${EXTRACXXFLAGS}" \
    -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX} \
    -DTA_BUILD_UNITTEST=ON \
    -DTA_ERROR="throw" \
    -DENABLE_ELEMENTAL=ON -Wno-dev \
    -DMADNESS_CMAKE_EXTRA_ARGS="-Wno-dev;-DELEMENTAL_CMAKE_BUILD_TYPE=$BUILD_TYPE;-DELEMENTAL_MATH_LIBS='-L/usr/lib/libblas -L/usr/lib/lapack -lblas -llapack';-DELEMENTAL_CMAKE_EXTRA_ARGS=-DCMAKE_Fortran_COMPILER=$F77"

fi

# Build all libraries, examples, and applications
make -j2 all VERBOSE=1
make install

# Validate
make -j1 ta_test VERBOSE=1
export MAD_NUM_THREADS=2
setarch `uname -m` -R make check

# Build examples
make -j2 examples VERBOSE=1

# run evd example manually TODO add run_examples target
# with Debug can only use 1 thread , but since TBB is ON for Debug builds let's just skip it entirely
if [ "$BUILD_TYPE" = "Release" ]; then
  ${MPI_HOME}/bin/mpirun -n 1 examples/elemental/evd 512 64 2
  setarch `uname -m` -R ${MPI_HOME}/bin/mpirun -n 2 examples/elemental/evd 512 64 2
fi
