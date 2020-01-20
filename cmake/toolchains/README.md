# Summary
This directory contains toolchains for standard platforms and specific high-end instances.

# Specific Platforms

## OLCF Summit

recommended configure script (tested 1/16/2020):

```
module load DefApps
module load cuda/10.1.168
module load essl/6.2.0-20190419
module load gcc/9.1.0
module load cmake/3.14.2
module load lsf-tools/2.0
module load netlib-lapack/3.8.0
module load spectrum-mpi/10.3.0.1-20190611
module load boost/1.66.0

export CUDA_GCC_DIR=/sw/summit/gcc/7.4.0

# clean out previous build and install artifacts ... minimally should do this:
# `rm -rf CMakeFiles/ CMakeCache.txt external`

cmake ../../tiledarray \
-DCMAKE_TOOLCHAIN_FILE=<path to TiledArray source dir>/cmake/toolchains/olcf-summit-gcc-essl.cmake \
-DENABLE_CUDA=ON \
-DCMAKE_CUDA_HOST_COMPILER=${CUDA_GCC_DIR}/bin/g++ \
-DENABLE_TBB=OFF \
-DCMAKE_PREFIX_PATH="${HOME}/code/install/eigen;${CMAKE_PREFIX_PATH}" \
-DBUILD_SHARED_LIBS=OFF \
<additional CMake cache variables, such as CMAKE_INSTALL_PREFIX, etc.>
```
Note that this assumes that Eigen was CMake-configured and installed. Omit the `eigen` entry in `CMAKE_PREFIX_PATH` if don't have Eigen pre-installed.
