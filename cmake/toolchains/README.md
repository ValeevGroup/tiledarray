# Summary
TiledArray in general is recommended to use the toolchains distributed via [the Valeev Group CMake kit](https://github.com/ValeevGroup/kit-cmake/tree/master/toolchains). TiledArray by default downloads (via [the FetchContent CMake module](https://cmake.org/cmake/help/latest/module/FetchContent.html)) the VG CMake toolkit which makes the toolchains available without having to download the toolchain files manually. E.g., to use toolchain `x` from the VG CMake kit repository provide `-DCMAKE_TOOLCHAIN_FILE=cmake/vg/toolchains/x.cmake` to CMake when configuring TiledArray.

This directory contains compilation notes for specific high-end platform instances.

# Specific Platforms

## OLCF Summit

recommended configure script (tested 04/20/2020):

```
module purge
module load DefApps
module load cuda/10.1.243
module load essl/6.2.0-20190419
module load gcc/9.1.0
module load lsf-tools/2.0
module load netlib-lapack/3.8.0
module load spectrum-mpi/10.3.1.2-20200121
module load boost/1.66.0

export CUDA_GCC_DIR=/sw/summit/gcc/7.4.0

# clean out previous build and install artifacts ... minimally should do this:
# `rm -rf CMakeFiles/ CMakeCache.txt external cmake _deps`

cmake ../../tiledarray \
-DCMAKE_TOOLCHAIN_FILE=cmake/vg/toolchains/olcf-summit-gcc-essl.cmake \
-DTA_CUDA=ON \
-DCMAKE_CUDA_HOST_COMPILER=${CUDA_GCC_DIR}/bin/g++ \
-DCUDAToolkit_ROOT=/sw/summit/cuda/10.1.243 \
-DENABLE_TBB=OFF \
-DCMAKE_PREFIX_PATH="${EIGEN3_DIR};${CMAKE_PREFIX_PATH}" \
-DBUILD_SHARED_LIBS=OFF \
<additional CMake cache variables, such as CMAKE_INSTALL_PREFIX, etc.>
```
Note that this assumes:
- CMake 3.17.0 or later is in `PATH`; as of 4/20/2020 Summit default environment does not provide most recent CMake, hence you must install it yourself;
- Eigen3 was CMake-configured and installed into `${EIGEN3_DIR}` (set the variable appropriately). Omit the `EIGEN3` entry in `CMAKE_PREFIX_PATH` if don't have Eigen pre-installed.

## ALCF Theta

See instructions in the toolchain file `alcf-theta-mkl-tbb.cmake` (contributed by @victor-anisimov ). This should work for other generic x86-based platforms with Cray compiler wrappers.
