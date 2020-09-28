# Synopsis

```.cpp
$ git clone https://github.com/ValeevGroup/TiledArray.git tiledarray
$ cd tiledarray
$ cmake -B build \
    -D CMAKE_INSTALL_PREFIX=/path/to/tiledarray/install \
    -D CMAKE_TOOLCHAIN_FILE=cmake/vg/toolchains/<toolchain-file-for-your-platform>.cmake \
    .
$ cmake --build build
(recommended, but optional): $ cmake --build build --target check
$ cmake --build build --target install
```

# Prerequisites

- C++ compiler with support for the [C++17 standard](http://www.iso.org/standard/68564.html), or a more recent standard. This includes the following compilers:
  - [GNU C++](https://gcc.gnu.org/), version 7.0 or higher
  - [Clang](https://clang.llvm.org/), version 5 or higher
  - [Apple Clang](https://en.wikipedia.org/wiki/Xcode), version 9.3 or higher
  - [Intel C++ compiler](https://software.intel.com/en-us/c-compilers), version 19 or higher

  See the current [Travis CI matrix](.travis.yml) for the most up-to-date list of compilers that are known to work.

- [CMake](https://cmake.org/), version 3.15 or higher; if CUDA support is needed, CMake 3.18 or higher is required.
- [Git](https://git-scm.com/) 1.8 or later (required to obtain TiledArray and MADNESS source code from GitHub)
- [Eigen](http://eigen.tuxfamily.org/), version 3.3 or higher; if CUDA is enabled then 3.3.7 is required (will be downloaded automatically, if missing)
- [Boost libraries](www.boost.org/), version 1.33 or higher (will be downloaded automatically, if missing). The following principal Boost components are used:
  - Boost.Iterator: header-only
  - Boost.Container: header-only
  - Boost.Test: header-only or (optionally) as a compiled library, *only used for unit testing*
  - Boost.Range: header-only, *only used for unit testing*
- [BTAS](http://github.com/BTAS/BTAS), tag f5a50ebf11b5cdc219ff4c06e3652eea2cd6c697 (will be downloaded automatically, if missing)
- BLAS library
- [MADNESS](https://github.com/m-a-d-n-e-s-s/madness), tag 0add54a549e509b00925c8d14fcba9546c9d60e5 .
  Only the MADworld runtime and BLAS/LAPACK C API component of MADNESS is used by TiledArray.
  If usable MADNESS installation is not found, TiledArray will download and compile
  MADNESS. *This is the recommended way to compile MADNESS for all users*.
  A detailed list of MADNESS prerequisites can be found at [MADNESS' INSTALL file](https://github.com/m-a-d-n-e-s-s/madness/blob/master/INSTALL_CMake);
  it also also contains detailed
  MADNESS compilation instructions.

  Compiling MADNESS requires the following prerequisites:
  - An implementation of Message Passing Interface version 2 or 3, with support
    for `MPI_THREAD_MULTIPLE`.
  - BLAS and LAPACK libraries (only BLAS is used by TiledArray, but without LAPACK MADNESS will not compile)
  - (optional)
    Intel Thread Building Blocks (TBB), available in a [commercial](software.intel.com/tbb‎) or
    an [open-source](https://www.threadingbuildingblocks.org/) form

Optional prerequisites:
- [CUDA compiler and runtime](https://developer.nvidia.com/cuda-zone) -- for execution on CUDA-enabled accelerators. CUDA 11 or later is required. Support for CUDA also requires the following additional prerequisites, both of which will be built and installed automatically if missing:
  - [cuTT](github.com/ValeevGroup/cutt) -- CUDA transpose library; note that our fork of the [original cuTT repo](github.com/ap-hynninen/cutt) is required to provide thread-safety (tag 0e8685bf82910bc7435835f846e88f1b39f47f09).
  - [Umpire](github.com/LLNL/Umpire) -- portable memory manager for heterogeneous platforms (tag f04abd1dd038c84262915a493d8f78576bb80fd0).
- [Doxygen](http://www.doxygen.nl/) -- for building documentation (version 1.8.12 or later).
- [ScaLAPACK](http://www.netlib.org/scalapack/) -- a distributed-memory linear algebra package. If detected, the following C++ components will also be sought and downloaded, if missing:
  - [blacspp](https://github.com/wavefunction91/blacspp.git) -- a modern C++ (C++17) wrapper for BLACS (tag 88076f1706be083ead882f6ce0bfc6884a72fc03)
  - [scalapackpp](https://github.com/wavefunction91/scalapackpp.git) -- a modern C++ (C++17) wrapper for ScaLAPACK (tag 28433942197aee141cd9e96ed1d00f6ec7b902cb)
- Python3 interpreter -- to test (optionally-built) Python bindings
- [Range-V3](https://github.com/ericniebler/range-v3.git) -- a Ranges library that served as the basis for Ranges component of C++20; only used for some unit testing of the functionality anticipated to be supported by future C++ standards.

Most of the dependencies (except for MADNESS) can be installed with a package manager,
such as Homebrew on OS X or apt-get on Debian Linux distributions;
this is the preferred method. Since on some systems configuring
and building MADNESS can be difficult even for experts, we recommend letting the
TiledArray download and build MADNESS for you.

# Obtain TiledArray source code

Check out the source code as follows:

```
$ git clone https://github.com/ValeevGroup/tiledarray.git
```

It is necessary to compile TiledArray outside of the source code tree. Most users can simply create
a build directory inside the source tree:

```
$ cd tiledarray; mkdir build; cd build
```

Instructions below assume that you are located in the build directory. We will assume that
the environment variable `TILEDARRAY_SOURCE_DIR` specifies the location of the
TiledArray source tree.

# Configure TiledArray

TiledArray is configured and built with CMake. When configuring with CMake, you specify a set
of CMake variables on the command line, where each variable argument is prepended with the '-D'
option. Typically, you will need to specify the install path for TiledArray,
build type, MPI Compiler wrappers, and BLAS and LAPACK libraries.

For many platforms TiledArray provides *toolchain* files that can greatly simplify configuration;
they are located under `$TILEDARRAY_SOURCE_DIR/cmake/toolchains`.
It is strongly recommended that all users use one of the provided toolchains as is or as the
basis for a custom toolchain file.
Here's how to compile TiledArray on a macOS system:

```
$ cmake -D CMAKE_INSTALL_PREFIX=/path/to/install/tiledarray \
        -D CMAKE_BUILD_TYPE=Release \
        -D CMAKE_TOOLCHAIN_FILE=cmake/vg/toolchains/macos-clang-mpi-accelerate.cmake \
        $TILEDARRAY_SOURCE_DIR
```

Following are several common examples of configuring TiledArray where instead of a toolchain file
we specify CMake variables "manually" (on the command line).

* Basic configuration. This will search for dependencies on your system. If the 
  required dependencies are not found on your system, they will be downloaded 
  and installed during the build process (this includes Eigen, Boost,
  and MADNESS, but not MPI or TBB). The CMAKE_PREFIX_PATH cache variables
  is a semicolon separated list of search paths. 

```
$ cmake -D CMAKE_INSTALL_PREFIX=/path/to/install/tiledarray \
        -D CMAKE_BUILD_TYPE=Release \
        -D CMAKE_PREFIX_PATH=/path/to/dependency;/path/to/another/dependency \
        $TILEDARRAY_SOURCE_DIR
```

* Basic configuration with unit tests.

```
$ cmake -D CMAKE_INSTALL_PREFIX=/path/to/install/tiledarray \
        -D CMAKE_BUILD_TYPE=Debug \
        -D TA_BUILD_UNITTEST=ON \
        -D BOOST_ROOT=/path/to/boost \
        -D CMAKE_PREFIX_PATH=/path/to/dependency;/path/to/another/dependency \
        $TILEDARRAY_SOURCE_DIR
```

* Specify dependencies, w/o MADNESS installed on the system

```
$ cmake -D CMAKE_INSTALL_PREFIX=/path/to/install/tiledarray \
        -D CMAKE_BUILD_TYPE=Release \
        -D EIGEN3_INCLUDE_DIR=/path/to/eigen/include/eigen3 \
        -D LAPACK_LIBRARIES="-L/path/to/lapack/lib -llapack -L/path/to/blas/lib -lblas" \
        -D TBB_ROOT_DIR=/path/to/tbb \
        -D CMAKE_C_COMPILER=gcc \
        -D CMAKE_CXX_COMPILER=g++ \
        -D MPI_C_COMPILER=mpicc \
        -D MPI_CXX_COMPILER=mpicxx \
        $TILEDARRAY_SOURCE_DIR
```

* Specify dependencies, w/ MADNESS installed on the system

```
$ cmake -D CMAKE_INSTALL_PREFIX=/path/to/install/tiledarray \
        -D CMAKE_BUILD_TYPE=Release \
        -D EIGEN3_INCLUDE_DIR=/path/to/eigen/include/eigen3 \
        -D LAPACK_LIBRARIES="-L/path/to/lapack/lib -llapack -L/path/to/blas/lib -lblas" \
        -D MADNESS_ROOT_DIR=/path/to/madness \
        -D CMAKE_C_COMPILER=gcc \
        -D CMAKE_CXX_COMPILER=g++ \
        -D MPI_C_COMPILER=mpicc \
        -D MPI_CXX_COMPILER=mpicxx \
        $TILEDARRAY_SOURCE_DIR
```

Additional CMake variables are given below.

## Common CMake variables

* `CMAKE_C_COMPILER` -- The C compiler
* `CMAKE_CXX_COMPILER` -- The C++ compiler
* `CMAKE_C_FLAGS` -- The C compile flags (includes CPPFLAGS and CFLAGS)
* `CMAKE_CXX_FLAGS` -- The C++ compile flags (includes CPPFLAGS and CXXFLAGS)
* `CMAKE_EXE_LINKER_FLAGS` -- The linker flags
* `CMAKE_BUILD_TYPE` -- Optimization/debug build type options include
  `Debug` (optimization off, debugging symbols and assersions on), `Release` (optimization on, debugging symbols and assertions off), `RelWithDebInfo` (optimization on, debugging symbols and assertions on) and `MinSizeRel` (same as `Release` but optimized for executable size). The default is empty build type. It is recommended that you set the build type explicitly.
* `BUILD_SHARED_LIBS` -- Enable shared libraries. This option is only available if the platform supports shared libraries; if that's true and `TA_ASSUMES_ASLR_DISABLED` is `ON` (see below) the default is `ON`, otherwise the default is `OFF`.
* `CMAKE_CXX_STANDARD` -- Specify the C++ ISO Standard to use. Valid values are `17` (default), and `20`.

Most of these are best specified in a _toolchain file_. TiledArray is recommended to use the toolchains distributed via [the Valeev Group CMake kit](https://github.com/ValeevGroup/kit-cmake/tree/master/toolchains). TiledArray by default downloads (via [the FetchContent CMake module](https://cmake.org/cmake/help/latest/module/FetchContent.html)) the VG CMake toolkit which makes the toolchains available without having to download the toolchain files manually. E.g., to use toolchain `x` from the VG CMake kit repository provide `-DCMAKE_TOOLCHAIN_FILE=cmake/vg/toolchains/x.cmake` to CMake when configuring TiledArray.

-D CMAKE_BUILD_TYPE=(Release|Debug|RelWithDebInfo)
-D BUILD_SHARED_LIBS=(TRUE|FALSE)
-D CMAKE_CXX_STANDARD=(17|20)
-D TA_ERROR=(none|throw|assert)
It is typically not necessary to specify optimization or debug flags as the
default values provided by CMake are usually correct.

## TiledArray, MADNESS runtime, and the Address Space Layout Randomization (ASLR)

ASLR is a standard technique for increasing platform security implemented by the OS kernel and/or the dynamic linker by randomizing both where the shared libraries are loaded as well as (when enabled) the absolute position of the executable in memory (such executables are known as position-independent executables). Until recently TiledArray and other MADNESS-based applications could not be used on platforms with ASLR if ran with more than 1 MPI rank; if properly configured and built TA can now be safely used on ASLR platforms. Use the following variables to control the ASLR-related aspects of TiledArray and the underlying MADNESS runtime.

* `TA_ASSUMES_ASLR_DISABLED` -- TiledArray and MADNESS runtime will assume that the Address Space Layout Randomization (ASLR) is off. By default `TA_ASSUMES_ASLR_DISABLED` is set to `OFF` (i.e. ASLR is assumed to be enabled); this will cause all TiledArray libraries by default to be static (`BUILD_SHARED_LIBS=OFF`) and compiled as position-independent code (`CMAKE_POSITION_INDEPENDENT_CODE=ON`). This will also enable a runtime check in MADworld for ASLR.
* `CMAKE_POSITION_INDEPENDENT_CODE` -- This standard CMake variable controls whether targets are compiled by default as position-independent code or not. If `BUILD_SHARED_LIBS=OFF` need to set this to `ON` if want to use the TiledArray libraries to build shared libraries or position-independent executables.

To make things more concrete, consider the following 2 scenarios:
* Platform with ASLR disabled -- set `TA_ASSUMES_ASLR_DISABLED=ON` to set the defaults correctly and enable the ASLR check. `BUILD_SHARED_LIBS` can be set to `ON` (to produce shared TA/MADworld libraries, e.g., to minimize the executable size) or to `OFF` to produce static libraries. If the TA+MADworld static libraries will be linked into shared libraries set `CMAKE_POSITION_INDEPENDENT_CODE=ON`, otherwise `CMAKE_POSITION_INDEPENDENT_CODE` will be set to OFF for maximum efficiency of function calls.
* Platform with ASLR enabled -- this is the default. Setting `BUILD_SHARED_LIBS=ON` in this scenario will produce executables that can only be safely used with 1 MPI rank, thus `BUILD_SHARED_LIBS` will be defaulted to OFF (i.e. TA+MADworld libraries will be built as static libraries). `CMAKE_POSITION_INDEPENDENT_CODE` is by default set to `ON`, thus TA+MADworld libraries can be linked into position-independent executables safely. TA+MADworld libraries can also be linked into a shared library, provided that *ALL* code using TA+MADworld is part of the *SAME* shared library. E.g. to link TA+MADworld into a Python module compile TA+MADworld libraries and their dependents as static libraries (with `CMAKE_POSITION_INDEPENDENT_CODE=ON`) and link them all together into a single module (same logic applies to shared libraries using TA+MADworld).

## MPI

You may choose from MPICH, MVAPICH, OpenMPI, Intel MPI, or your vendor provided 
MPI implementation. Specify the C and C++ MPI compiler wrappers with the 
following CMake cache variables:

* MPI_C_COMPILER -- The MPI C compiler wrapper
* MPI_CXX_COMPILER -- The MPI C++ compiler wrapper

You can build TiledArray without MPI support by setting ENABLE_MPI to OFF.
Though we strongly recommend compiling with MPI even if you do not intend
to use TiledArray in a distributed memory environment. Note, if you
build MADNESS yourself, you must also configure MADNESS with `ENABLE_MPI=OFF`
to enable this option.

## BLAS and LAPACK

TiledArray requires a serial BLAS implementation, either by linking with a
serial version of the BLAS library or by setting the number of threads to one
(1) with an environment variable. This is necessary because TiledArray evaluates tensor
expressions in parallel by subdividing them into small tasks, each of which is assumed
to be single-threaded; attempting to run a multi-threaded BLAS function inside
tasks will oversubscribe the hardware cores.

BLAS library dependency is provided by the MADNESS library, which checks for presence
of BLAS and LAPACK (which also depends on BLAS) at the configure time. Therefore, if
MADNESS is already installed on your machine you do not need to do anything. However,
the most common scenario is where TiledArray will configure and compile
MADNESS as part of its compilation; in this case it is necessary to specify
how to find the LAPACK library to TiledArray, which will in turn pass this info
to MADNESS. This is done by setting the following
CMake variables:

* `LAPACK_LIBRARIES` -- a string specifying LAPACK libraries and all of its dependencies (such as BLAS library, math library, etc.); this string can also include linker directory flags (`-L`)
* `LAPACK_INCLUDE_DIRS` -- (optional) a list of directories which contain BLAS/LAPACK-related header files
* `LAPACK_COMPILE_DEFINITIONS` -- (optional) a list of preprocessor definitions required for any code that uses BLAS/LAPACK-related header files
* `LAPACK_COMPILE_OPTIONS` -- (optional) a list of compiler options required for any code that uses BLAS/LAPACK-related header files
* `BLA_STATIC` -- indicates whether static or shared LAPACK and BLAS libraries will be preferred.

The last three variables are only needed if your code will use non-Fortran BLAS/LAPACK library API (such as CBLAS or LAPACKE)
and thus needs access to the header files. TiledArray only uses BLAS via the Fortran API, hence the last three
variables do not need to be specified.

Since TiledArray uses the Fortran API of BLAS, it may be necessary to
specify the Fortran integer size used by the particular BLAS library:

* `INTEGER4` -- Specifies the integer size (in bytes) assumed by the BLAS/LAPACK Fortran API [Default=TRUE]
      TRUE = Fortran integer*4, FALSE = Fortran integer*8

You should use the default value unless you know it is necessary for your BLAS
implementation.

Common optimized libraries OpenBLAS/GotoBLAS, BLIS, MKL (on Intel platforms),
Atlas, Accelerate (on OS X), ESSL (on BlueGene platforms), or ACML (on AMD 
platforms). You can also use the Netlib reference implementation if nothing else
is available, but this will be very slow.

Example flags:

* Accelerate on OS X

  -D LAPACK_LIBRARIES="-framework Accelerate"

* OpenBLAS with Netlib LAPACK

  -D LAPACK_LIBRARIES="-L/path/to/lapack/lib -llapack -L/path/to/openblas/lib -lopenblas -lpthread"

* Netlib

  -D LAPACK_LIBRARIES="-L/path/to/lapack/lib -llapack -L/path/to/blas/lib -lblas"

* MKL on Linux

  -D LAPACK_LIBRARIES="-L${MKLROOT}/lib/intel64 -Wl,--start-group -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -Wl,--end-group -lpthread -lm”
  
* MKL on OS X

  -D LAPACK_LIBRARIES="-L${MKLROOT}/lib -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm"

For additional information on linking different versions of MKL, see the MKL
Link Advisor page.

    https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor

## CUDA

Support for execution on CUDA-enabled hardware is controlled by the following variables:

* `ENABLE_CUDA`  -- Set to `ON` to turn on CUDA support. [Default=OFF].
* `CMAKE_CUDA_HOST_COMPILER`  -- Set to the path to the host C++ compiler to be used by CUDA compiler. CUDA compilers used to be notorious for only being able to use specific C++ host compilers, but support for more recent C++ host compilers has improved. The default is determined by the CUDA compiler and the user environment variables (`PATH` etc.).
* `ENABLE_CUDA_ERROR_CHECK` -- Set to `ON` to turn on assertions for successful completion of calls to CUDA runtime and libraries. [Default=OFF].
* `CUTT_INSTALL_DIR` -- the installation prefix of the pre-installed cuTT library. This should not be normally needed; it is strongly recommended to let TiledArray build and install cuTT.
* `UMPIRE_INSTALL_DIR` -- the installation prefix of the pre-installed Umpire library. This should not be normally needed; it is strongly recommended to let TiledArray build and install Umpire.

For the CUDA compiler and toolkit to be discoverable the CUDA compiler (`nvcc`) should be in the `PATH` environment variable. Refer to the [FindCUDAToolkit module](https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html) for more info.

## Eigen 3

You can specify the install location of Eigen 3 with the following CMake cache
variable:

* `EIGEN3_INCLUDE_DIR` -- The path to the Eigen 3 include directory

If Eigen is not found at the configure time, it will be downloaded from the Bitbucket repository.

## Pythom

To build Python bindings use the following variable:

* `TA_PYTHON` -- Set to `ON` to build Python bindings.

This also requires setting `BUILD_SHARED_LIBS` to `ON`.

## MADNESS

TiledArray uses a non-release version of MADNESS. Therefore, you should NOT
expect the most recent release of MADNESS to work with TiledArray. To ensure you
are using the correct version of MADNESS, we recommend allowing CMake to
automatically download, configure, and build MADNESS (this is the default
behavior). When CMake is configuring TiledArray, it will checkout
the correct revision of MADNESS.

The following CMake options may be used to modify build behavior or find MADNESS:

* `ENABLE_MPI` -- Enable MPI [Default=ON]
* `ENABLE_SCALAPACK` -- Enable use of ScaLAPACK bindings [Default=OFF]
* `ENABLE_TBB` -- Enable the use of TBB when building MADNESS [Default=ON]
* `ENABLE_MKL` -- Enable the use of MKL when building MADNESS [Default=ON]
* `ENABLE_GPERFTOOLS` -- Enable the use of gperftools when building MADNESS [Default=OFF]
* `ENABLE_TCMALLOC_MINIMAL` -- Enable the use of gperftool's tcmalloc_minimal library only (the rest of gperftools is skipped) when building MADNESS [Default=OFF]
* `ENABLE_LIBUNWIND` -- Force the discovery of libunwind library when building MADNESS [Default=OFF]
* `MADNESS_SOURCE_DIR` -- Path to the MADNESS source directory
* `MADNESS_BINARY_DIR` -- Path to the MADNESS build directory
* `MADNESS_URL` -- Path to the MADNESS repository [Default=MADNESS git repository]
* `MADNESS_TAG` -- Revision hash or tag to use when building MADNESS (expert only)
* `MADNESS_CMAKE_EXTRA_ARGS` -- Extra flags passed to MADNESS cmake command
* `MADNESS_CMAKE_GENERATOR` -- the CMake generator to use for building MADNESS [Default=Generator used to build TiledArray]

The following environment variables can be used to help discovery of MADNESS dependencies:
* `TBBROOT` -- the install prefix of TBB
* `MKLROOT` -- the install prefix of MKL
* `GPERFTOOLS_DIR` -- the install prefix of gperftools
* `LIBUNWIND_DIR` -- the install prefix of libunwind

If you wish to install MADNESS yourself, we recommend downloading the latest
version from the MADNESS git repository. You should not expect the latest 
release version to work correctly with TiledArray. You can specify the install
directory with:

* `MADNESS_ROOT_DIR` -- MADNESS install directory
* `CMAKE_INSTALL_PREFIX` -- Semicolon separated list of directory CMake will use
      to search for software dependencies.

## Advanced configure options:

The following CMake cache variables are tuning parameters. You should only
modify these values if you know the values for your patricular system.

* `VECTOR_ALIGNMENT` -- The alignment of memory for Tensor in bytes [Default=16]
* `CACHE_LINE_SIZE` -- The cache line size in bytes [Default=64]

`VECTOR_ALIGNMENT` controls the alignment of Tensor data, and `CACHE_LINE_SIZE`
controls the size of automatic loop unrolling for tensor operations. TiledArray
does not currently use explicit vector instructions (i.e. intrinsics), but
the code is written in such a way that compilers can more easily autovectorize
the operations when supported. In a future version, explicit vectorization
support may be added.

## Expert configure options:

* `TA_EXPERT` -- Set to `ON` to disable automatic installation of prerequisites. Useful for experts, hence the name. [Default=OFF].
* `TA_TRACE_TASKS` -- Set to `ON` to enable tracing of MADNESS tasks using custom task tracer. Note that standard profilers/tracers are generally useless (except in the trivial cases) with MADWorld-based programs since the submission context of tasks is not captured by standard tracing tools; this makes it impossible in a nontrivial program to attribute tasks to source code. WARNING: task tracing his will greatly increase the memory requirements. [Default=OFF].
* `TA_ENABLE_RANGEV3` -- Set to `ON` to find or fetch the Range-V3 library and enable additional tests of TA components with constructs anticipated to be supported in the future. [Default=OFF].
* `TA_SIGNED_1INDEX_TYPE` -- Set to `OFF` to use unsigned 1-index coordinate type (default for TiledArray 1.0.0-alpha.2 and older). The default is `ON`, which enables the use of negative indices in coordinates.

# Build TiledArray

```
    $ cmake --build .
    (optional) $ cmake --build . --target check
    $ cmake --build . --target install
```
