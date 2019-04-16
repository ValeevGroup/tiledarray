# Prerequisites

- C++ compiler with support for the [C++14 standard](https://www.iso.org/standard/64029.html). This includes the following compilers:
  - [GNU C++](https://gcc.gnu.org/), version 5.0 or higher
  - [Clang](https://clang.llvm.org/), version 3.4 or higher
  - Apple Clang, version 5.0 or higher
  See the current [Travis CI matrix]() for the most up-to-date list of compilers that are known to work.
- [CMake](https://cmake.org/), version 3.1 or higher
- [Git]() 1.8 or later (required to obtain TiledArray and MADNESS source code from GitHub)
- [Eigen](http://eigen.tuxfamily.org), version 3.3 or higher
- BLAS library
- [MADNESS](https://github.com/m-a-d-n-e-s-s/madness)
  Only the MADworld runtime and BLAS C API component of MADNESS is used by TiledArray.
  If usable MADNESS installation is now found, TiledArray will download and compile
  MADNESS. *This is the recommended way to compile MADNESS for all users*.
  A detailed list of MADNESS prerequisites can be found at [MADNESS' INSTALL file](https://github.com/m-a-d-n-e-s-s/madness/blob/INSTALL_CMake);
  it also also contains detailed
  MADNESS compilation instructions.

  Compiling MADNESS requires the following prerequisites:
  - An implementation of Message Passing Interface version 2 or 3, with suppport
    for MPI_THREAD_MULTIPLE.
  - BLAS and LAPACK libraries (only BLAS is used by TiledArray, but without LAPACK MADNESS will not compile)
  - (optional) [Elemental](http://libelemental.org/), a distributed-memory linear algebra library
  - (optional, strongly recommended on x86 platforms)
    Intel Thread Building Blocks (TBB), available in a [commercial](software.intel.com/tbb‎) or
    an [open-source](https://www.threadingbuildingblocks.org/) form

Optional prerequisites:
- Doxygen (required to generating documentation)
- [Boost libraries](www.boost.org/), version 1.30 or higher, required for unit tests

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
        -D CMAKE_TOOLCHAIN_FILE=$TILEDARRAY_SOURCE_DIR/cmake/toolchains/osx-clang-mpi-accelerate.cmake \
        $TILEDARRAY_SOURCE_DIR
```

Following are several common examples of configuring TiledArray where instead of a toolchain file
we specify CMake variables "manually" (on the command line).

* Basic configuration. This will search for dependencies on your system. If the 
  required dependencies are not found on your system, they will be downloaded 
  and installed during the build process (this includes Eigen, Boost, Elemental,
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
* `CMAKE_BUILD_TYPE` -- Optimization/debug build type options include empty,
  Debug, Release, RelWithDebInfo and MinSizeRel.
* `BUILD_SHARED_LIBS` -- Enable shared libraries [Default=ON if supported by the platform]. With `BUILD_SHARED_LIBS=ON` only uniprocess runs will be possible due to the limitations of the MADWorld runtime.
* `CMAKE_CXX_STANDARD` -- Specify the C++ ISO Standard to use. Valid values are `14` (default), `17`, and `20`.

It is typically not necessary to specify optimization or debug flags as the
default values provided by CMake are usually correct.

## MPI

You may choose from MPICH, MVAPICH, OpenMPI, Intel MPI, or your vendor provided 
MPI implementation. Specify the C and C++ MPI compiler wrappers with the 
following CMake cache variables:

* MPI_C_COMPILER -- The MPI C compiler wrapper
* MPI_CXX_COMPILER -- The MPI C++ compiler wrapper

You can build TiledArray without MPI support by setting ENABLE_MPI to OFF.
Though we strongly recommend compiling with MPI even if you do not intend
to use TiledArray in a distributed memory environment. Note, if you
build MADNESS yourself, you must also configure MADNESS with --enable-stub-mpi
to enable this option.

## BLAS and LAPACK

TiledArray requires a serial BLAS implementation, either by linking with a
serial version of the BLAS library or by setting the number of threads to one
(1) with an environment variable. This is necessary because TiledArray evaluates tensor
expressions in parallel by subdividing them into small tasks, each of which is assumed
to be single-threaded; attempting to run a multi-threaded BLAS function inside
tasks will over subscribe the hardware cores.

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

## Eigen 3

You can specify the install location of Eigen 3 with the following CMake cache
variable:

* `EIGEN3_INCLUDE_DIR` -- The path to the Eigen 3 include directory

If Eigen is not found at the configure time, it will be downloaded from the Bitbucket repository.

## MADNESS

TiledArray uses a non-release version of MADNESS. Therefore, you should NOT
expect the most recent release of MADNESS to work with TiledArray. To ensure you
are using the correct version of MADNESS, we recommend allowing CMake to
automatically download, configure, and build MADNESS (this is the default
behavior). When CMake is configuring TiledArray, it will checkout
the correct revision of MADNESS.

The following CMake options may be used to modify build behavior or find
MADNESS:

* `ENABLE_MPI` -- Enable MPI [Default=ON]
* `ENABLE_ELEMENTAL` -- Enable use of MADNESS provided Elemental [Default=OFF]
* `ENABLE_TBB` -- Enable the use of TBB when building MADNESS [Default=ON]
* `TBB_ROOT_DIR` -- The install directory for TBB
* `TBB_INCLUDE_DIR` -- The include directory for TBB header files
* `TBB_LIBRARY` -- The library directory for TBB shared libraries
* `MADNESS_SOURCE_DIR` -- Path to the MADNESS source directory
* `MADNESS_BINARY_DIR` -- Path to the MADNESS build directory
* `MADNESS_URL` -- Path to the MADNESS repository [Default=MADNESS git repository]
* `MADNESS_TAG` -- Revision hash or tag to use when building MADNESS (expert only)
* `MADNESS_CMAKE_EXTRA_ARGS` -- Extra flags passed to MADNESS cmake command

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

VECTOR_ALIGNMENT controls the alignment of Tensor data, and CACHE_LINE_SIZE
controls the size of automatic loop unrolling for tensor operations. TiledArray
does not currently use explicit vector instructions (i.e. intrinsics), but
the code is written in such a way that compilers can more easily autovectorize 
the operations when supported. In a future version, explicit vectorization
support may be added.

# Build TiledArray

```
    $ make -j
    ... many lines omitted ...
    $ make check
    ... many lines omitted ...
    $ make install
```