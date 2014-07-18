# Set compilers
set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpic++11)

# Set environment paths
set(GCC_ROOT  "/bgsys/drivers/ppcfloor/gnu-linux")
set(CLANG_ROOT "/home/projects/llvm")

set(IBM_DIR $ENV{IBM_MAIN_DIR})
set(XLF_DIR "${IBM_DIR}/xlf/bg/14.1")
set(XLSMP_DIR "${IBM_DIR}/xlsmp/bg/3.1")
set(ESSL_DIR "/soft/libraries/essl/current/essl/5.1")
set(LAPACK_DIR "/soft/libraries/alcf/current/xl/LAPACK")

set(CMAKE_CXX_COMPILE_FLAGS "")
set(CMAKE_EXE_LINKER_FLAGS "")
set(CMAKE_STATIC_LINKER_FLAGS "")

# V1R2M0
#set(MPI_ROOT  "/bgsys/drivers/ppcfloor/comm/gcc")
#set(PAMI_ROOT "/bgsys/drivers/ppcfloor/comm/sys")
# V1R2M1
set(MPI_ROOT  "/bgsys/drivers/ppcfloor/comm")
set(PAMI_ROOT "/bgsys/drivers/ppcfloor/comm")
set(SPI_ROOT  "/bgsys/drivers/ppcfloor/spi")

# Set library

set(XL_LIBRARIES ${XLSMP_DIR}/bglib64/libxlsmp.a)
set(XLF_LIBRARIES ${XLF_DIR}/bglib64/libxlf90_r.a;${XLF_DIR}/bglib64/libxlfmath.a;${XLF_DIR}/bglib64/libxlopt.a;${XLF_DIR}/bglib64/libxl.a)
set(BLAS_LIBRARY ${ESSL_DIR}/lib64/libesslbg.a)
set(BLAS_LIBRARIES ${BLAS_LIBRARY};${XLF_LIBRARIES};${XL_LIBRARIES})
set(LAPACK_LIBRARY ${LAPACK_DIR}/lib/liblapack.a)
set(LAPACK_LIBRARIES ${LAPACK_LIBRARY};${BLAS_LIBRARIES})

##############################################################

# set the search path for the environment coming with the compiler
# and a directory where you can install your own compiled software
set(CMAKE_FIND_ROOT_PATH
    /bgsys/drivers/ppcfloor/
    ${MPI_ROOT}
    ${PAMI_ROOT}
    ${GCC_ROOT}
    ${IBM_DIR}
    ${XLF_DIR}
    ${XLSMP_DIR}
    ${ESSL_DIR})

# adjust the default behaviour of the FIND_XXX() commands:
# search headers and libraries in the target environment, search
# programs in the host environment
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

##############################################################