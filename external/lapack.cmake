# Find BLAS and LAPACK.
include(CheckCFortranFunctionExists)
include(CMakePushCheckState)

if(LAPACK_LIBRARIES OR BLAS_LIBRARIES OR LAPACK_LINKER_FLAGS OR BLAS_LINKER_FLAGS)
  # Here we verify that the we can link against LAPACK and BLAS based on the
  # given library and linker flags. If BLAS_FOUND and/or LAPACK_FOUND are true,
  # we assume that these values have been verified.

  if(NOT BLAS_FOUND)
    # Verify that we can link against BLAS

    cmake_push_check_state()
    
    if(UNIX AND BLA_STATIC AND BLAS_LIBRARIES)
      set(CMAKE_REQUIRED_LIBRARIES ${BLAS_LINKER_FLAGS} "-Wl,-start-group"
          ${BLAS_LIBRARIES} "-Wl,-end-group" ${TiledArray_CONFIG_LIBRARIES}
          ${CMAKE_REQUIRED_LIBRARIES})
    else()
      set(CMAKE_REQUIRED_LIBRARIES ${BLAS_LINKER_FLAGS} ${BLAS_LIBRARIES}
          ${TiledArray_CONFIG_LIBRARIES} ${CMAKE_REQUIRED_LIBRARIES})
    endif()

    check_c_fortran_function_exists(sgemm BLAS_FOUND)

    if(BLAS_FOUND)
      message(STATUS "A library with BLAS API found.")
    else()
      message(FATAL_ERROR "The user specified BLAS libraries do not support the BLAS API.\n"
                          "Rerun cmake with BLAS_LIBRARIES and/or BLAS_LINKER_FLAGS.")
    endif()
    
    cmake_pop_check_state()
  endif()
  
  if(NOT LAPACK_FOUND)
    # Verify that we can link against LAPACK

    cmake_push_check_state()

    if(UNIX AND BLA_STATIC AND (LAPACK_LIBRARIES OR BLAS_LIBRARIES))
      set(CMAKE_REQUIRED_LIBRARIES ${LAPACK_LINKER_FLAGS} ${BLAS_LINKER_FLAGS} "-Wl,--start-group"
          ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} "-Wl,--end-group" ${TiledArray_CONFIG_LIBRARIES}
          ${CMAKE_REQUIRED_LIBRARIES})
    else()
      set(CMAKE_REQUIRED_LIBRARIES ${LAPACK_LINKER_FLAGS} ${BLAS_LINKER_FLAGS} ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES}
          ${TiledArray_CONFIG_LIBRARIES} ${CMAKE_REQUIRED_LIBRARIES})
    endif()

    check_c_fortran_function_exists(cheev LAPACK_FOUND)

    if(LAPACK_FOUND)
      message(STATUS "A library with LAPACK API found.")
    else()
      message(FATAL_ERROR "The user specified LAPACK libraries do not support the LAPACK API.\n"
                          "Rerun cmake with LAPACK_LIBRARIES and/or LAPACK_LINKER_FLAGS.")
    endif()
    
    cmake_pop_check_state()
    
  endif()
else()
  # Try to find BLAS and LAPACK
  find_package(LAPACK REQUIRED)
endif()

# Set the  build variables
append_flags(CMAKE_EXE_LINKER_FLAGS "${BLAS_LINKER_FLAGS}")
append_flags(CMAKE_EXE_LINKER_FLAGS "${LAPACK_LINKER_FLAGS}")
if(UNIX AND BLA_STATIC AND (LAPACK_LIBRARIES OR BLAS_LIBRARIES))
  set(TiledArray_LIBRARIES "-Wl,--start-group"
      ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} "-Wl,--end-group" ${TiledArray_LIBRARIES})
  set(TiledArray_CONFIG_LIBRARIES "-Wl,--start-group"
      ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} "-Wl,--end-group" ${TiledArray_CONFIG_LIBRARIES})
else()
  set(TiledArray_LIBRARIES ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES}
      ${TiledArray_LIBRARIES})
  set(TiledArray_CONFIG_LIBRARIES ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES}
      ${TiledArray_CONFIG_LIBRARIES})
endif()

