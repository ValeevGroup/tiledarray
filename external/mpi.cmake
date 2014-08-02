# Find MPI

include(AppendFlags)

if(ENABLE_MPI)
  # Set the MPICC compiler
  if(NOT DEFINED MPI_C_COMPILER)
    set(MPI_C_COMPILER ${CMAKE_C_COMPILER})
  endif()
  
  # Set the MPICXX complier
  if(NOT DEFINED MPI_CXX_COMPILER)
    set(MPI_CXX_COMPILER ${CMAKE_CXX_COMPILER})
  endif()

  # Try to find MPI
  find_package(MPI)
  
  # Set the variables 
  if(MPI_C_FOUND)
    set(MPI_FOUND         ${MPI_C_FOUND})
    set(MPI_COMPILE_FLAGS ${MPI_C_COMPILE_FLAGS})
    set(MPI_INCLUDE_PATH  ${MPI_C_INCLUDE_PATH})
    set(MPI_LINK_FLAGS    ${MPI_C_LINK_FLAGS})
    set(MPI_LIBRARIES     ${MPI_C_LIBRARIES})
  elseif(MPI_CXX_FOUND)
    set(MPI_FOUND         ${MPI_CXX_FOUND})
    set(MPI_COMPILE_FLAGS ${MPI_CXX_COMPILE_FLAGS})
    set(MPI_INCLUDE_PATH  ${MPI_CXX_INCLUDE_PATH})
    set(MPI_LINK_FLAGS    ${MPI_CXX_LINK_FLAGS})
    set(MPI_LIBRARIES     ${MPI_CXX_LIBRARIES})
  else()
    message(FATAL_ERROR "No suitable MPI compiler was not found.")
  endif()
  
  # Add compile definitions to disable C++ bindings for OpenMPI and MPICH
  append_flags(MPI_COMPILE_FLAGS "-DOMPI_SKIP_MPICXX=1 -DMPICH_SKIP_MPICXX=1")

endif()