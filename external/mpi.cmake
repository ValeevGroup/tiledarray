# Find MPI

include(AppendFlags)

if(ENABLE_MPI)

  # Try to find MPI
  find_package(MPI REQUIRED)
  
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
  
  # Set the  build variables
  include_directories(${MPI_INCLUDE_PATH})
  list(APPEND TiledArray_CONFIG_INCLUDE_DIRS ${MPI_INCLUDE_PATH})
  append_flags(CMAKE_CXX_FLAGS "${MPI_COMPILE_FLAGS}")
  append_flags(CMAKE_EXE_LINKER_FLAGS "${MPI_LINK_FLAGS}")
  set(TiledArray_LIBRARIES ${MPI_LIBRARIES} ${TiledArray_LIBRARIES})
  set(TiledArray_CONFIG_LIBRARIES ${MPI_LIBRARIES} ${TiledArray_CONFIG_LIBRARIES})

endif()