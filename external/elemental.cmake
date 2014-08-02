# -*- mode: cmake -*-

######################
# Find Elemental
######################

include(ExternalProject)

find_package(Elemental COMPONENTS elemental;pmrrr;lapack-addons)

if (ELEMENTAL_LIBRARIES)
  if (NOT ELEMENTAL_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find elemental.hpp")
  endif(NOT ELEMENTAL_INCLUDE_DIR)
  
  if(NOT TILEDARRAY_HAS_CXX11)
    message(FATAL_ERROR "Elemental requires a C++11 compatible compiler.")
  endif(NOT TILEDARRAY_HAS_CXX11)
  
  cmake_push_check_state()
  
  # Elemental compiles check
  MESSAGE(STATUS "ELEMENTAL LIBRARIES = ${ELEMENTAL_LIBRARIES}")
  list(APPEND CMAKE_REQUIRED_INCLUDES ${ELEMENTAL_INCLUDE_DIR})
  list(APPEND CMAKE_REQUIRED_LIBRARIES ${ELEMENTAL_LIBRARIES})
  list(APPEND CMAKE_REQUIRED_LIBRARIES ${LAPACK_LIBRARIES})
  CHECK_CXX_SOURCE_COMPILES(
    "
    #include <elemental.hpp>
    using namespace elem;
    int main (int argc, char** argv){
      Initialize(argc, argv);
      mpi::Comm comm = mpi::COMM_WORLD;
      const Grid grid(comm);
      DistMatrix<double> X(grid);
      Identity(X, 16, 16);
      
      Finalize();
      return 0;
    }
    " 
    ELEMENTAL_COMPILES
  )   

  cmake_pop_check_state()
  
  if (NOT ELEMENTAL_COMPILES)
    message(STATUS "Could not compile Elemental test program")
  endif(NOT ELEMENTAL_COMPILES)
  
  if (ELEMENTAL_COMPILES)
    set (TILEDARRAY_HAS_ELEMENTAL TRUE)
    message(STATUS "Found ELEMENTAL")
    message(STATUS "\tELEMENTAL_LIBRARIES = ${ELEMENTAL_LIBRARIES}")
    message(STATUS "\tELEMENTAL_INCLUDE_DIR = ${ELEMENTAL_INCLUDE_DIR}")
    include_directories(${ELEMENTAL_INCLUDE_DIR})
  endif(ELEMENTAL_COMPILES)

endif(ELEMENTAL_LIBRARIES)
