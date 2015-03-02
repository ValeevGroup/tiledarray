# -*- mode: cmake -*-

###################
# Find MADNESS
###################

include(ExternalProject)
include(ConvertIncludesListToCompilerArgs)
include(ConvertLibrariesListToCompilerArgs)

find_package(Madness CONFIG QUIET COMPONENTS MADworld MADlinalg)

if(MADNESS_FOUND)
  
  if(ENABLE_ELEMENTAL)
    find_package(Elemental REQUIRED COMPONENTS pmrrr)
    
    # Set config variables
    include_directories(${Elemental_INCLUDE_DIRS})
    set(TiledArray_LIBRARIES ${Elemental_LIBRARIES} ${TiledArray_LIBRARIES})
    list(APPEND TiledArray_CONFIG_INCLUDE_DIRS ${Elemental_INCLUDE_DIRS})
    set(TiledArray_CONFIG_LIBRARIES ${Elemental_LIBRARIES} ${TiledArray_CONFIG_LIBRARIES})
    
    # Check to that MADNESS was compiled with Elemental support.
    CHECK_CXX_SOURCE_COMPILES(
        "
        #include <madness/madness_config.h>
        #ifndef MADNESS_HAS_ELEMENTAL
        # error MADNESS does not have Elemental
        #endif
        int main(int, char**) { return 0; }
        "  MADNESS_HAS_ELEMENTAL_SUPPORT)
        
    if(NOT MADNESS_HAS_ELEMENTAL_SUPPORT)
      message(FATAL_ERROR "MADNESS does not include Elemental support.")
    endif() 
    
    set(TILEDARRAY_HAS_ELEMENTAL ${MADNESS_HAS_ELEMENTAL_SUPPORT})
  endif()

  cmake_push_check_state()
  
  list(APPEND CMAKE_REQUIRED_INCLUDES ${Madness_INCLUDE_DIRS} ${TiledArray_CONFIG_INCLUDE_DIRS})
  list(APPEND CMAKE_REQUIRED_LIBRARIES "${Madness_LINKER_FLAGS}" ${Madness_LIBRARIES}
      "${CMAKE_EXE_LINKER_FLAGS}" ${TiledArray_CONFIG_LIBRARIES})
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} ${Madness_COMPILE_FLAGS} ${CMAKE_CXX_FLAGS}")

  # sanity check: try compiling a simple program
  CHECK_CXX_SOURCE_COMPILES(
    "
    #include <madness/world/world.h>
    int main(int argc, char** argv) {
      madness::World& world = madness::initialize(argc, argv);
      madness::finalize();
      return 0;
    }
    "  Madness_COMPILES)

  if (NOT Madness_COMPILES)
    message(FATAL_ERROR "MADNESS found, but does not compile correctly.")
  endif()

  cmake_pop_check_state()

  # Set config variables
  list(APPEND TiledArray_CONFIG_INCLUDE_DIRS ${Madness_INCLUDE_DIRS})
  set(TiledArray_CONFIG_LIBRARIES ${Madness_LIBRARIES} ${TiledArray_CONFIG_LIBRARIES})
  set(TiledArray_LIBRARIES ${Madness_LIBRARIES} ${TiledArray_LIBRARIES})

  
elseif(TA_EXPERT)

  message("** MADNESS was not found or explicitly set")
  message(FATAL_ERROR "** Downloading and building MADNESS is explicitly disabled in EXPERT mode")

else()

  if(NOT DEFINED Madness_URL)
    set(Madness_URL "https://github.com/m-a-d-n-e-s-s/madness.git")
  endif()
  if(NOT DEFINED Madness_TAG)
    set(Madness_TAG "9b84ab30dfa95eb2de4acd9a6e0682f9fb1f352f")
  endif()
  message(STATUS "Will pull MADNESS from ${Madness_URL}")
  
  # Setup configure variables
  
  # Set compile flags
  set(MAD_CPPFLAGS "${CMAKE_CPP_FLAGS}")
  set(MAD_CFLAGS "${CMAKE_C_FLAGS}")
  set(MAD_CXXFLAGS "${CMAKE_CXX_FLAGS}")
  set(MAD_LDFLAGS "${CMAKE_EXE_LINKER_FLAGS}")

  if(CMAKE_BUILD_TYPE)
    string(TOLOWER MAD_BUILD_TYPE "${CMAKE_BUILD_TYPE}")
    append_flags(MAD_CFLAGS "${CMAKE_C_FLAGS_${MAD_BUILD_TYPE}}")
    append_flags(MAD_CXXFLAGS "${CMAKE_CXX_FLAGS_${MAD_BUILD_TYPE}}")
  endif()
  
  # Set compile flags required for Elemental
  if(ENABLE_ELEMENTAL)
    include(external/elemental.cmake)
    
    foreach(_inc_dir ${Elemental_INCLUDE_DIRS})
      append_flags(MAD_CPPFLAGS "-I${_inc_dir}")
    endforeach()
    foreach(_lib ${Elemental_LIBRARIES})
      append_flags(MAD_LIBS "${_lib}")
    endforeach()
    set(MAD_ELEMENTAL_FLAG "yes")
  else()
    set(MAD_ELEMENTAL_FLAG "no")
  endif()
  
    # Set compile flags required for Elemental
  if(ENABLE_TBB)
    message(FATAL_ERROR "MADNESSS with TBB is not implemented")
    set(MAD_TBB_FLAG "yes")
  else()
    set(MAD_TBB_FLAG "no")
  endif()
  
  
  # Set compile flags required for MPI
  if(ENABLE_MPI)
    foreach(_inc_dir ${MPI_INCLUDE_PATH})
      append_flags(MAD_CPPFLAGS "-I${_inc_dir}")
    endforeach()
    foreach(_lib ${MPI_LIBRARIES})
      append_flags(MAD_LIBS "${_lib}")
    endforeach()
    append_flags(MAD_CFLAGS "${MPI_COMPILE_FLAGS}")
    append_flags(MAD_CXXFLAGS "${MPI_COMPILE_FLAGS}")
    append_flags(MAD_LDFLAGS "${MPI_LINK_FLAGS}") 
    set(MAD_STUB_MPI "no")
  else()
    set(MAD_STUB_MPI "yes")
  endif()
  
  # Set compile flags required for LAPACK, BLAS, and Pthreads
  append_flags(MAD_LDFLAGS "${LAPACK_LINKER_FLAGS}")
  append_flags(MAD_LDFLAGS "${BLAS_LINKER_FLAGS}")
  foreach(_lib ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
    append_flags(MAD_LIBS "${_lib}")
  endforeach()
  append_flags(MAD_LIBS "${CMAKE_THREAD_LIBS_INIT}")
  
  # Set the configuration flags for MADNESS
  
  # Set Fortran integer size
  if(INTEGER4) 
    set(MAD_F77_INT32 yes)
  else()
    set (MAD_F77_INT32 no)
  endif()
  
  # Set error handling method
  if(TA_ERROR STREQUAL none)
    set(MAD_EXCEPTION disable)
  elseif(TA_ERROR STREQUAL throw)
    set(MAD_EXCEPTION throw)
  elseif(TA_ERROR STREQUAL assert)
    set(MAD_EXCEPTION assert)
  else()
    set(MAD_EXCEPTION throw)
  endif()
  
  # Set paths for MADNESS project
  set(MADNESS_SOURCE_DIR ${PROJECT_SOURCE_DIR}/external/src/madness)
  set(MADNESS_BINARY_DIR  ${PROJECT_BINARY_DIR}/external/build/madness)
  
  ExternalProject_Add(madness
    DEPENDS ${MAD_DEPENDS}
    PREFIX ${CMAKE_INSTALL_PREFIX}
    STAMP_DIR ${MADNESS_BINARY_DIR}/stamp
   #--Download step--------------
    GIT_REPOSITORY ${Madness_URL}
    GIT_TAG ${Madness_TAG}
   #--Update/Patch step----------
    UPDATE_COMMAND ""
    PATCH_COMMAND /bin/sh ${MADNESS_SOURCE_DIR}/autogen.sh
   #--Configure step-------------
    SOURCE_DIR ${MADNESS_SOURCE_DIR}
    CONFIGURE_COMMAND ${MADNESS_SOURCE_DIR}/configure
      --quiet
      --prefix=${CMAKE_INSTALL_PREFIX}
      --disable-debugging
      --disable-optimization
      --disable-optimal
      --enable-madex=${MAD_EXCEPTION}
      --enable-dependency-tracking
      --with-mpi-thread=multiple
      --with-fortran-int32=${MAD_F77_INT32}
      --with-stubmpi=${MAD_STUB_MPI}
      --with-elemental=${MAD_ELEMENTAL_FLAG}
      --with-tbb=${MAD_TBB_FLAG}
      --without-mkl
      ${MAD_EXTRA_CONFIGURE_FLAGS}
      MPICXX=${CMAKE_CXX_COMPILER}
      MPICC=${CMAKE_C_COMPILER}
      CPPFLAGS=${MAD_CPPFLAGS}
      CC=${CMAKE_C_COMPILER} CFLAGS=${MAD_CFLAGS}
      CXX=${CMAKE_CXX_COMPILER} CXXFLAGS=${MAD_CXXFLAGS}
      F77=${CMAKE_Fortran_COMPILER} FFLAGS=${CMAKE_Fortran_FLAGS}
      LDFLAGS=${MAD_LDFLAGS}
      LIBS=${MAD_LIBS}
    CMAKE_GENERATOR "Unix Makefiles"
   #--Build step-----------------
    BINARY_DIR ${MADNESS_BINARY_DIR}
    BUILD_COMMAND $(MAKE) tensor world V=0
   #--Install step---------------
    INSTALL_COMMAND ""
    STEP_TARGETS download patch configure build
    )

  # Add madness-update target that will pull updates to the madness source from
  # the git repository. This is done outside ExternalProject_add to prevent
  # madness from doing a full pull, configure, and build everytime the project
  # is built.
  add_custom_target(madness-update
    COMMAND ${GIT_EXECUTABLE} fetch
    COMMAND ${GIT_EXECUTABLE} checkout ${Madness_TAG}
    COMMAND ${CMAKE_COMMAND} -E touch_nocreate ${MADNESS_BINARY_DIR}/stamp/madness-configure
    WORKING_DIRECTORY ${MADNESS_SOURCE_DIR}
    COMMENT "Updating source for 'madness' from ${MADNESS_URL}")

  # Add madness-clean target that will delete files generated by MADNESS build.
  add_custom_target(madness-clean
    COMMAND $(MAKE) clean
    COMMAND ${CMAKE_COMMAND} -E touch_nocreate ${MADNESS_BINARY_DIR}/stamp/madness-configure
    WORKING_DIRECTORY ${MADNESS_BINARY_DIR}
    COMMENT Cleaning build directory for 'madness')
  
  # Since 'madness-install' target cannot be linked to the 'install' target,
  # we will do it manually here.
  install(CODE
      "
      execute_process(
          COMMAND \"${CMAKE_MAKE_PROGRAM}\" \"install-tensor\" \"install-world\" 
          WORKING_DIRECTORY \"${MADNESS_BINARY_DIR}\"
          RESULT_VARIABLE error_code)
      if(error_code)
        message(FATAL_ERROR \"Failed to install 'madness'\")
      endif()
      "
      )

  # Set build dependencies and compiler arguments
  add_dependencies(External madness)

  # Set config variables 
  if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin" AND NOT ${CMAKE_SYSTEM_VERSION} VERSION_LESS 11.0)
    # Building on OS X 10.7 or later, so add "-Wl,-no_pie" linker flags.
    set(Madness_LINKER_FLAGS "-Wl,-no_pie")
  endif()
  set(Madness_INCLUDE_DIRS ${MADNESS_BINARY_DIR}/src ${MADNESS_SOURCE_DIR}/src)
  set(Madness_LIBRARIES 
      ${MADNESS_BINARY_DIR}/src/madness/world/${CMAKE_STATIC_LIBRARY_PREFIX}MADworld${CMAKE_STATIC_LIBRARY_SUFFIX})
  set(TiledArray_CONFIG_LIBRARIES 
      "${CMAKE_INSTALL_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}MADworld${CMAKE_STATIC_LIBRARY_SUFFIX}"
      ${TiledArray_CONFIG_LIBRARIES})

endif()

include_directories(${Madness_INCLUDE_DIRS})
set(TiledArray_LIBRARIES ${Madness_LIBRARIES} ${TiledArray_LIBRARIES})
append_flags(CMAKE_CXX_FLAGS "${Madness_COMPILE_FLAGS}")
append_flags(CMAKE_EXE_LINKER_FLAGS "${Madness_LINKER_FLAGS}")

