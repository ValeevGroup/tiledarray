# -*- mode: cmake -*-

###################
# Find MADNESS
###################

include(ExternalProject)
include(ConvertIncludesListToCompilerArgs)
include(ConvertLibrariesListToCompilerArgs)

find_package(Madness COMPONENTS MADworld MADlinalg)

if(MADNESS_FOUND)
  
  if(ENABLE_ELEMENTAL)
    find_package(Elemental REQUIRED COMPONENTS pmrrr)
    
    include_directories(${Elemental_INCLUDE_DIRS})
    list(APPEND TiledArray_LIBRARIES "${Elemental_LIBRARIES}")
  endif()

  cmake_push_check_state()
  
  list(APPEND CMAKE_REQUIRED_INCLUDES ${Madness_INCLUDE_DIRS} 
      ${Elemental_INCLUDE_DIRS} ${MPI_INCLUDE_PATH})
  list(APPEND CMAKE_REQUIRED_LIBRARIES ${Madness_LIBRARIES} ${Elemental_LIBRARIES}
      ${LAPACK_LINKER_FLAGS} ${BLAS_LINKER_FLAGS} ${MPI_LINK_FLAGS}
      ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} ${MPI_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} ${MPI_COMPILE_FLAGS}")

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
    
  if(ENABLE_ELEMENTAL)
  
    find_package(Elemental COMPONENTS pmrrr;lapack-addons REQUIRED)
    
    # Check to that MADNESS was compiled with Elemental support.
    CHECK_CXX_SOURCE_COMPILES(
        "
        #include <madness/madness_config.h>
        #ifndef MADNESS_HAS_ELEMENTAL
        # error MADNESS does not have Elemental
        #endif
        int main(int argc, char** argv) {
          return 0;
        }
        "  MADNESS_HAS_ELEMENTAL_SUPPORT)
        
    if(NOT MADNESS_HAS_ELEMENTAL_SUPPORT)
      message(FATAL_ERROR "MADNESS does not include Elemental support.")
    endif() 
    
    set(TILEDARRAY_HAS_ELEMENTAL ${MADNESS_HAS_ELEMENTAL_SUPPORT})
  endif()

  cmake_pop_check_state()
  
elseif(TA_EXPERT)

  message("** MADNESS was not found or explicitly set")
  message(FATAL_ERROR "** Downloading and building MADNESS is explicitly disabled in EXPERT mode")

else()

  if(NOT DEFINED Madness_URL)
    set(Madness_URL "https://github.com/m-a-d-n-e-s-s/madness.git")
  endif()
  if(NOT DEFINED Madness_TAG)
    set(Madness_TAG "5b53396a06a75badf71719e2327f7fa41e58e377")
  endif()
  message(STATUS "Will pull MADNESS from ${Madness_URL}")  
  
  if(ENABLE_ELEMENTAL)
    include(external/elemental.cmake)
  endif()
  
  # Setup configure variables
  
  # Set compile flags
  set(MAD_CPPFLAGS "${CMAKE_CPP_FLAGS}")
  set(MAD_CFLAGS "${CMAKE_C_FLAGS}")
  set(MAD_CXXFLAGS "${CMAKE_CXX_FLAGS}")
  set(MAD_LDFLAGS "${CMAKE_EXE_LINKER_FLAGS}")
  
  if(CMAKE_BUILD_TYPE STREQUAL Debug)
    append_flags(MAD_CFLAGS "${CMAKE_C_FLAGS_DEBUG}")
    append_flags(MAD_CXXFLAGS "${CMAKE_CXX_FLAGS_DEBUG}")
    append_flags(MAD_LDFLAGS "${CMAKE_EXE_LINKER_FLAGS_DEBUG}")
  elseif(CMAKE_BUILD_TYPE STREQUAL Release)
    append_flags(MAD_CFLAGS "${CMAKE_C_FLAGS_RELEASE}")
    append_flags(MAD_CXXFLAGS "${CMAKE_CXX_FLAGS_RELEASE}")
    append_flags(MAD_LDFLAGS "${CMAKE_EXE_LINKER_FLAGS_RELEASE}")
  elseif(CMAKE_BUILD_TYPE STREQUAL RelWithDebInfo)
    append_flags(MAD_CFLAGS "${CMAKE_C_FLAGS_RELWITHDEBINFO}")
    append_flags(MAD_CXXFLAGS "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
    append_flags(MAD_LDFLAGS "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO}")
  elseif(CMAKE_BUILD_TYPE STREQUAL MinSizeRel)
    append_flags(MAD_CFLAGS "${CMAKE_C_FLAGS_MINSIZEREL}")
    append_flags(MAD_CXXFLAGS "${CMAKE_CXX_FLAGS_MINSIZEREL}")
    append_flags(MAD_LDFLAGS "${CMAKE_EXE_LINKER_FLAGS_MINSIZEREL}")
  endif()
  
  # Set compile flags required for Elemental
  if(ENABLE_ELEMENTAL)
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
  
  # Set compile flags required for LAPACK
  append_flags(MAD_LDFLAGS "${LAPACK_LINKER_FLAGS}")
  foreach(_lib ${LAPACK_LIBRARIES})
    append_flags(MAD_LIBS "${_lib}")
  endforeach()
  
  # Set compile flags required for Pthreads
  foreach(_lib ${CMAKE_THREAD_LIBS_INIT})
    append_flags(MAD_LIBS "${_lib}")
  endforeach()
  
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
      --without-mkl
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
    BUILD_COMMAND $(MAKE) libraries V=0
   #--Install step---------------
    INSTALL_COMMAND ""
    STEP_TARGETS download patch configure build
    )

  # Add madness-update target that will pull updates to the madness source from
  # the git repository. This is done outside ExternalProject_add to prevent
  # madness from doing a full pull, configure, and build everytime the project
  # is built.
  add_custom_target(madness-update
    COMMAND ${GIT_EXECUTABLE} pull --rebase origin master
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
          COMMAND \"${CMAKE_MAKE_PROGRAM}\" \"install-libraries\" 
          WORKING_DIRECTORY \"${MADNESS_BINARY_DIR}\"
          RESULT_VARIABLE error_code)
      if(error_code)
        message(FATAL_ERROR \"Failed to install 'madness'\")
      endif()
      "
      )

  # Set build dependencies and compiler arguments
  add_dependencies(External madness)

  # MadnessFind will set Madness_INCLUDE_DIRS and Madness_LIBRARIES with the
  # dependencies. So all that needs to be done here is set Madness_INCLUDE_DIR,
  # Madness_INCLUDE_DIRS, Madness_LIBRARY, and Madness_LIBRARIES with the paths
  # and libraries for the built version of MADNESS above.
  set(Madness_INCLUDE_DIRS
      ${MADNESS_BINARY_DIR}/src
      ${MADNESS_SOURCE_DIR}/src)
  set(Madness_LIBRARIES ${MADNESS_BINARY_DIR}/src/madness/world/libMADworld.a)

endif()

include_directories(${Madness_INCLUDE_DIRS})
set(TiledArray_LIBRARIES ${Madness_LIBRARIES} ${TiledArray_LIBRARIES})

