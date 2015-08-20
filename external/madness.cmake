# -*- mode: cmake -*-

###################
# Find MADNESS
###################

include(ExternalProject)
include(ConvertIncludesListToCompilerArgs)
include(ConvertLibrariesListToCompilerArgs)

find_package(Madness CONFIG QUIET COMPONENTS MADworld MADlinalg HINTS ${Madness_ROOT_DIR})

if(Madness_FOUND)

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
    
  if(ENABLE_ELEMENTAL)
    
    # Check to that MADNESS was compiled with Elemental support.
    CHECK_CXX_SOURCE_COMPILES(
        "
        #include <madness/world/parallel_runtime.h>
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

  cmake_pop_check_state()

  # Set config variables
  list(APPEND TiledArray_CONFIG_INCLUDE_DIRS ${Madness_INCLUDE_DIRS})
  set(TiledArray_CONFIG_LIBRARIES ${Madness_LIBRARIES} ${TiledArray_CONFIG_LIBRARIES})
  set(TiledArray_LIBRARIES ${Madness_LIBRARIES} ${TiledArray_LIBRARIES})

  
elseif(TA_EXPERT)

  message("** MADNESS was not found or explicitly set")
  message(FATAL_ERROR "** Downloading and building MADNESS is explicitly disabled in EXPERT mode")

else()

  find_package(Git REQUIRED)
  message(STATUS "git found: ${GIT_EXECUTABLE}")

  # Create a cache entry for MADNESS build variables.
  # Note: This will not overwrite user specified values.
  set(MADNESS_SOURCE_DIR "${PROJECT_SOURCE_DIR}/external/src/madness" CACHE PATH 
        "Path to the MADNESS source directory")
  set(MADNESS_BINARY_DIR "${PROJECT_BINARY_DIR}/external/build/madness" CACHE PATH 
        "Path to the MADNESS build directory")
  set(Madness_URL "https://github.com/m-a-d-n-e-s-s/madness.git" CACHE STRING 
        "Path to the MADNESS repository")
  set(Madness_TAG "1da8039add7996dc9468e73329160be2dd29980a" CACHE STRING 
        "Revision hash or tag to use when building MADNESS")
  
  if("${Madness_TAG}" STREQUAL "")
    message(FATAL_ERROR "Invalid value given for Madness_TAG; specify a valid hash or tag.")
  endif()
  
  # Check the MADNESS source directory to make sure it contains the source files
  # If the MADNESS source directory is the default location and does not exist,
  # MADNESS will be downloaded from git.
  message(STATUS "Checking MADNESS source directory: ${MADNESS_SOURCE_DIR}")
  if("${MADNESS_SOURCE_DIR}" STREQUAL "${PROJECT_SOURCE_DIR}/external/src/madness")

    # Create the external source directory
    if(NOT EXISTS ${PROJECT_SOURCE_DIR}/external/src)
      set(error_code 1)
      execute_process(
          COMMAND "${CMAKE_COMMAND}" -E make_directory "${PROJECT_SOURCE_DIR}/external/src"
          RESULT_VARIABLE error_code)
      if(error_code)
        message(FATAL_ERROR "Failed to create the MADNESS source directory.")
      endif()
    endif()

    # Clone the MADNESS repository
    if(NOT EXISTS ${MADNESS_SOURCE_DIR}/.git)
      message(STATUS "Pulling MADNESS from: ${Madness_URL}")
      set(error_code 1)
      set(number_of_tries 0)
      while(error_code AND number_of_tries LESS 3)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} clone ${Madness_URL} madness
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/external/src
            RESULT_VARIABLE error_code)
        math(EXPR number_of_tries "${number_of_tries} + 1")
      endwhile()
      if(number_of_tries GREATER 1)
        message(STATUS "Had to git clone more than once: ${number_of_tries} times.")
      endif()
      if(error_code)
        message(FATAL_ERROR "Failed to clone repository: '${Madness_URL}'")
      endif()
    endif()
  
  elseif(EXISTS "${MADNESS_SOURCE_DIR}")
    message(STATUS "Checking MADNESS source directory: ${MADNESS_SOURCE_DIR} - found")
  else()
    message(STATUS "Checking MADNESS source directory: ${MADNESS_SOURCE_DIR} - not found")
    message(FATAL_ERROR "Path to MADNESS source directory does not exist.")
  endif()
  
  if(EXISTS ${MADNESS_SOURCE_DIR}/.git)
    # Checkout the correct MADNESS revision
    set(error_code 1)
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" fetch
      COMMAND "${GIT_EXECUTABLE}" checkout ${Madness_TAG}
      COMMAND "./autogen.sh"
      WORKING_DIRECTORY "${MADNESS_SOURCE_DIR}"
      RESULT_VARIABLE error_code)
    if(error_code)
      message(FATAL_ERROR "Failed to checkout tag: '${Madness_TAG}'")
    endif()
        
    # Add madness-update target that will pull updates to the madness source 
    # from the git repository. This is done outside ExternalProject_add to 
    # prevent madness from doing a full pull, configure, and build everytime the 
    # project is built.
    add_custom_target(madness-update
      COMMAND ${GIT_EXECUTABLE} fetch
      COMMAND ${GIT_EXECUTABLE} checkout ${Madness_TAG}
      COMMAND "./autogen.sh"
      WORKING_DIRECTORY ${MADNESS_SOURCE_DIR}
      COMMENT "Updating source for 'madness' from ${MADNESS_URL}")

  endif()
  
  # Check that the MADNESS source contains madness.h
  message(STATUS "Looking for madness.h")
  if(EXISTS ${MADNESS_SOURCE_DIR}/src/madness.h)
    message(STATUS "Looking for madness.h - found")
  else()
    message(STATUS "Looking for madness.h - not found")
    message("The MADNESS source was not found in ${MADNESS_SOURCE_DIR}.")
    message("You can download the MADNESS source with:")
    message("$ git clone https://github.com/m-a-d-n-e-s-s/madness.git madness")
    message(FATAL_ERROR "MADNESS source not found.")
  endif()
  
  # Generate the MADNESS configure script if not present
  if(NOT EXISTS ${MADNESS_SOURCE_DIR}/configure)
    set(error_code 1)
    execute_process(
      COMMAND "./autogen.sh"
      WORKING_DIRECTORY "${MADNESS_SOURCE_DIR}"
      RESULT_VARIABLE error_code)
    if(error_code)
      message(FATAL_ERROR "Failed to generate MADNESS configure script.")
    endif()
  endif()

  
  # Setup configure variables
  
  # Set compile flags
  set(MAD_CPPFLAGS "${CMAKE_CPP_FLAGS}")
  set(MAD_CFLAGS "${CMAKE_C_FLAGS}")
  set(MAD_CXXFLAGS "${CMAKE_CXX_FLAGS}")
  set(MAD_LDFLAGS "${CMAKE_EXE_LINKER_FLAGS}")

  if(CMAKE_BUILD_TYPE)
    string(TOUPPER "${CMAKE_BUILD_TYPE}" MAD_BUILD_TYPE)
    append_flags(MAD_CFLAGS "${CMAKE_C_FLAGS_${MAD_BUILD_TYPE}}")
    append_flags(MAD_CXXFLAGS "${CMAKE_CXX_FLAGS_${MAD_BUILD_TYPE}}")
  endif()
  
  if(ENABLE_SHARED_LIBRARIES)
    set(MAD_ENABLE_SHARED "--enable-shared")
    set(MAD_ENABLE_STATIC "--disable-static")
  else()
    set(MAD_ENABLE_SHARED "--disable-shared")
    set(MAD_ENABLE_STATIC "--enable-static")
  endif()
  
#  message("${MAD_ENABLE_SHARED} ${MAD_ENABLE_STATIC}")

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
  
  # Set compile flags required for Intel TBB
  if(ENABLE_TBB)
    if(TBB_INCLUDE_DIR AND EXISTS ${TBB_INCLUDE_DIR})
      append_flags(MAD_TBB_INCLUDE_FLAG "--with-tbb-include=${TBB_INCLUDE_DIR}")
    endif()
    if(TBB_LIBRARY AND EXISTS ${TBB_LIBRARY})
      append_flags(MAD_TBB_LIB_FLAG "--with-tbb-lib=${TBB_LIBRARY}")
    endif()
    if(TBB_ROOT_DIR AND EXISTS ${TBB_ROOT_DIR})
      append_flags(MAD_TBB_FLAG "--with-tbb=${TBB_ROOT_DIR}")
    endif()
    if("${MAD_TBB_FLAG}" STREQUAL "")
      set(MAD_TBB_FLAG "--with-tbb=yes")
    endif()
  else()
    set(MAD_TBB_FLAG "--with-tbb=no")
  endif()
  
  
  # Set compile flags required for MPI
  if(ENABLE_MPI)
    foreach(_inc_dir ${MPI_INCLUDE_PATH})
      append_flags(MAD_CPPFLAGS "-I${_inc_dir}")
    endforeach()
    foreach(_lib ${MPI_LIBRARIES})
      append_flags(MAD_LIBS "${_lib}")
    endforeach()
#    append_flags(MAD_CFLAGS "${MPI_COMPILE_FLAGS}")
    append_flags(MAD_CXXFLAGS "${MPI_COMPILE_FLAGS}")
    append_flags(MAD_LDFLAGS "${MPI_LINK_FLAGS}") 
    set(MAD_STUB_MPI "no")
  else()
    set(MAD_STUB_MPI "yes")
  endif()
  
  # Set compile flags required for LAPACK, BLAS, and Pthreads
#  append_flags(MAD_LDFLAGS "${LAPACK_LINKER_FLAGS}")
#  append_flags(MAD_LDFLAGS "${BLAS_LINKER_FLAGS}")
  if(LAPACK_LIBRARIES OR BLAS_LIBRARIES)
    if(UNIX AND BLA_STATIC)
      append_flags(MAD_LIBS "-Wl,--start-group")
    endif()
      foreach(_lib ${LAPACK_LIBRARIES})
        append_flags(MAD_LIBS "${_lib}")
      endforeach()
      foreach(_lib ${BLAS_LIBRARIES})
        append_flags(MAD_LIBS "${_lib}")
      endforeach()
    if(UNIX AND BLA_STATIC)
      append_flags(MAD_LIBS "-Wl,--end-group")
    endif()
  endif()
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
  
  # Create or clean the build directory
  if(EXISTS "${MADNESS_BINARY_DIR}")
    set(error_code 1)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E remove -f "./*"
        WORKING_DIRECTORY ${MADNESS_BINARY_DIR}
        RESULT_VARIABLE error_code)
    if(error_code)
      message(FATAL_ERROR "Failed to delete existing files the MADNESS build directory.")
    endif()
  else()
    set(error_code 1)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${MADNESS_BINARY_DIR}"
        RESULT_VARIABLE error_code)
    if(error_code)
      message(FATAL_ERROR "Failed to create the MADNESS build directory.")
    endif()
  endif()


  # Configure MADNESS
  set(error_code 1)
  execute_process(
      COMMAND "${MADNESS_SOURCE_DIR}/configure"
                "--prefix=${CMAKE_INSTALL_PREFIX}"
                "${MAD_ENABLE_SHARED}"
                "${MAD_ENABLE_STATIC}"
                "--disable-debugging"
                "--disable-optimization"
                "--disable-optimal"
                "--disable-warning"
                "--enable-madex=${MAD_EXCEPTION}"
                "--enable-dependency-tracking"
                "--with-mpi-thread=multiple"
                "--with-fortran-int32=${MAD_F77_INT32}"
                "--with-stubmpi=${MAD_STUB_MPI}"
                "--with-elemental=${MAD_ELEMENTAL_FLAG}"
                "${MAD_TBB_FLAG}" 
                "${MAD_TBB_INCLUDE_FLAG}"
                "${MAD_TBB_LIB_FLAG}"
                "--without-mkl"
                "--without-libxc"
                "${MAD_EXTRA_CONFIGURE_FLAGS}"
                "MPICXX=${CMAKE_CXX_COMPILER}"
                "MPICC=${CMAKE_C_COMPILER}"
                "CPPFLAGS=${MAD_CPPFLAGS}"
                "CC=${CMAKE_C_COMPILER}" "CFLAGS=${MAD_CFLAGS}"
                "CXX=${CMAKE_CXX_COMPILER}" "CXXFLAGS=${MAD_CXXFLAGS}"
                "F77=${CMAKE_Fortran_COMPILER}" "FFLAGS=${CMAKE_Fortran_FLAGS}"
                "LDFLAGS=${MAD_LDFLAGS}"
                "LIBS=${MAD_LIBS}"
      WORKING_DIRECTORY "${MADNESS_BINARY_DIR}"
      RESULT_VARIABLE error_code)
  if(error_code)
    message(FATAL_ERROR "The MADNESS configure script failed.")
  endif(error_code)
  
  include(${MADNESS_BINARY_DIR}/config/madness-project.cmake)

#  message("Madness_INCLUDE_DIRS        = ${Madness_INCLUDE_DIRS}")
#  message("Madness_LIBRARIES           = ${Madness_LIBRARIES}")
#  message("Madness_MADchem_LIBRARY     = ${Madness_MADchem_LIBRARY}")
#  message("Madness_MADmra_LIBRARY      = ${Madness_MADmra_LIBRARY}")
#  message("Madness_MADtinyxml_LIBRARY  = ${Madness_MADtinyxml_LIBRARY}")
#  message("Madness_MADmuparser_LIBRARY = ${Madness_MADmuparser_LIBRARY}")
#  message("Madness_MADlinalg_LIBRARY   = ${Madness_MADlinalg_LIBRARY}")
#  message("Madness_MADtensor_LIBRARY   = ${Madness_MADtensor_LIBRARY}")
#  message("Madness_MADmisc_LIBRARY     = ${Madness_MADmisc_LIBRARY}")
#  message("Madness_MADworld_LIBRARY    = ${Madness_MADworld_LIBRARY}")
#  message("Madness_COMPILE_FLAGS       = ${Madness_COMPILE_FLAGS}")
#  message("Madness_LINKER_FLAGS        = ${Madness_LINKER_FLAGS}")
#  message("Madness_VERSION             = ${Madness_VERSION}")
#  message("Madness_F77_INTEGER_SIZE    = ${Madness_F77_INTEGER_SIZE}")
  
  # Removed all the flags passed to MADNESS configure
  string(REGEX REPLACE "-(O[0-9s]|g[0-9]?)([ ]+|$)" "" MAD_CXXFLAGS "${MAD_CXXFLAGS}")
  string(STRIP "${MAD_CXXFLAGS}" MAD_CXXFLAGS)
  string(REPLACE "${MAD_CXXFLAGS}" "" 
        Madness_COMPILE_FLAGS "${Madness_COMPILE_FLAGS}")
  string(REPLACE "${MAD_CPPFLAGS}" "" 
        Madness_COMPILE_FLAGS "${Madness_COMPILE_FLAGS}")
  string(STRIP "${Madness_COMPILE_FLAGS}" Madness_COMPILE_FLAGS)
  string(REPLACE "${MAD_CXXFLAGS}" "" 
        Madness_LINKER_FLAGS "${Madness_LINKER_FLAGS}")
  string(REPLACE "${MAD_LDFLAGS}" "" 
        Madness_LINKER_FLAGS "${Madness_LINKER_FLAGS}")
  string(REPLACE "${MAD_LIBS}" "" 
        Madness_LIBRARIES "${Madness_LIBRARIES}")
  string(STRIP "${Madness_LIBRARIES}" Madness_LIBRARIES)
  string(STRIP "${Madness_LINKER_FLAGS}" Madness_LINKER_FLAGS)
  
  # Removed the MADNESS libraries that are not needed by TiledArray.
  list(REMOVE_ITEM Madness_LIBRARIES 
    ${Madness_MADchem_LIBRARY} ${Madness_MADmra_LIBRARY}
    ${Madness_MADtinyxml_LIBRARY} ${Madness_MADmuparser_LIBRARY}
    ${Madness_MADlinalg_LIBRARY} ${Madness_MADtensor_LIBRARY}
    ${Madness_MADmisc_LIBRARY})
  
#  message("Madness_COMPILE_FLAGS = '${Madness_COMPILE_FLAGS}'")
#  message("Madness_LIBRARIES     = '${Madness_LIBRARIES}'")
#  message("Madness_LINKER_FLAGS  = '${Madness_LINKER_FLAGS}'")

  set(MAD_CPPFLAGS "${CMAKE_CPP_FLAGS}")
  set(MAD_CFLAGS "${CMAKE_C_FLAGS}")
  set(MAD_CXXFLAGS "${CMAKE_CXX_FLAGS}")
  set(MAD_LDFLAGS "${CMAKE_EXE_LINKER_FLAGS}")

  if(CMAKE_BUILD_TYPE)
    string(TOUPPER "${CMAKE_BUILD_TYPE}" MAD_BUILD_TYPE)
    append_flags(MAD_CFLAGS "${CMAKE_C_FLAGS_${MAD_BUILD_TYPE}}")
    append_flags(MAD_CXXFLAGS "")
  endif()
  add_custom_target(madness-build ALL
      COMMAND $(MAKE) world V=0
      WORKING_DIRECTORY ${MADNESS_BINARY_DIR}
      COMMENT Building 'madness')

  # Add madness-clean target that will delete files generated by MADNESS build.
  add_custom_target(madness-clean
    COMMAND ${CMAKE_MAKE_PROGRAM} clean
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
  add_dependencies(External madness-build)

  # Set config variables 
  set(TiledArray_CONFIG_LIBRARIES ${Madness_LIBRARIES}
      ${TiledArray_CONFIG_LIBRARIES})

endif()


include_directories(${Madness_INCLUDE_DIRS})
set(TiledArray_LIBRARIES ${Madness_LIBRARIES} ${TiledArray_LIBRARIES})
append_flags(CMAKE_CXX_FLAGS "${Madness_COMPILE_FLAGS}")
append_flags(CMAKE_EXE_LINKER_FLAGS "${Madness_LINKER_FLAGS}")

