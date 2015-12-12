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
  set(Madness_TAG "HEAD" CACHE STRING 
        "Revision hash or tag to use when building MADNESS")
  
  if("${Madness_TAG}" STREQUAL "")
    message(FATAL_ERROR "Invalid value given for Madness_TAG; specify a valid hash or tag.")
  endif()
  
  # Setup configure variables
  
  # Set compile flags
  set(MAD_CPPFLAGS ${CMAKE_CPP_FLAGS})
  set(MAD_CFLAGS ${CMAKE_C_FLAGS})
  set(MAD_CXXFLAGS ${CMAKE_CXX_FLAGS})
  set(MAD_LDFLAGS ${CMAKE_EXE_LINKER_FLAGS})

  if(CMAKE_BUILD_TYPE)
    string(TOUPPER "${CMAKE_BUILD_TYPE}" MAD_BUILD_TYPE)
    list(APPEND MAD_CFLAGS ${CMAKE_C_FLAGS_${MAD_BUILD_TYPE}})
    list(APPEND MAD_CXXFLAGS ${CMAKE_CXX_FLAGS_${MAD_BUILD_TYPE}})
  endif()
  
  # Set Fortran integer size
  if(INTEGER4) 
    set(F77_INT_SIZE 4)
  else(INTEGER4)
    set (F77_INT_SIZE 8)
  endif(INTEGER4)
  
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

  
  set(error_code 1)
  message (STATUS "** Configuring MADNESS")
  execute_process(
      COMMAND ${CMAKE_COMMAND}
      ARGS
      ${MADNESS_SOURCE_DIR}
      -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
      -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
      -DCMAKE_CPP_FLAGS=${MAD_CPPFLAGS}
      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      -DCMAKE_C_FLAGS=${MAD_CFLAGS}
      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
# F Fortran, assume we can link without its runtime
#      -DCMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER}
#      -DCMAKE_Fortran_FLAGS=${CMAKE_Fortran_FLAGS}
      -DENABLE_MPI=${ENABLE_MPI}
      -DMPI_THREAD=multiple
      -DMPI_CXX_COMPILER=${MPI_CXX_COMPILER}
      -DMPI_C_COMPILER=${MPI_C_COMPILER}
      -DENABLE_ELEMENTAL=${ENABLE_ELEMENTAL}
      -DENABLE_MKL=${ENABLE_MKL}
      -DFORTRAN_INTEGER_SIZE=${F77_INT_SIZE}
      -DENABLE_LIBXC=FALSE
      -DENABLE_GPERFTOOLS=FALSE
      -DASSERTION_TYPE=${MAD_EXCEPTION}
#      -DCMAKE_CXX_FLAGS=${MAD_CXXFLAGS}
#      -DCMAKE_EXE_LINKER_FLAGS=${MAD_LDFLAGS}
#      -DCMAKE_REQUIRED_LIBRARIES=${MAD_LIBS}
      WORKING_DIRECTORY "${MADNESS_BINARY_DIR}"
      RESULT_VARIABLE error_code)
  if(error_code)
    message(FATAL_ERROR "The MADNESS configure script failed.")
  else(error_code)
    message (STATUS "** Done configuring MADNESS")
  endif(error_code)

  set(MADNESS_DIR ${MADNESS_BINARY_DIR})
  find_package(MADNESS 0.10.0 REQUIRED
               COMPONENTS world)

 message("Madness_VERSION             = ${Madness_VERSION}")
 message("Madness_INCLUDE_DIRS        = ${Madness_INCLUDE_DIRS}")
 message("Madness_LIBRARIES           = ${Madness_LIBRARIES}")
 message("Madness_F77_INTEGER_SIZE    = ${Madness_F77_INTEGER_SIZE}")
  
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
    ${Madness_chem_LIBRARY} ${Madness_mra_LIBRARY}
    ${Madness_tinyxml_LIBRARY} ${Madness_muparser_LIBRARY}
    ${Madness_linalg_LIBRARY} ${Madness_tensor_LIBRARY}
    ${Madness_misc_LIBRARY})
  
  if(CMAKE_BUILD_TYPE)
    string(TOUPPER "${CMAKE_BUILD_TYPE}" MAD_BUILD_TYPE)
    append_flags(MAD_CFLAGS "${CMAKE_C_FLAGS_${MAD_BUILD_TYPE}}")
    append_flags(MAD_CXXFLAGS "")
  endif()
  add_custom_target(madness-build ALL
      COMMAND ${CMAKE_COMMAND} --build . --target MADworld
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
          COMMAND \"${CMAKE_MAKE_PROGRAM}\" \"install-world\"
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

