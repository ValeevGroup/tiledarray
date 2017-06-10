# -*- mode: cmake -*-

###################
# Find MADNESS
###################

# extra preprocessor definitions that MADNESS needs from TiledArray
set(MADNESS_EXTRA_CPP_FLAGS "")

include(ExternalProject)
include(ConvertIncludesListToCompilerArgs)
include(ConvertLibrariesListToCompilerArgs)

find_package(MADNESS 0.10.1 CONFIG QUIET COMPONENTS world HINTS ${MADNESS_ROOT_DIR})

macro(replace_mad_targets_with_libnames _mad_libraries _mad_config_libs)
  set(${_mad_config_libs} )
  foreach (_lib ${${_mad_libraries}})
    if (${_lib} MATCHES "MAD*")
      set(${_mad_config_libs} "${${_mad_config_libs}} -l${_lib}")
    else ()
      set(${_mad_config_libs} "${${_mad_config_libs}} ${_lib}")
    endif()
  endforeach()
endmacro()

if(MADNESS_FOUND)

  cmake_push_check_state()

  set(MADNESS_CONFIG_DIR ${MADNESS_DIR})
  
  list(APPEND CMAKE_REQUIRED_INCLUDES ${MADNESS_INCLUDE_DIRS} ${TiledArray_CONFIG_INCLUDE_DIRS})
  list(APPEND CMAKE_REQUIRED_LIBRARIES "${MADNESS_LINKER_FLAGS}" ${MADNESS_LIBRARIES}
      "${CMAKE_EXE_LINKER_FLAGS}" ${TiledArray_CONFIG_LIBRARIES})
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} ${MADNESS_COMPILE_FLAGS} ${CMAKE_CXX_FLAGS} ${MADNESS_EXTRA_CPP_FLAGS}")

  # sanity check: try compiling a simple program
  CHECK_CXX_SOURCE_COMPILES(
    "
    #include <madness/world/world.h>
    int main(int argc, char** argv) {
      madness::World& world = madness::initialize(argc, argv);
      madness::finalize();
      return 0;
    }
    "  MADNESS_COMPILES)

  if (NOT MADNESS_COMPILES)
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
        "
        MADNESS_HAS_ELEMENTAL_SUPPORT
     )
        
    if(NOT MADNESS_HAS_ELEMENTAL_SUPPORT)
      message(FATAL_ERROR "MADNESS does not include Elemental support.")
    endif() 
    
    set(TILEDARRAY_HAS_ELEMENTAL ${MADNESS_HAS_ELEMENTAL_SUPPORT})
  endif()

  cmake_pop_check_state()

  # Set config variables
  list(APPEND TiledArray_CONFIG_INCLUDE_DIRS ${MADNESS_INCLUDE_DIRS})
  replace_mad_targets_with_libnames(MADNESS_LIBRARIES MADNESS_CONFIG_LIBRARIES)
  list(APPEND TiledArray_CONFIG_LIBRARIES ${MADNESS_CONFIG_LIBRARIES})
  set(TiledArray_LIBRARIES ${MADNESS_LIBRARIES} ${TiledArray_LIBRARIES})

  
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
  set(MADNESS_URL "https://github.com/m-a-d-n-e-s-s/madness.git" CACHE STRING 
        "Path to the MADNESS repository")
  set(MADNESS_TAG "01c1f2fde546d0737a06e58c396ad35947bdfa4b" CACHE STRING 
        "Revision hash or tag to use when building MADNESS")
  
  if("${MADNESS_TAG}" STREQUAL "")
    message(FATAL_ERROR "Invalid value given for MADNESS_TAG; specify a valid hash or tag.")
  endif()
  
  # Setup configure variables
  
  # Set Fortran integer size
  if(INTEGER4) 
    set(F77_INT_SIZE 4)
  else(INTEGER4)
    set (F77_INT_SIZE 8)
  endif(INTEGER4)
  
  # Set error handling method (for TA_DEFAULT_ERROR values see top-level CMakeLists.txt)
  if(TA_DEFAULT_ERROR EQUAL 0)
    set(_MAD_ASSERT_TYPE disable)
  elseif(TA_DEFAULT_ERROR EQUAL 1)
    set(_MAD_ASSERT_TYPE throw)
  elseif(TA_DEFAULT_ERROR EQUAL 2)
    set(_MAD_ASSERT_TYPE assert)
  elseif(TA_DEFAULT_ERROR EQUAL 3)
    set(_MAD_ASSERT_TYPE abort)
  endif()
  set(MAD_ASSERT_TYPE ${_MAD_ASSERT_TYPE} CACHE INTERNAL "MADNESS assert type")
  
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
      message(STATUS "Pulling MADNESS from: ${MADNESS_URL}")
      set(error_code 1)
      set(number_of_tries 0)
      while(error_code AND number_of_tries LESS 3)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} clone ${MADNESS_URL} madness
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/external/src
            RESULT_VARIABLE error_code)
        math(EXPR number_of_tries "${number_of_tries} + 1")
      endwhile()
      if(number_of_tries GREATER 1)
        message(STATUS "Had to git clone more than once: ${number_of_tries} times.")
      endif()
      if(error_code)
        message(FATAL_ERROR "Failed to clone repository: '${MADNESS_URL}'")
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
      COMMAND "${GIT_EXECUTABLE}" fetch origin master
      COMMAND "${GIT_EXECUTABLE}" checkout ${MADNESS_TAG}
      WORKING_DIRECTORY "${MADNESS_SOURCE_DIR}"
      RESULT_VARIABLE error_code)
    if(error_code)
      message(FATAL_ERROR "Failed to checkout tag: '${MADNESS_TAG}'")
    endif()
        
    # Add update-madness target that will pull updates to the madness source 
    # from the git repository. This is done outside ExternalProject_add to 
    # prevent madness from doing a full pull, configure, and build everytime the 
    # project is built.
    add_custom_target(update-madness
      COMMAND ${GIT_EXECUTABLE} fetch origin master
      COMMAND ${GIT_EXECUTABLE} checkout ${MADNESS_TAG}
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

  # if ELEMENTAL_TAG provided, package pass it on to MADNESS
  set(MAD_ELEMENTAL_OPTIONS -DENABLE_ELEMENTAL=${ENABLE_ELEMENTAL})
  if (DEFINED ELEMENTAL_TAG)
    set(MAD_ELEMENTAL_OPTIONS -DELEMENTAL_TAG=${ELEMENTAL_TAG} ${MAD_ELEMENTAL_OPTIONS})
  endif (DEFINED ELEMENTAL_TAG)
  if (DEFINED ELEMENTAL_URL)
    set(MAD_ELEMENTAL_OPTIONS -DELEMENTAL_URL=${ELEMENTAL_URL} ${MAD_ELEMENTAL_OPTIONS})
  endif (DEFINED ELEMENTAL_URL)
  
  # update all CMAKE_CXX_FLAGS to include extra preprocessor flags MADNESS needs
  set(CMAKE_CXX_FLAGS                "${CMAKE_CXX_FLAGS} ${MADNESS_EXTRA_CPP_FLAGS}")
  set(CMAKE_CXX_FLAGS_DEBUG          "${CMAKE_CXX_FLAGS_DEBUG} ${MADNESS_EXTRA_CPP_FLAGS}")
  set(CMAKE_CXX_FLAGS_RELEASE        "${CMAKE_CXX_FLAGS_RELEASE} ${MADNESS_EXTRA_CPP_FLAGS}")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${MADNESS_EXTRA_CPP_FLAGS}")
  set(CMAKE_CXX_FLAGS_MINSIZEREL     "${CMAKE_CXX_FLAGS_MINSIZEREL} ${MADNESS_EXTRA_CPP_FLAGS}")
  
  set(error_code 1)
  message (STATUS "** Configuring MADNESS")
  message (STATUS "MADNESS Extra Args: ${MADNESS_CMAKE_EXTRA_ARGS}") 
  execute_process(
      COMMAND ${CMAKE_COMMAND}
      ARGS
      ${MADNESS_SOURCE_DIR}
      -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
      "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
      -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      "-DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}"
      "-DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}"
      "-DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}"
      "-DCMAKE_C_FLAGS_RELWITHDEBINFO=${CMAKE_C_FLAGS_RELWITHDEBINFO}"
      "-DCMAKE_C_FLAGS_MINSIZEREL=${CMAKE_C_FLAGS_MINSIZEREL}"
      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
      "-DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}"
      "-DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}"
      "-DCMAKE_CXX_FLAGS_RELWITHDEBINFO=${CMAKE_CXX_FLAGS_RELWITHDEBINFO}"
      "-DCMAKE_CXX_FLAGS_MINSIZEREL=${CMAKE_CXX_FLAGS_MINSIZEREL}"
# F Fortran, assume we can link without its runtime
# if you need Fortran checks enable Elemental
#      -DCMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER}
#      "-DCMAKE_Fortran_FLAGS=${CMAKE_Fortran_FLAGS}"
#      "-DCMAKE_Fortran_FLAGS_DEBUG=${CMAKE_Fortran_FLAGS_DEBUG}"
#      "-DCMAKE_Fortran_FLAGS_RELEASE=${CMAKE_Fortran_FLAGS_RELEASE}"
#      "-DCMAKE_Fortran_FLAGS_RELWITHDEBINFO=${CMAKE_Fortran_FLAGS_RELWITHDEBINFO}"
#      "-DCMAKE_Fortran_FLAGS_MINSIZEREL=${CMAKE_Fortran_FLAGS_MINSIZEREL}"
      -DENABLE_MPI=${ENABLE_MPI}
      -DMPI_THREAD=multiple
      -DMPI_CXX_COMPILER=${MPI_CXX_COMPILER}
      -DMPI_C_COMPILER=${MPI_C_COMPILER}
      -DENABLE_TBB=${ENABLE_TBB}
      "-DTBB_ROOT_DIR=${TBB_ROOT_DIR}"
      ${MAD_ELEMENTAL_OPTIONS}
      -DENABLE_MKL=${ENABLE_MKL}
      -DFORTRAN_INTEGER_SIZE=${F77_INT_SIZE}
      -DENABLE_LIBXC=FALSE
      "-DLAPACK_LIBRARIES=${LAPACK_LIBRARIES}"
      -DENABLE_GPERFTOOLS=${ENABLE_GPERFTOOLS}
      -DASSERTION_TYPE=${MAD_ASSERT_TYPE}
      "-DCMAKE_EXE_LINKER_FLAGS=${MAD_LDFLAGS}"
      -DDISABLE_WORLD_GET_DEFAULT=ON
      ${MADNESS_CMAKE_EXTRA_ARGS}
      WORKING_DIRECTORY "${MADNESS_BINARY_DIR}"
      RESULT_VARIABLE error_code)
  if(error_code)
    message(FATAL_ERROR "The MADNESS cmake configuration failed.")
  else(error_code)
    message (STATUS "** Done configuring MADNESS")
  endif(error_code)

  set(MADNESS_DIR ${MADNESS_BINARY_DIR})
  find_package(MADNESS 0.10.1 CONFIG REQUIRED
               COMPONENTS world HINTS ${MADNESS_BINARY_DIR})
  set(TILEDARRAY_HAS_ELEMENTAL ${ENABLE_ELEMENTAL})
  
  # TiledArray only needs MADworld library compiled to be ...
  # as long as you mark dependence on it correcty its target properties
  # will be used correctly (header locations, etc.)
  set(MADNESS_LIBRARIES ${MADNESS_world_LIBRARY})
  # BUT it also need cblas/clapack headers ... these are not packaged into a library with a target
  # these headers depend on LAPACK which is a dependency of MADlinalg, hence
  # add MADlinalg's include dirs to MADNESS_INCLUDE_DIRS and MADNESS's LAPACK_LIBRARIES to MADNESS_LINKER_FLAGS (!)
  list(APPEND MADNESS_LIBRARIES "${LAPACK_LIBRARIES}")
  # this is not necessary since we use nested #include paths in build and install trees,
  # hence dependence on MADworld should provide proper include paths for ALL madness libs ...
  #list(APPEND MADNESS_INCLUDE_DIRS $<TARGET_PROPERTY:MADlinalg,INTERFACE_INCLUDE_DIRECTORIES>)
  # external Elemental is *installed* to be usable, hence need to add the path to the install tree
  if (DEFINED ELEMENTAL_TAG)
    list(APPEND MADNESS_INCLUDE_DIRS ${CMAKE_INSTALL_PREFIX}/include)
  endif (DEFINED ELEMENTAL_TAG)
  
  
  # build MADNESS components .. only MADworld here! Headers from MADlinalg do not need compilation
  add_custom_target(build-madness ALL
      COMMAND ${CMAKE_COMMAND} --build . --target MADworld
      WORKING_DIRECTORY ${MADNESS_BINARY_DIR}
      COMMENT Building 'madness')

  if (ENABLE_ELEMENTAL)
    set(ELEMENTAL_CLEAN_TARGET clean-elemental)
    set(ELEMENTAL_INSTALL_TARGET \"install-elemental\")
  else (ENABLE_ELEMENTAL)
    set(ELEMENTAL_CLEAN_TARGET )
    set(ELEMENTAL_INSTALL_TARGET )
  endif (ENABLE_ELEMENTAL)

  # Add clean-madness target that will delete files generated by MADNESS build.
  add_custom_target(clean-madness
    COMMAND ${CMAKE_MAKE_PROGRAM} clean ${ELEMENTAL_CLEAN_TARGET}
    WORKING_DIRECTORY ${MADNESS_BINARY_DIR}
    COMMENT Cleaning build directory for 'madness')
  
  # Since 'install-madness' target cannot be linked to the 'install' target,
  # we will do it manually here.
  install(CODE
      "
      execute_process(
          COMMAND \"${CMAKE_MAKE_PROGRAM}\" \"install-world\" \"install-clapack\" \"install-config\" \"install-common\" ${ELEMENTAL_INSTALL_TARGET}
          WORKING_DIRECTORY \"${MADNESS_BINARY_DIR}\"
          RESULT_VARIABLE error_code)
      if(error_code)
        message(FATAL_ERROR \"Failed to install 'madness'\")
      endif()
      "
      )

  # Set build dependencies and compiler arguments
  add_dependencies(External build-madness)

  # Set config variables
  replace_mad_targets_with_libnames(MADNESS_LIBRARIES MADNESS_CONFIG_LIBRARIES)
  list(APPEND TiledArray_CONFIG_LIBRARIES ${MADNESS_CONFIG_LIBRARIES})

endif()


include_directories(${MADNESS_INCLUDE_DIRS})
list (APPEND TiledArray_LIBRARIES ${MADNESS_LIBRARIES})
append_flags(CMAKE_CXX_FLAGS "${MADNESS_COMPILE_FLAGS}")
append_flags(CMAKE_EXE_LINKER_FLAGS "${MADNESS_LINKER_FLAGS}")

