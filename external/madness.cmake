# -*- mode: cmake -*-

###################
# Find MADNESS
###################

# extra compiler flags that MADNESS needs from TiledArray
set(MADNESS_EXTRA_CXX_FLAGS "")

include(ExternalProject)
include(ConvertIncludesListToCompilerArgs)
include(ConvertLibrariesListToCompilerArgs)

# user specified madness
set(MADNESS_TAG "" CACHE STRING "Revision hash or tag to use when building MADNESS")
mark_as_advanced(FORCE MADNESS_TAG)

set(_madness_tag ${TA_TRACKED_MADNESS_TAG})

if (MADNESS_TAG)
  set(_madness_tag ${MADNESS_TAG})
endif()

find_package_regimport(MADNESS ${TA_TRACKED_MADNESS_VERSION} CONFIG QUIET COMPONENTS world HINTS ${MADNESS_ROOT_DIR})

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

# if found, make sure the MADNESS tag matches exactly
if (MADNESS_FOUND AND NOT TILEDARRAY_DOWNLOADED_MADNESS)
  set(TILEDARRAY_DOWNLOADED_MADNESS OFF CACHE BOOL "Whether TA downloaded MADNESS")
  mark_as_advanced(TILEDARRAY_DOWNLOADED_MADNESS)

  set(CONFIG_H_PATH "${MADNESS_DIR}/../../../include/madness/config.h")  # if MADNESS were installed
  if (NOT EXISTS "${CONFIG_H_PATH}")
    set(CONFIG_H_PATH "${MADNESS_DIR}/src/madness/config.h")  # if MADNESS were used from build tree
    if (NOT EXISTS "${CONFIG_H_PATH}")
      message(FATAL_ERROR "did not find MADNESS' config.h")
    endif()
  endif()
  file(STRINGS "${CONFIG_H_PATH}" MADNESS_REVISION_LINE REGEX "define MADNESS_REVISION")
  if (MADNESS_REVISION_LINE) # MADNESS_REVISION found? make sure it matches the required tag exactly
    string(REGEX REPLACE ".*define[ \t]+MADNESS_REVISION[ \t]+\"([a-z0-9]+)\"" "\\1" MADNESS_REVISION "${MADNESS_REVISION_LINE}")
    if ("${MADNESS_REVISION}" STREQUAL "${_madness_tag}")
      message(STATUS "Found MADNESS with required revision ${MADNESS_REQUIRED_TAG}")
    else()
      message(FATAL_ERROR "Found MADNESS with revision ${MADNESS_REVISION}, but ${MADNESS_REQUIRED_TAG} is required; if MADNESS was built by TiledArray, remove the TiledArray install directory, else build the required revision of MADNESS")
    endif()
  else (MADNESS_REVISION_LINE) # MADNESS_REVISION not found? MADNESS is not recent enough, reinstall
    message(FATAL_ERROR "Found MADNESS, but it is not recent enough; either provide MADNESS with revision ${TA_TRACKED_MADNESS_TAG} or let TiledArray built it")
  endif(MADNESS_REVISION_LINE)

  if ((NOT TA_ASSUMES_ASLR_DISABLED AND MADNESS_ASSUMES_ASLR_DISABLED) OR (TA_ASSUMES_ASLR_DISABLED AND NOT MADNESS_ASSUMES_ASLR_DISABLED))
    message(FATAL_ERROR "Found MADNESS configured with MADNESS_ASSUMES_ASLR_DISABLED=${MADNESS_ASSUMES_ASLR_DISABLED} but TA is configured with TA_ASSUMES_ASLR_DISABLED=${TA_ASSUMES_ASLR_DISABLED}; MADNESS_ASSUMES_ASLR_DISABLED and TA_ASSUMES_ASLR_DISABLED should be the same")
  endif()

  cmake_push_check_state()

  set(MADNESS_CONFIG_DIR ${MADNESS_DIR})

  list(APPEND CMAKE_REQUIRED_INCLUDES ${MADNESS_INCLUDE_DIRS} ${TiledArray_CONFIG_INCLUDE_DIRS})
  list(APPEND CMAKE_REQUIRED_LIBRARIES "${MADNESS_LINKER_FLAGS}" ${MADNESS_LIBRARIES}
      "${CMAKE_EXE_LINKER_FLAGS}" ${TiledArray_CONFIG_LIBRARIES})
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} ${MADNESS_COMPILE_FLAGS} ${CMAKE_CXX_FLAGS} ${MADNESS_EXTRA_CXX_FLAGS}")

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
    
  # ensure fresh MADNESS
  CHECK_CXX_SOURCE_COMPILES(
        "
    #include <madness/world/world.h>
    #include <madness/world/worldmem.h>
    int main(int argc, char** argv) {
      // test 1
      madness::print_meminfo_enable();
      // test 2
      madness::World::is_default(SafeMPI::COMM_WORLD);
      return 0;
    }
    "  MADNESS_IS_FRESH)

  if (NOT MADNESS_IS_FRESH)
    message(FATAL_ERROR "MADNESS is not fresh enough; update to ${_madness_tag}")
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

  # look for C and MPI here to make troubleshooting easier and be able to override defaults for MADNESS
  enable_language(C)
  find_package(MPI REQUIRED COMPONENTS C CXX)

  find_package(Git REQUIRED)
  message(STATUS "git found: ${GIT_EXECUTABLE}")

  # Create a cache entry for MADNESS build variables.
  # Note: This will not overwrite user specified values.
  set(MADNESS_SOURCE_DIR "${PROJECT_BINARY_DIR}/external/madness-src" CACHE PATH
        "Path to the MADNESS source directory")
  set(MADNESS_BINARY_DIR "${PROJECT_BINARY_DIR}/external/madness-build" CACHE PATH
        "Path to the MADNESS build directory")
  set(MADNESS_URL "https://github.com/m-a-d-n-e-s-s/madness.git" CACHE STRING 
        "Path to the MADNESS repository")
  
  # Setup configure variables

  # Set error handling method (for TA_ASSERT_POLICY allowed values see top-level CMakeLists.txt)
  if(TA_ASSERT_POLICY STREQUAL TA_ASSERT_IGNORE)
    set(_MAD_ASSERT_TYPE disable)
  elseif(TA_ASSERT_POLICY STREQUAL TA_ASSERT_THROW)
    set(_MAD_ASSERT_TYPE throw)
  elseif(TA_ASSERT_POLICY STREQUAL TA_ASSERT_ABORT)
    set(_MAD_ASSERT_TYPE abort)
  endif()
  set(MAD_ASSERT_TYPE ${_MAD_ASSERT_TYPE} CACHE INTERNAL "MADNESS assert type")
  
  # Check the MADNESS source directory to make sure it contains the source files
  # If the MADNESS source directory is the default location and does not exist,
  # MADNESS will be downloaded from git.
  message(STATUS "Checking MADNESS source directory: ${MADNESS_SOURCE_DIR}")
  if("${MADNESS_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}/external/madness-src")

    # Create the external source directory
    if(NOT EXISTS ${PROJECT_BINARY_DIR}/external)
      set(error_code 1)
      execute_process(
          COMMAND "${CMAKE_COMMAND}" -E make_directory "${PROJECT_BINARY_DIR}/external"
          RESULT_VARIABLE error_code)
      if(error_code)
        message(FATAL_ERROR "Failed to create directory \"${PROJECT_BINARY_DIR}/external\"")
      endif()
    endif()

    # Clone the MADNESS repository
    if(NOT EXISTS ${MADNESS_SOURCE_DIR}/.git)
      message(STATUS "Pulling MADNESS from: ${MADNESS_URL}")
      set(error_code 1)
      set(number_of_tries 0)
      while(error_code AND number_of_tries LESS 3)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} clone ${MADNESS_URL} madness-src
            WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/external
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
      COMMAND "${GIT_EXECUTABLE}" checkout ${_madness_tag}
      WORKING_DIRECTORY "${MADNESS_SOURCE_DIR}"
      RESULT_VARIABLE error_code)
    if(error_code)
      message(FATAL_ERROR "Failed to checkout tag: '${_madness_tag}'")
    endif()
        
    # Add update-madness target that will pull updates to the madness source 
    # from the git repository. This is done outside ExternalProject_add to 
    # prevent madness from doing a full pull, configure, and build everytime the 
    # project is built.
    add_custom_target(update-madness
      COMMAND ${GIT_EXECUTABLE} fetch origin master
      COMMAND ${GIT_EXECUTABLE} checkout ${_madness_tag}
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

  set(MADNESS_CMAKE_GENERATOR "${CMAKE_GENERATOR}" CACHE STRING "CMake generator to use for compiling MADNESS")

  # update all CMAKE_CXX_FLAGS to include extra preprocessor flags MADNESS needs
  set(CMAKE_CXX_FLAGS                "${CMAKE_CXX_FLAGS} ${MADNESS_EXTRA_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS_DEBUG          "${CMAKE_CXX_FLAGS_DEBUG} ${MADNESS_EXTRA_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS_RELEASE        "${CMAKE_CXX_FLAGS_RELEASE} ${MADNESS_EXTRA_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${MADNESS_EXTRA_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS_MINSIZEREL     "${CMAKE_CXX_FLAGS_MINSIZEREL} ${MADNESS_EXTRA_CXX_FLAGS}")

  set(MADNESS_CMAKE_ARGS
          -DMADNESS_BUILD_MADWORLD_ONLY=ON
          -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
          -DMADNESS_ASSUMES_ASLR_DISABLED=${TA_ASSUMES_ASLR_DISABLED}
          -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}
          -DCMAKE_POSITION_INDEPENDENT_CODE=${CMAKE_POSITION_INDEPENDENT_CODE}
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
          -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
          -DCMAKE_CXX_EXTENSIONS=${CMAKE_CXX_EXTENSIONS}
          # F Fortran, assume we can link without its runtime
          #      -DCMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER}
          #      "-DCMAKE_Fortran_FLAGS=${CMAKE_Fortran_FLAGS}"
          #      "-DCMAKE_Fortran_FLAGS_DEBUG=${CMAKE_Fortran_FLAGS_DEBUG}"
          #      "-DCMAKE_Fortran_FLAGS_RELEASE=${CMAKE_Fortran_FLAGS_RELEASE}"
          #      "-DCMAKE_Fortran_FLAGS_RELWITHDEBINFO=${CMAKE_Fortran_FLAGS_RELWITHDEBINFO}"
          #      "-DCMAKE_Fortran_FLAGS_MINSIZEREL=${CMAKE_Fortran_FLAGS_MINSIZEREL}"
          -DCMAKE_AR=${CMAKE_AR}
          -DENABLE_MPI=${ENABLE_MPI}
          -DMPI_THREAD=multiple
          -DMPI_CXX_COMPILER=${MPI_CXX_COMPILER}
          -DMPI_C_COMPILER=${MPI_C_COMPILER}
          -DMPI_CXX_SKIP_MPICXX=ON  # introduced in cmake 3.10, disables search for C++ MPI-2 bindings
          -DENABLE_TBB=${ENABLE_TBB}
          "-DTBB_ROOT_DIR=${TBB_ROOT_DIR}"
          -DENABLE_GPERFTOOLS=${ENABLE_GPERFTOOLS}
          -DENABLE_TCMALLOC_MINIMAL=${ENABLE_TCMALLOC_MINIMAL}
          -DENABLE_LIBUNWIND=${ENABLE_LIBUNWIND}
          -DASSERTION_TYPE=${MAD_ASSERT_TYPE}
          "-DCMAKE_EXE_LINKER_FLAGS=${MAD_LDFLAGS}"
          -DDISABLE_WORLD_GET_DEFAULT=ON
          -DENABLE_MEM_PROFILE=ON
          "-DENABLE_TASK_DEBUG_TRACE=${TILEDARRAY_ENABLE_TASK_DEBUG_TRACE}"
          ${MADNESS_CMAKE_EXTRA_ARGS})

  if (CMAKE_TOOLCHAIN_FILE)
    if (IS_ABSOLUTE CMAKE_TOOLCHAIN_FILE)
      set(absolute_toolchain_file_path "${CMAKE_TOOLCHAIN_FILE}")
    else(IS_ABSOLUTE CMAKE_TOOLCHAIN_FILE)
      # try relative to (TA) project source dir first, then to binary dir
      get_filename_component(absolute_toolchain_file_path "${CMAKE_TOOLCHAIN_FILE}" ABSOLUTE
          BASE_DIR "${PROJECT_SOURCE_DIR}")
      if (NOT absolute_toolchain_file_path OR NOT EXISTS "${absolute_toolchain_file_path}")
        get_filename_component(absolute_toolchain_file_path "${CMAKE_TOOLCHAIN_FILE}" ABSOLUTE
            BASE_DIR "${PROJECT_BINARY_DIR}")
      endif()
      # better give up, if cannot resolve, then end up with MADNESS built with a different toolchain
      if (NOT absolute_toolchain_file_path OR NOT EXISTS "${absolute_toolchain_file_path}")
        message(FATAL_ERROR "could not resolve the absolute path to the toolchain file: CMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}; specify the CMAKE_TOOLCHAIN_FILE as the absolute path to work around")
      endif()
    endif(IS_ABSOLUTE CMAKE_TOOLCHAIN_FILE)
    set(MADNESS_CMAKE_ARGS  "${MADNESS_CMAKE_ARGS}"
        "-DCMAKE_TOOLCHAIN_FILE=${absolute_toolchain_file_path}")
  endif(CMAKE_TOOLCHAIN_FILE)

  set(error_code 1)
  message (STATUS "** Configuring MADNESS")
  message (STATUS "MADNESS CMake generator: ${MADNESS_CMAKE_GENERATOR}")
  message (STATUS "MADNESS CMake Arguments: ${MADNESS_CMAKE_ARGS}")
  execute_process(
      COMMAND ${CMAKE_COMMAND}
      "${MADNESS_SOURCE_DIR}"
      -G "${MADNESS_CMAKE_GENERATOR}"
      ${MADNESS_CMAKE_ARGS}
      WORKING_DIRECTORY "${MADNESS_BINARY_DIR}"
      RESULT_VARIABLE error_code)
  if(error_code)
    message(FATAL_ERROR "The MADNESS cmake configuration failed.")
  else(error_code)
    message (STATUS "** Done configuring MADNESS")
  endif(error_code)

  set(MADNESS_DIR ${MADNESS_BINARY_DIR})
  find_package_regimport(MADNESS ${TA_TRACKED_MADNESS_VERSION} CONFIG REQUIRED
                         COMPONENTS world HINTS ${MADNESS_BINARY_DIR})
  if (NOT TARGET MADworld)
    message(FATAL_ERROR "Did not receive target MADworld")
  endif()
  if (MADNESS_CMAKE_EXTRA_ARGS MATCHES -DENABLE_PARSEC=ON)
    if (NOT TARGET PaRSEC::parsec)
      find_package_regimport(PaRSEC CONFIG REQUIRED COMPONENTS parsec)
    endif()
  endif()
  set(TILEDARRAY_DOWNLOADED_MADNESS ON CACHE BOOL "Whether TA downloaded MADNESS")
  mark_as_advanced(TILEDARRAY_DOWNLOADED_MADNESS)

  # TiledArray only needs MADworld library compiled to be ...
  # as long as you mark dependence on it correcty its target properties
  # will be used correctly (header locations, etc.)
  set(MADNESS_WORLD_LIBRARY MADworld)
  if (BUILD_SHARED_LIBS)
    set(MADNESS_DEFAULT_LIBRARY_SUFFIX ${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(MADNESS_EL_DEFAULT_LIBRARY_ABI_SUFFIX ".88-dev")
  else(BUILD_SHARED_LIBS)
    set(MADNESS_DEFAULT_LIBRARY_SUFFIX ${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(MADNESS_EL_DEFAULT_LIBRARY_ABI_SUFFIX "")
  endif(BUILD_SHARED_LIBS)
  set(MADNESS_LIBRARIES ${MADNESS_WORLD_LIBRARY})

  # custom target for building MADNESS components .. only MADworld here!
  # N.B. Ninja needs spelling out the byproducts of custom targets, see https://cmake.org/cmake/help/v3.3/policy/CMP0058.html
  set(MADNESS_BUILD_BYPRODUCTS "${MADNESS_BINARY_DIR}/src/madness/world/lib${MADNESS_WORLD_LIBRARY}${CMAKE_STATIC_LIBRARY_SUFFIX}")
  message(STATUS "custom target build-madness is expected to build these byproducts: ${MADNESS_BUILD_BYPRODUCTS}")
  add_custom_target(build-madness ALL
      COMMAND ${CMAKE_COMMAND} --build . --target ${MADNESS_WORLD_LIBRARY}
      WORKING_DIRECTORY ${MADNESS_BINARY_DIR}
      BYPRODUCTS "${MADNESS_BUILD_BYPRODUCTS}"
      COMMENT Building 'madness')

  # Add clean-madness target that will delete files generated by MADNESS build.
  add_custom_target(clean-madness
    COMMAND ${CMAKE_COMMAND} --build . --target clean
    WORKING_DIRECTORY ${MADNESS_BINARY_DIR}
    COMMENT Cleaning build directory for 'madness')
  
  # Since 'install-madness' target cannot be linked to the 'install' target,
  # we will do it manually here.
  set(INSTALL_MADNESS_SUBTARGETS install-madness-world install-madness-config install-madness-common)
  foreach(INSTALL_MADNESS_SUBTARGET IN LISTS INSTALL_MADNESS_SUBTARGETS)
    install(CODE
      "execute_process(
         COMMAND \"${CMAKE_COMMAND}\" \"--build\" \".\" \"--target\" \"${INSTALL_MADNESS_SUBTARGET}\"
         WORKING_DIRECTORY \"${MADNESS_BINARY_DIR}\"
         RESULT_VARIABLE error_code)
       if(error_code)
         message(FATAL_ERROR \"Failed to install 'madness'\")
       endif()
      "
    )
  endforeach()

  # Set build dependencies and compiler arguments
  add_dependencies(External-tiledarray build-madness)

  # Set config variables
  replace_mad_targets_with_libnames(MADNESS_LIBRARIES MADNESS_CONFIG_LIBRARIES)
  list(APPEND TiledArray_CONFIG_LIBRARIES ${MADNESS_CONFIG_LIBRARIES})

endif()


include_directories(${MADNESS_INCLUDE_DIRS})
list (APPEND TiledArray_LIBRARIES ${MADNESS_LIBRARIES})
append_flags(CMAKE_CXX_FLAGS "${MADNESS_COMPILE_FLAGS}")
append_flags(CMAKE_EXE_LINKER_FLAGS "${MADNESS_LINKER_FLAGS}")
