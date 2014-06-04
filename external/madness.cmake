# -*- mode: cmake -*-

###################
# Find MADNESS
###################

include(ExternalProject)
include(ConvertIncludesListToCompilerArgs)
include(ConvertLibrariesListToCompilerArgs)

find_package(Madness COMPONENTS MADworld)

if(Madness_FOUND)

  # sanity check: try compiling a simple program
  list(APPEND CMAKE_REQUIRED_INCLUDES ${Madness_INCLUDE_DIRS})
  list(APPEND CMAKE_REQUIRED_LIBRARIES ${Madness_LIBRARIES})
  CHECK_CXX_SOURCE_COMPILES(
    "
    #include <world/world.h>
    int main(int argc, char** argv) {
      madness::World& world = madness::initialize(argc, argv);
      return 0;
    }
    "  Madness_COMPILES)
  if (NOT Madness_COMPILES)
    message(FATAL_ERROR "MADNESS found at ${Madness_INCLUDE_DIR}, but could not compile test program")
  endif()

  CHECK_CXX_SOURCE_COMPILES(
    "
    #include <madness_config.h>
    #ifndef MADNESS_HAS_ELEMENTAL
    # error \"MADNESS does not have Elemental\"
    #endif
    int main(int argc, char** argv) {
      return 0;
    }
    "  MADNESS_HAS_ELEMENTAL)

elseif(TA_EXPERT)

  message("** MADNESS was not found or explicitly set")
  message(FATAL_ERROR "** Downloading and building MADNESS is explicitly disabled in EXPERT mode")

else()

  # Set paths for MADNESS project
  set(EXTERNAL_SOURCE_DIR ${PROJECT_SOURCE_DIR}/external/src/madness)
  set(EXTERNAL_BUILD_DIR  ${PROJECT_BINARY_DIR}/external/build/madness)
  set(MADNESS_URL https://github.com/m-a-d-n-e-s-s/madness.git)

  message(STATUS "MADNESS not found")
  message("** Will pull MADNESS from ${MADNESS_URL}")  
  
    
  # If user provided elemental add it to MAD_INCLUDE_DIRS
  if(Elemental_DIR)
    list(APPEND MAD_INCLUDE_DIRS "${Elemental_DIR}/include")
  endif()

  # Convert Madness_INCLUDE_DIRS to compiler arguments for MADNESS configure script
  convert_incs_to_compargs(MAD_CPPFLAGS "${Madness_INCLUDE_DIRS}")
  append_flags(MAD_CPPFLAGS "${CMAKE_CPP_FLAGS}")
  
  convert_libs_to_compargs(MAD_LIBS "${Madness_LIBRARIES}")
  
  # Determine the proper configuration flags for MADNESS
  
  # Set Stub MPI option
  if(MPI_FOUND)
    set(MAD_STUB_MPI "no")
  else()
    set(MAD_STUB_MPI "yes")
  endif()
  
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
  
  # Set configuration flags
  set(MAD_CFLAGS "${CMAKE_C_FLAGS}")
  set(MAD_CXXFLAGS "${CMAKE_CXX_FLAGS}")
  append_flags(MAD_CFLAGS "${Madness_COMPILE_FLAGS}")
  append_flags(MAD_CXXFLAGS "${Madness_COMPILE_FLAGS}")
  
  if(CMAKE_BUILD_TYPE STREQUAL Debug)
    append_flags(MAD_CFLAGS "${CMAKE_C_FLAGS_DEBUG}")
    append_flags(MAD_CXXFLAGS "${CMAKE_CXX_FLAGS_DEBUG}")
  elseif(CMAKE_BUILD_TYPE STREQUAL Release)
    append_flags(MAD_CFLAGS "${CMAKE_C_FLAGS_RELEASE}")
    append_flags(MAD_CXXFLAGS "${CMAKE_CXX_FLAGS_RELEASE}")
  elseif(CMAKE_BUILD_TYPE STREQUAL RelWithDebInfo)
    append_flags(MAD_CFLAGS "${CMAKE_C_FLAGS_RELWITHDEBINFO}")
    append_flags(MAD_CFLAGS "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
  elseif(CMAKE_BUILD_TYPE STREQUAL MinSizeRel)
    append_flags(MAD_CFLAGS "${CMAKE_C_FLAGS_MINSIZEREL}")
    append_flags(MAD_CXXFLAGS "${CMAKE_CXX_FLAGS_MINSIZEREL}")
  endif()
  
  set(MAD_LDFLAGS "${CMAKE_EXE_LINKER_FLAGS}")
  append_flags(MAD_LDFLAGS "${STATIC_LIBRARY_FLAGS}")
  append_flags(MAD_LDFLAGS "${MADNESS_LINK_FLAGS}")
  
  # Check for elemental
  if(Elemental_DIR)
    set(MAD_ELEMENTAL_FLAG "${Elemental_DIR}")
  else()
    set(MAD_ELEMENTAL_FLAG "no")
  endif()
  
  ExternalProject_Add(madness
    PREFIX ${CMAKE_INSTALL_PREFIX}
    STAMP_DIR ${EXTERNAL_BUILD_DIR}/stamp
   #--Download step--------------
    GIT_REPOSITORY ${MADNESS_URL}
   #--Update/Patch step----------
    UPDATE_COMMAND ""
    PATCH_COMMAND /bin/sh ${EXTERNAL_SOURCE_DIR}/autogen.sh
   #--Configure step-------------
    SOURCE_DIR ${EXTERNAL_SOURCE_DIR}
    CONFIGURE_COMMAND ${EXTERNAL_SOURCE_DIR}/configure
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
      "MPICXX=${CMAKE_CXX_COMPILER}"
      "MPICC=${CMAKE_C_COMPILER}"
      "CPPFLAGS=${MAD_CPPFLAGS}"
      "CC=${CMAKE_C_COMPILER}" "CFLAGS=${MAD_CFLAGS}"
      "CXX=${CMAKE_CXX_COMPILER}" "CXXFLAGS=${MAD_CXXFLAGS}"
      "F77=${CMAKE_Fortran_COMPILER}" "FFLAGS=${CMAKE_Fortran_FLAGS}"
      "LDFLAGS=${MAD_LDFLAGS}"
      "LIBS=${MAD_LIBS}"
    CMAKE_GENERATOR "Unix Makefiles"
   #--Build step-----------------
    BINARY_DIR ${EXTERNAL_BUILD_DIR}
    BUILD_COMMAND $(MAKE) libraries V=0
   #--Install step---------------
    INSTALL_COMMAND ""
    STEP_TARGETS download patch configure build
    )

  # Add madness-update target that will pull updates to the madness source from
  # the svn repository. This is done outside ExternalProject_add to prevent
  # madness from doing a full pull, configure, and build everytime the project
  # is built.
  add_custom_target(madness-update
    COMMAND ${GIT_EXECUTABLE} pull --rebase origin master
    COMMAND ${CMAKE_COMMAND} -E touch_nocreate ${EXTERNAL_BUILD_DIR}/stamp/madness-configure
    WORKING_DIRECTORY ${EXTERNAL_SOURCE_DIR}
    COMMENT "Updating source for 'madness' from ${MADNESS_URL}")

  # Add madness-clean target that will delete files generated by MADNESS build.
  add_custom_target(madness-clean
    COMMAND $(MAKE) clean
    COMMAND ${CMAKE_COMMAND} -E touch_nocreate ${EXTERNAL_BUILD_DIR}/stamp/madness-configure
    WORKING_DIRECTORY ${EXTERNAL_BUILD_DIR}
    COMMENT Cleaning build directory for 'madness')
  
  # Since 'madness-install' target cannot be linked to the 'install' target,
  # we will do it manually here. Hopefully this does not break in the future.
  install(
    DIRECTORY
        ${EXTERNAL_SOURCE_DIR}/include/
        ${EXTERNAL_BUILD_DIR}/include/
        ${EXTERNAL_SOURCE_DIR}/src/lib/
        ${EXTERNAL_BUILD_DIR}/src/lib/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT madness
    FILES_MATCHING PATTERN "*.h"
    PATTERN ".deps" EXCLUDE
    )

  install(
    DIRECTORY
        ${EXTERNAL_BUILD_DIR}/src/lib/misc/
        ${EXTERNAL_BUILD_DIR}/src/lib/muParser/
        ${EXTERNAL_BUILD_DIR}/src/lib/tensor/
        ${EXTERNAL_BUILD_DIR}/src/lib/tinyxml/
        ${EXTERNAL_BUILD_DIR}/src/lib/world/
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT madness
    FILES_MATCHING PATTERN "*MADlinalg*"
    PATTERN "*MADmisc*"
    PATTERN "*MADmuparser*"
    PATTERN "*MADtensor*"
    PATTERN "*MADtinyxml*"
    PATTERN "*MADworld*"
    PATTERN ".deps" EXCLUDE
    )

  install(
    DIRECTORY
        ${EXTERNAL_BUILD_DIR}/config/
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
    COMPONENT madness
    FILES_MATCHING PATTERN "MADNESS.pc"
    PATTERN ".deps" EXCLUDE
    )

  # Set build dependencies and compiler arguments
  add_dependencies(External madness)
  
  # MadnessFind will set Madness_INCLUDE_DIRS and Madness_LIBRARIES with the
  # dependencies. So all that needs to be done here is set Madness_INCLUDE_DIR,
  # Madness_INCLUDE_DIRS, Madness_LIBRARY, and Madness_LIBRARIES with the paths
  # and libraries for the built version of MADNESS above.
  set(Madness_INCLUDE_DIR 
      ${EXTERNAL_BUILD_DIR}/include
      ${EXTERNAL_SOURCE_DIR}/include
      ${EXTERNAL_BUILD_DIR}/src/lib
      ${EXTERNAL_SOURCE_DIR}/src/lib)
  set(Madness_INCLUDE_DIRS ${Madness_INCLUDE_DIR} ${Madness_INCLUDE_DIRS})
  set(Madness_LIBRARY ${EXTERNAL_BUILD_DIR}/src/lib/world/libMADworld.a)
  set(Madness_LIBRARIES ${Madness_LIBRARY} ${Madness_LIBRARIES})
    
endif()

message(STATUS "\tMADNESS include dir: ${Madness_INCLUDE_DIRS}")
message(STATUS "\tMADNESS compile flags: ${Madness_COMPILE_FLAGS}")
message(STATUS "\tMADNESS libraries: ${Madness_LIBRARIES}")
message(STATUS "\tMADNESS link flags: ${Madness_LINK_FLAGS}")
