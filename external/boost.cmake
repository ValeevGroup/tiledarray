# -*- mode: cmake -*-

# Limit scope of the search if BOOST_ROOT or BOOST_INCLUDEDIR is provided.
if (BOOST_ROOT OR BOOST_INCLUDEDIR)
  set(Boost_NO_SYSTEM_PATHS TRUE)
endif()
  
# Check for Boost
# - require version 1.68 where Boost.Test preconditions work correctly (see https://svn.boost.org/trac10/ticket/12095)
find_package(Boost ${TA_TRACKED_BOOST_VERSION})

if (Boost_FOUND)

  # Perform a compile check with Boost
  list(APPEND CMAKE_REQUIRED_INCLUDES ${Boost_INCLUDE_DIRS})

  CHECK_CXX_SOURCE_COMPILES(
      "
      #define BOOST_TEST_MAIN main_tester
      #include <boost/test/included/unit_test.hpp>

      BOOST_AUTO_TEST_CASE( tester )
      {
        BOOST_CHECK( true );
      }
      "  BOOST_COMPILES)

  if (NOT BOOST_COMPILES)
    message(FATAL_ERROR "Boost found at ${BOOST_ROOT}, but could not compile test program")
  endif()
  
elseif(TA_EXPERT)

  message("** BOOST was not explicitly set")
  message(FATAL_ERROR "** Downloading and building Boost is explicitly disabled in EXPERT mode")

else()

  include(ExternalProject)
  
  # Set source and build path for Boost in the TiledArray Project
  set(BOOST_DOWNLOAD_DIR ${PROJECT_BINARY_DIR}/external/source)
  set(BOOST_SOURCE_DIR   ${PROJECT_BINARY_DIR}/external/source/boost)
  set(BOOST_BUILD_DIR   ${PROJECT_BINARY_DIR}/external/build/boost)
  # where to get boost
  set(BOOST_URL
          https://dl.bintray.com/boostorg/release/1.70.0/source/boost_1_70_0.tar.gz)
  set(BOOST_URL_HASH SHA256=882b48708d211a5f48e60b0124cf5863c1534cd544ecd0664bb534a4b5d506e9)

  message("** Will build Boost from ${BOOST_URL}")

  ExternalProject_Add(boost
    PREFIX ${CMAKE_INSTALL_PREFIX}
    STAMP_DIR ${BOOST_BUILD_DIR}/stamp
    TMP_DIR ${BOOST_BUILD_DIR}/tmp
   #--Download step--------------
    URL ${BOOST_URL}
    URL_HASH ${BOOST_URL_HASH}
    DOWNLOAD_DIR ${BOOST_DOWNLOAD_DIR}
   #--Configure step-------------
    SOURCE_DIR ${BOOST_SOURCE_DIR}
    CONFIGURE_COMMAND ""
   #--Build step-----------------
    BUILD_COMMAND ""
   #--Install step---------------
    INSTALL_COMMAND ""
   #--Custom targets-------------
    STEP_TARGETS download
    )

  add_dependencies(External-tiledarray boost)
  install(
    DIRECTORY ${BOOST_SOURCE_DIR}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT boost
    )
  set(Boost_INCLUDE_DIRS ${BOOST_SOURCE_DIR})

endif()

# Set the  build variables
include_directories(${Boost_INCLUDE_DIRS})
