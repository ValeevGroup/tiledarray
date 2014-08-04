# -*- mode: cmake -*-

# Limit scope of the search if BOOST_ROOT or BOOST_INCLUDEDIR is provided.
if (BOOST_ROOT OR BOOST_INCLUDEDIR)
  set(Boost_NO_SYSTEM_PATHS TRUE)
endif()
  
# Check for Boost
find_package(Boost 1.33)

if (Boost_FOUND)

  # Perform a compile check with Boost
  list(APPEND CMAKE_REQUIRED_INCLUDES ${Boost_INCLUDE_DIR})

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
  set(EXTERNAL_DOWNLOAD_DIR ${PROJECT_SOURCE_DIR}/external/src)
  set(EXTERNAL_SOURCE_DIR   ${PROJECT_SOURCE_DIR}/external/src/boost)

  # Set the external source
  if (EXISTS ${PROJECT_SOURCE_DIR}/external/src/boost.tar.gz)
    # Use local file
    set(BOOST_URL ${PROJECT_SOURCE_DIR}/external/src/boost.tar.gz)
    set(BOOST_URL_HASH "")
  else()
    # Downlaod remote file
    set(BOOST_URL
        http://downloads.sourceforge.net/project/boost/boost/1.54.0/boost_1_54_0.tar.gz)
    set(BOOST_URL_HASH MD5=efbfbff5a85a9330951f243d0a46e4b9)
  endif()

  message("** Will build Boost from ${BOOST_URL}")

  ExternalProject_Add(boost
    PREFIX ${CMAKE_INSTALL_PREFIX}
    STAMP_DIR ${EXTERNAL_BUILD_DIR}/stamp
   #--Download step--------------
    URL ${BOOST_URL}
    URL_HASH ${BOOST_URL_HASH}
    DOWNLOAD_DIR ${EXTERNAL_DOWNLOAD_DIR}
   #--Configure step-------------
    SOURCE_DIR ${EXTERNAL_SOURCE_DIR}
    CONFIGURE_COMMAND ""
   #--Build step-----------------
    BUILD_COMMAND ""
   #--Install step---------------
    INSTALL_COMMAND ""
   #--Custom targets-------------
    STEP_TARGETS download
    )

  add_dependencies(External boost)
  install(
    DIRECTORY ${EXTERNAL_SOURCE_DIR}/boost
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT boost
    )
  set(Boost_INCLUDE_DIRS ${EXTERNAL_SOURCE_DIR})

endif()

# Set the  build variables
include_directories(${Boost_INCLUDE_DIRS})
