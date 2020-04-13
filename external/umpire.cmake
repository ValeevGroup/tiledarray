##
## find Umpire
##

find_path(_UMPIRE_INSTALL_DIR NAMES include/umpire/Umpire.hpp HINTS ${UMPIRE_INSTALL_DIR})

# if user provides UMPIRE, use it
if(_UMPIRE_INSTALL_DIR)

    ## check umpire
#    set(umpire_DIR ${UMPIRE_INSTALL_DIR}/share/umpire/cmake)
#    find_package(umpire REQUIRED)
    message(STATUS "Umpire found at ${_UMPIRE_INSTALL_DIR}")

elseif(TA_EXPERT)

    message("** Umpire was not found")
    message(STATUS "** Downloading and building Umpire is explicitly disabled in EXPERT mode")

else()

    ## build umpire automatically

    include(ExternalProject)

    # set source and build path for Umpire in the TiledArray project
    set(EXTERNAL_SOURCE_DIR   ${PROJECT_BINARY_DIR}/external/source/Umpire)
    set(EXTERNAL_BUILD_DIR  ${PROJECT_BINARY_DIR}/external/build/Umpire)
    set(EXTERNAL_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/external/Umpire)

    if (NOT UMPIRE_URL)
        set(UMPIRE_URL https://github.com/LLNL/Umpire.git)
    endif (NOT UMPIRE_URL)
    if (NOT UMPIRE_TAG)
        set(UMPIRE_TAG ${TA_TRACKED_UMPIRE_TAG})
    endif (NOT UMPIRE_TAG)

    message("** Will clone Umpire from ${UMPIRE_URL}")

    set(UMPIRE_CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_DIR}
        -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
        -DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}
        -DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}
        -DCMAKE_C_FLAGS_RELWITHDEBINFO=${CMAKE_C_FLAGS_RELWITHDEBINFO}
        -DCMAKE_C_FLAGS_MINSIZEREL=${CMAKE_C_FLAGS_MINSIZEREL}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
        -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}
        -DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}
        -DCMAKE_CXX_FLAGS_RELWITHDEBINFO=${CMAKE_CXX_FLAGS_RELWITHDEBINFO}
        -DCMAKE_CXX_FLAGS_MINSIZEREL=${CMAKE_CXX_FLAGS_MINSIZEREL}
        -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
        -DCMAKE_CXX_EXTENSIONS=${CMAKE_CXX_EXTENSIONS}
        -DBLT_CXX_STD=c++${CMAKE_CUDA_STANDARD}
        -DENABLE_CUDA=ON
        -DENABLE_BENCHMARKS=OFF
        -DENABLE_OPENMP=OFF
        -DENABLE_TESTS=OFF
        -DENABLE_EXAMPLES=OFF
        -DENABLE_LOGGING=OFF
        -DENABLE_ASSERTS=${TA_DEFAULT_ERROR}
        -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR}
        -DCMAKE_CUDA_HOST_COMPILER=${CMAKE_CUDA_HOST_COMPILER}
        )
    if (CMAKE_TOOLCHAIN_FILE)
        set(UMPIRE_CMAKE_ARGS "${UMPIRE_CMAKE_ARGS}"
            "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
            )
    endif(CMAKE_TOOLCHAIN_FILE)

    if (BUILD_SHARED_LIBS)
        set(UMPIRE_DEFAULT_LIBRARY_SUFFIX ${CMAKE_SHARED_LIBRARY_SUFFIX})
    else(BUILD_SHARED_LIBS)
        set(UMPIRE_DEFAULT_LIBRARY_SUFFIX ${CMAKE_STATIC_LIBRARY_SUFFIX})
    endif(BUILD_SHARED_LIBS)

    # N.B. Ninja needs spelling out the byproducts of custom targets, see https://cmake.org/cmake/help/v3.3/policy/CMP0058.html
    set(UMPIRE_BUILD_BYPRODUCTS "${EXTERNAL_BUILD_DIR}/lib/libumpire${UMPIRE_DEFAULT_LIBRARY_SUFFIX}")
    message(STATUS "custom target Umpire is expected to build these byproducts: ${UMPIRE_BUILD_BYPRODUCTS}")

    ExternalProject_Add(Umpire
            PREFIX ${CMAKE_INSTALL_PREFIX}
            STAMP_DIR ${PROJECT_BINARY_DIR}/external/Umpire-stamp
            TMP_DIR ${PROJECT_BINARY_DIR}/external/tmp
            #--Download step--------------
            DOWNLOAD_DIR ${EXTERNAL_SOURCE_DIR}
            GIT_REPOSITORY ${UMPIRE_URL}
            GIT_TAG ${UMPIRE_TAG}
            #--Update step----------------
            UPDATE_COMMAND git submodule init && git submodule update
            #--Configure step-------------
            SOURCE_DIR ${EXTERNAL_SOURCE_DIR}
            LIST_SEPARATOR ::
            UPDATE_DISCONNECTED 1
            CMAKE_ARGS
                ${UMPIRE_CMAKE_ARGS}
                ${EXTERNAL_SOURCE_DIR}
            #--Build step-----------------
            BINARY_DIR ${EXTERNAL_BUILD_DIR}
            BUILD_COMMAND ${CMAKE_COMMAND} --build . -v
            BUILD_BYPRODUCTS ${UMPIRE_BUILD_BYPRODUCTS}
            #--Install step---------------
            INSTALL_COMMAND ${CMAKE_COMMAND} --build . --target install
            #--Custom targets-------------
            STEP_TARGETS download
            )

    # Add Umpire dependency to External
    add_dependencies(External-tiledarray Umpire)

    set(_UMPIRE_INSTALL_DIR ${EXTERNAL_INSTALL_DIR})

endif(_UMPIRE_INSTALL_DIR)

# manually add Umpire library

add_library(TiledArray_UMPIRE INTERFACE)

set_property(TARGET
        TiledArray_UMPIRE
        PROPERTY
        INTERFACE_INCLUDE_DIRECTORIES
        ${_UMPIRE_INSTALL_DIR}/include
        )

set_property(TARGET TiledArray_UMPIRE
        PROPERTY
        INTERFACE_LINK_LIBRARIES
        ${_UMPIRE_INSTALL_DIR}/lib/libumpire.${UMPIRE_DEFAULT_LIBRARY_SUFFIX}
        )

install(TARGETS TiledArray_UMPIRE EXPORT tiledarray COMPONENT tiledarray)

#TODO test Umpire
