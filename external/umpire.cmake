##
## find Umpire
##

if (NOT TARGET TiledArray_UMPIRE)

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

    # to pass CMAKE_C_* vars to external project
    enable_language(C)

    # set source and build path for Umpire in the TiledArray project
    set(EXTERNAL_SOURCE_DIR ${FETCHCONTENT_BASE_DIR}/umpire-src)
    set(EXTERNAL_BUILD_DIR ${FETCHCONTENT_BASE_DIR}/umpire-build)
    set(EXTERNAL_INSTALL_DIR ${CMAKE_INSTALL_PREFIX})

    if (NOT UMPIRE_URL)
        set(UMPIRE_URL https://github.com/LLNL/Umpire.git)
    endif (NOT UMPIRE_URL)
    if (NOT UMPIRE_TAG)
        set(UMPIRE_TAG ${TA_TRACKED_UMPIRE_TAG})
    endif (NOT UMPIRE_TAG)

    message("** Will clone Umpire from ${UMPIRE_URL}")

    if (TA_ASSERT_POLICY STREQUAL TA_ASSERT_IGNORE)
        set(enable_umpire_asserts OFF)
    else()
        set(enable_umpire_asserts ON)
    endif()

    set(UMPIRE_CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_DIR}
        -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}
        -DCMAKE_POSITION_INDEPENDENT_CODE=${CMAKE_POSITION_INDEPENDENT_CODE}
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
        -DCMAKE_AR=${CMAKE_AR}
        -DBLT_CXX_STD=c++${CMAKE_CXX_STANDARD}
        -DENABLE_BENCHMARKS=OFF
        -DENABLE_OPENMP=OFF
        -DENABLE_TESTS=OFF
        -DENABLE_EXAMPLES=OFF
        -DENABLE_LOGGING=OFF
        -DENABLE_ASSERTS=${enable_umpire_asserts}
        )

    # caveat: on recent Ubuntu default libstdc++ provides filesystem, but if using older gcc (gcc-8) must link against
    # libstdc++fs: https://bugs.launchpad.net/ubuntu/+source/gcc-8/+bug/1824721 ... skip the use of std::filesystem altogether with pre-9 gcc!!!
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
        # disable by populating cache with compile test result variable
        list(APPEND UMPIRE_CMAKE_ARGS
             -DUMPIRE_ENABLE_FILESYSTEM=OFF)
    endif()

    if (ENABLE_CUDA)
        list(APPEND UMPIRE_CMAKE_ARGS
                -DENABLE_CUDA=ON
                -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
                -DCMAKE_CUDA_STANDARD=${CMAKE_CUDA_STANDARD}
                -DCMAKE_CUDA_EXTENSIONS=${CMAKE_CUDA_EXTENSIONS}
                -DCMAKE_CUDA_HOST_COMPILER=${CMAKE_CUDA_HOST_COMPILER}
                -DCUDA_TOOLKIT_ROOT_DIR=${CUDAToolkit_ROOT}
                )
        if (DEFINED CMAKE_CUDA_ARCHITECTURES)
            list(APPEND UMPIRE_CMAKE_ARGS -DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES})
        endif(DEFINED CMAKE_CUDA_ARCHITECTURES)
    endif(ENABLE_CUDA)
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
            STAMP_DIR ${FETCHCONTENT_BASE_DIR}/umpire-ep-artifacts
            TMP_DIR ${FETCHCONTENT_BASE_DIR}/umpire-ep-artifacts   # needed in case CMAKE_INSTALL_PREFIX is not writable
            #--Download step--------------
            DOWNLOAD_DIR ${EXTERNAL_SOURCE_DIR}
            GIT_REPOSITORY ${UMPIRE_URL}
            GIT_TAG ${UMPIRE_TAG}
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
            INSTALL_COMMAND ${CMAKE_COMMAND} -E echo "Umpire will be installed during TiledArray's installation."
            #--Custom targets-------------
            STEP_TARGETS build
            )

    # TiledArray_UMPIRE target depends on existence of these directories to be usable from the build tree at configure time
    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${EXTERNAL_SOURCE_DIR}/src/umpire/tpl/camp/include")
    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${EXTERNAL_BUILD_DIR}/include")

    # do install of Umpire as part of building TiledArray's install target
    install(CODE
            "execute_process(
               COMMAND \"${CMAKE_COMMAND}\" \"--build\" \".\" \"--target\" \"install\"
               WORKING_DIRECTORY \"${EXTERNAL_BUILD_DIR}\"
               RESULT_VARIABLE error_code)
               if(error_code)
                 message(FATAL_ERROR \"Failed to install Umpire\")
               endif()
            ")

    # Add Umpire dependency to External
    add_dependencies(External-tiledarray Umpire-build)

    set(_UMPIRE_INSTALL_DIR ${EXTERNAL_INSTALL_DIR})

endif(_UMPIRE_INSTALL_DIR)

# manually add Umpire library

add_library(TiledArray_UMPIRE INTERFACE)

set_target_properties(
        TiledArray_UMPIRE
        PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES
        "$<BUILD_INTERFACE:${EXTERNAL_SOURCE_DIR}/src>;$<BUILD_INTERFACE:${EXTERNAL_SOURCE_DIR}/src/umpire/tpl/camp/include>;$<BUILD_INTERFACE:${EXTERNAL_BUILD_DIR}/include>;$<INSTALL_INTERFACE:${_UMPIRE_INSTALL_DIR}/include>"
        INTERFACE_LINK_LIBRARIES
        "$<BUILD_INTERFACE:${UMPIRE_BUILD_BYPRODUCTS}>;$<INSTALL_INTERFACE:${_UMPIRE_INSTALL_DIR}/lib/libumpire${UMPIRE_DEFAULT_LIBRARY_SUFFIX}>"
        )

install(TARGETS TiledArray_UMPIRE EXPORT tiledarray COMPONENT tiledarray)

#TODO test Umpire

endif(NOT TARGET TiledArray_UMPIRE)
