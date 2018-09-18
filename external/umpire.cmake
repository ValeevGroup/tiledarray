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
        set(UMPIRE_URL https://github.com/pchong90/Umpire.git)
    endif (NOT UMPIRE_URL)

    set(UMPIRE_TAG develop)

    message("** Will clone Umpire from ${UMPIRE_URL}")

    # find HOST compiler
    if(${CUDA_HOST_COMPILER})
        set(UMPIRE_CUDA_HOST_COMPILER ${CUDA_HOST_COMPILER})
    else()
        # find HOST compiler if set in NVCC_FLAGS
        list(FIND CUDA_NVCC_FLAGS -ccbin CCBIN_INDEX)
        if(NOT ${CCBIN_INDEX} EQUAL -1)
            #message(STATUS ${CCBIN_INDEX})
            math(EXPR CCBIN_INDEX "${CCBIN_INDEX}+1")
            #message(STATUS ${CCBIN_INDEX})
            list(GET CUDA_NVCC_FLAGS ${CCBIN_INDEX} UMPIRE_CUDA_HOST_COMPILER)
        endif()
    endif()
    message(STATUS "UMPIRE CUDA_HOST_COMPILER: " "${UMPIRE_CUDA_HOST_COMPILER}")

    set(CUDA_FLAGS "${CUDA_NVCC_FLAGS}")
    string(REPLACE ";" "::" CUDA_FLAGS "${CUDA_NVCC_FLAGS}")
    message(STATUS "UMPIRE CUDA_NVCC_FLAGS: " "${CUDA_FLAGS}")

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
                -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_DIR}
                -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                -DENABLE_CUDA=ON
                -DENABLE_OPENMP=OFF
                -DENABLE_TESTS=OFF
                -DENABLE_ASSERTS=${TA_DEFAULT_ERROR}
                -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR}
                -DCUDA_NVCC_FLAGS=${CUDA_FLAGS}
                -DCMAKE_CUDA_HOST_COMPILER=${UMPIRE_CUDA_HOST_COMPILER}/g++
                ${EXTERNAL_SOURCE_DIR}
            #--Build step-----------------
            BINARY_DIR ${EXTERNAL_BUILD_DIR}
            BUILD_COMMAND make
            #--Install step---------------
            INSTALL_COMMAND make install
            #--Custom targets-------------
            STEP_TARGETS download
            )

    # Add Umpire dependency to External
    add_dependencies(External Umpire)

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
        ${_UMPIRE_INSTALL_DIR}/lib/libumpire.a ${_UMPIRE_INSTALL_DIR}/lib/libumpire_op.a ${_UMPIRE_INSTALL_DIR}/lib/libumpire_resource.a ${_UMPIRE_INSTALL_DIR}/lib/libumpire_strategy.a ${_UMPIRE_INSTALL_DIR}/lib/libumpire_util.a ${_UMPIRE_INSTALL_DIR}/lib/libumpire_tpl_judy.a
        )

install(TARGETS TiledArray_UMPIRE EXPORT tiledarray COMPONENT tiledarray)

#TODO test Umpire
