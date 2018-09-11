##
## find cuTT
##

find_path(_CUTT_INSTALL_DIR NAMES include/cutt.h lib/libcutt.a HINTS ${CUTT_INSTALL_DIR})

if( _CUTT_INSTALL_DIR )

    message(STATUS "cuTT found at ${_CUTT_INSTALL_DIR}")

elseif(TA_EXPERT)

    message("** cuTT was not found")
    message(STATUS "** Downloading and building cuTT is explicitly disabled in EXPERT mode")

else()

    # TODO need to fix the auto installation of cuTT

    include(ExternalProject)

    # set source and build path for cuTT in the TiledArray project
    set(EXTERNAL_SOURCE_DIR   ${PROJECT_BINARY_DIR}/external/source/cutt)
    # cutt only supports in source build
    set(EXTERNAL_BUILD_DIR  ${PROJECT_BINARY_DIR}/external/build/cutt)
    set(EXTERNAL_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/external/cutt)

    if (NOT CUTT_URL)
        set(CUTT_URL https://github.com/pchong90/cutt.git)
    endif (NOT CUTT_URL)
    set(CUTT_TAG master)

    message("** Will clone cuTT from ${CUTT_URL}")

    ExternalProject_Add(cutt
            PREFIX ${CMAKE_INSTALL_PREFIX}
            STAMP_DIR ${PROJECT_BINARY_DIR}/external/cutt-stamp
            TMP_DIR ${PROJECT_BINARY_DIR}/external/tmp
            #--Download step--------------
            DOWNLOAD_DIR ${EXTERNAL_SOURCE_DIR}
            GIT_REPOSITORY ${CUTT_URL}
            GIT_TAG ${CUTT_TAG}
            #--Configure step-------------
            SOURCE_DIR ${EXTERNAL_SOURCE_DIR}
            CONFIGURE_COMMAND cmake
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_DIR}
            -DENABLE_NO_ALIGNED_ALLOC=ON
            -DENABLE_UMPIRE=ON
            -DUMPIRE_INSTALL_DIR=${_UMPIRE_INSTALL_DIR}
            -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR}
            ${EXTERNAL_SOURCE_DIR}
            #--Build step-----------------
            BINARY_DIR ${EXTERNAL_BUILD_DIR}
            BUILD_COMMAND make
            #--Install step---------------
            INSTALL_COMMAND make install
            #--Custom targets-------------
            STEP_TARGETS download
            )

    # Add cuTT dependency to External
    add_dependencies(External cutt)
    if(TARGET Umpire)
        add_dependencies(cutt Umpire)
    endif()

    set(_CUTT_INSTALL_DIR ${EXTERNAL_INSTALL_DIR})

endif(_CUTT_INSTALL_DIR)

add_library(TiledArray_CUTT INTERFACE)

set_target_properties(TiledArray_CUTT
        PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES
        ${_CUTT_INSTALL_DIR}/include
        INTERFACE_LINK_LIBRARIES
        ${_CUTT_INSTALL_DIR}/lib/libcutt.a
        )

install(TARGETS TiledArray_CUTT EXPORT tiledarray COMPONENT tiledarray)


#TODO test cuTT
