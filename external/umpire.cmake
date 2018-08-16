##
## find Umpire
##

# if user provides UMPIRE, use it
if(UMPIRE_INSTALL_DIR)

    ## check umpire
    set(umpire_DIR ${UMPIRE_INSTALL_DIR}/share/umpire/cmake)

    find_package(umpire REQUIRED)

elseif(TA_EXPERT)

    message("** Umpire was not found")
    message(STATUS "** Downloading and building Umpire is explicitly disabled in EXPERT mode")

else()

    # TODO need to fix this
    ## build umpire automatically

    include(ExternalProject)

    # set source and build path for cuTT in the TiledArray project
    set(EXTERNAL_SOURCE_DIR   ${PROJECT_BINARY_DIR}/external/source/Umpire)
    # cutt only supports in source build
    set(EXTERNAL_BUILD_DIR  ${PROJECT_BINARY_DIR}/external/build/Umpire)
    if (NOT UMPIRE_URL)
        set(UMPIRE_URL https://github.com/LLNL/Umpire.git)
    endif (NOT UMPIRE_URL)

    set(UMPIRE_TAG master)

    message("** Will clone Umpire from ${UMPIRE_URL}")

    ExternalProject_Add(Umpire
            PREFIX ${CMAKE_INSTALL_PREFIX}
            STAMP_DIR ${PROJECT_BINARY_DIR}/external/Umpire-stamp
            TMP_DIR ${PROJECT_BINARY_DIR}/external/tmp
            #--Download step--------------
            DOWNLOAD_DIR ${EXTERNAL_SOURCE_DIR}
            GIT_REPOSITORY ${UMPIRE_URL}
            GIT_TAG ${UMPIRE_TAG}
            #--Configure step-------------
            SOURCE_DIR ${EXTERNAL_SOURCE_DIR}
            CONFIGURE_COMMAND "cmake -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} -DENABLE_CUDA=ON
                               -DENABLE_OPENMP=OFF
                               -DENABLE_TESTS=OFF
                               -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR}
                                ${EXTERNAL_SOURCE_DIR}"
            #--Build step-----------------
            BINARY_DIR ${EXTERNAL_BUILD_DIR}
            BUILD_COMMAND "make"
            #--Install step---------------
            INSTALL_COMMAND "make install"
            #--Custom targets-------------
            STEP_TARGETS download
            )

    # Add Umpire dependency to External
    add_dependencies(External Umpire)

    # create an exportable interface target for Umpire
    add_library(TiledArray_UMPIRE INTERFACE)

    set_property(TARGET TiledArray_UMPIRE PROPERTY
            INTERFACE_INCLUDE_DIRECTORIES
            $<BUILD_INTERFACE:${EXTERNAL_BUILD_DIR}>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}>
            )

#    install(TARGETS TiledArray_UMPIRE EXPORT tiledarray COMPONENT tiledarray)

endif(UMPIRE_INSTALL_DIR)