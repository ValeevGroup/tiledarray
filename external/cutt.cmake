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

    # to pass CMAKE_C_* vars to external project
    enable_language(C)

    # set source and build path for cuTT in the TiledArray project
    set(EXTERNAL_SOURCE_DIR   ${PROJECT_BINARY_DIR}/external/source/cutt)
    # cutt only supports in source build
    set(EXTERNAL_BUILD_DIR  ${PROJECT_BINARY_DIR}/external/build/cutt)
    set(EXTERNAL_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/external/cutt)

    if (NOT CUTT_URL)
        set(CUTT_URL https://github.com/ValeevGroup/cutt.git)
    endif (NOT CUTT_URL)
    if (NOT CUTT_TAG)
        set(CUTT_TAG ${TA_TRACKED_CUTT_TAG})
    endif (NOT CUTT_TAG)

    message("** Will clone cuTT from ${CUTT_URL}")

    # need to change the separator of list to avoid issues with ExternalProject parsing
#    set(CUDA_FLAGS "${CUDA_NVCC_FLAGS}")
#    string(REPLACE ";" "::" CUDA_FLAGS "${CUDA_NVCC_FLAGS}")
    #message(STATUS "CUDA_FLAGS: " "${CUDA_FLAGS}")

    set(CUTT_CMAKE_ARGS
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
        -DCMAKE_CUDA_STANDARD=${CMAKE_CUDA_STANDARD}
        -DCMAKE_CUDA_EXTENSIONS=${CMAKE_CUDA_EXTENSIONS}
        -DENABLE_UMPIRE=OFF
        -DCUTT_USES_THIS_UMPIRE_ALLOCATOR=ThreadSafeUMDynamicPool
        -DCMAKE_PREFIX_PATH=${_UMPIRE_INSTALL_DIR}
        -DENABLE_NO_ALIGNED_ALLOC=ON
        -DCMAKE_CUDA_HOST_COMPILER=${CMAKE_CUDA_HOST_COMPILER}
        -DCUDA_TOOLKIT_ROOT_DIR=${CUDAToolkit_ROOT}
        )
    if (CMAKE_TOOLCHAIN_FILE)
        set(CUTT_CMAKE_ARGS "${CUTT_CMAKE_ARGS}"
            "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}")
    endif(CMAKE_TOOLCHAIN_FILE)

    if (BUILD_SHARED_LIBS)
        set(CUTT_DEFAULT_LIBRARY_SUFFIX ${CMAKE_SHARED_LIBRARY_SUFFIX})
    else(BUILD_SHARED_LIBS)
        set(CUTT_DEFAULT_LIBRARY_SUFFIX ${CMAKE_STATIC_LIBRARY_SUFFIX})
    endif(BUILD_SHARED_LIBS)

    # N.B. Ninja needs spelling out the byproducts of custom targets, see https://cmake.org/cmake/help/v3.3/policy/CMP0058.html
    set(CUTT_BUILD_BYPRODUCTS "${EXTERNAL_BUILD_DIR}/src/libcutt${CUTT_DEFAULT_LIBRARY_SUFFIX}")
    message(STATUS "custom target cutt is expected to build these byproducts: ${CUTT_BUILD_BYPRODUCTS}")

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
            LIST_SEPARATOR ::
            UPDATE_DISCONNECTED 1
            CMAKE_ARGS
            ${CUTT_CMAKE_ARGS}
            	${EXTERNAL_SOURCE_DIR}
            #--Build step-----------------
            BINARY_DIR ${EXTERNAL_BUILD_DIR}
            BUILD_COMMAND ${CMAKE_COMMAND} --build . --target cutt -v
            BUILD_BYPRODUCTS ${CUTT_BUILD_BYPRODUCTS}
            #--Install step---------------
            INSTALL_COMMAND ${CMAKE_COMMAND} -E echo "cuTT will be installed during TiledArray's installation."
            #--Custom targets-------------
            STEP_TARGETS build
            )

    # do install of cuTT as part of building TiledArray's install target
    install(CODE
            "execute_process(
               COMMAND \"${CMAKE_COMMAND}\" \"--build\" \".\" \"--target\" \"install\"
               WORKING_DIRECTORY \"${EXTERNAL_BUILD_DIR}\"
               RESULT_VARIABLE error_code)
               if(error_code)
                 message(FATAL_ERROR \"Failed to install cuTT\")
               endif()
            ")

    # Add cuTT dependency to External
    add_dependencies(External-tiledarray cutt-build)

    set(_CUTT_INSTALL_DIR ${EXTERNAL_INSTALL_DIR})

endif(_CUTT_INSTALL_DIR)

add_library(TiledArray_CUTT INTERFACE)

set_target_properties(TiledArray_CUTT
        PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES
        "$<BUILD_INTERFACE:${EXTERNAL_SOURCE_DIR}/src>;$<INSTALL_INTERFACE:${_CUTT_INSTALL_DIR}/include>"
        INTERFACE_LINK_LIBRARIES
        "$<BUILD_INTERFACE:${CUTT_BUILD_BYPRODUCTS}>;$<INSTALL_INTERFACE:${_CUTT_INSTALL_DIR}/lib/libcutt.${CUTT_DEFAULT_LIBRARY_SUFFIX}>"
        )

install(TARGETS TiledArray_CUTT EXPORT tiledarray COMPONENT tiledarray)


#TODO test cuTT
