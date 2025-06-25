##
## find LibreTT
##

find_path(_LIBRETT_INSTALL_DIR NAMES include/librett.h lib/librett.a HINTS ${LIBRETT_INSTALL_DIR})

if( _LIBRETT_INSTALL_DIR )

    message(STATUS "LibreTT found at ${_LIBRETT_INSTALL_DIR}")

elseif(TA_EXPERT)

    message("** LibreTT was not found")
    message(STATUS "** Downloading and building LibreTT is explicitly disabled in EXPERT mode")

else()

    # TODO need to fix the auto installation of LibreTT

    include(ExternalProject)

    # to pass CMAKE_C_* vars to external project
    enable_language(C)

    # set source and build path for LibreTT in the TiledArray project
    set(EXTERNAL_SOURCE_DIR   ${FETCHCONTENT_BASE_DIR}/librett-src)
    # librett only supports in source build
    set(EXTERNAL_BUILD_DIR  ${FETCHCONTENT_BASE_DIR}/librett-build)
    set(EXTERNAL_INSTALL_DIR ${CMAKE_INSTALL_PREFIX})

    if (NOT LIBRETT_URL)
        set(LIBRETT_URL https://github.com/victor-anisimov/librett.git)
    endif (NOT LIBRETT_URL)
    if (NOT LIBRETT_TAG)
        set(LIBRETT_TAG ${TA_TRACKED_LIBRETT_TAG})
    endif (NOT LIBRETT_TAG)

    if (CMAKE_PREFIX_PATH)
        set(LIBRETT_CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH})
    endif()
    list(APPEND LIBRETT_CMAKE_PREFIX_PATH ${_UMPIRE_INSTALL_DIR})

    message("** Will clone LibreTT from ${LIBRETT_URL}")

    set(LIBRETT_CMAKE_ARGS
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
        -DENABLE_UMPIRE=OFF
        # N.B. ThreadSafeUMDynamicPool this no longer exists!!! Must teach LibreTT to take allocate/deallocate methods
        # from the user code
        -DLIBRETT_USES_THIS_UMPIRE_ALLOCATOR=ThreadSafeUMDynamicPool
        -DCMAKE_PREFIX_PATH=${LIBRETT_CMAKE_PREFIX_PATH}
        -DENABLE_NO_ALIGNED_ALLOC=ON
        )
    if (TA_CUDA)
        list(APPEND LIBRETT_CMAKE_ARGS
                -DENABLE_CUDA=ON
                -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
                -DCMAKE_CUDA_STANDARD=${CMAKE_CUDA_STANDARD}
                -DCMAKE_CUDA_EXTENSIONS=${CMAKE_CUDA_EXTENSIONS}
                -DCMAKE_CUDA_HOST_COMPILER=${CMAKE_CUDA_HOST_COMPILER}
                -DCUDA_TOOLKIT_ROOT_DIR=${CUDAToolkit_ROOT}
        )
        if (DEFINED CMAKE_CUDA_ARCHITECTURES)
            list(APPEND LIBRETT_CMAKE_ARGS "-DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
        endif(DEFINED CMAKE_CUDA_ARCHITECTURES)
    endif()
    if (TA_HIP)
        list(APPEND LIBRETT_CMAKE_ARGS
                -DENABLE_HIP=ON
                -DCMAKE_HIP_COMPILER=${CMAKE_HIP_COMPILER}
                -DCMAKE_HIP_STANDARD=${CMAKE_HIP_STANDARD}
                -DCMAKE_HIP_EXTENSIONS=${CMAKE_HIP_EXTENSIONS}
        )
        if (DEFINED CMAKE_HIP_ARCHITECTURES)
            list(APPEND LIBRETT_CMAKE_ARGS "-DCMAKE_HIP_ARCHITECTURES=${CMAKE_HIP_ARCHITECTURES}")
        endif(DEFINED CMAKE_HIP_ARCHITECTURES)
    endif()
    if (CMAKE_TOOLCHAIN_FILE)
        set(LIBRETT_CMAKE_ARGS "${LIBRETT_CMAKE_ARGS}"
            "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}")
    endif(CMAKE_TOOLCHAIN_FILE)
    if (DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION)
        set(LIBRETT_CMAKE_ARGS "${LIBRETT_CMAKE_ARGS}"
                "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=${CMAKE_INTERPROCEDURAL_OPTIMIZATION}")
    endif(DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION)
    if (DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION_${CMAKE_BUILD_TYPE})
        set(LIBRETT_CMAKE_ARGS "${LIBRETT_CMAKE_ARGS}"
                "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION_${CMAKE_BUILD_TYPE}=${CMAKE_INTERPROCEDURAL_OPTIMIZATION_${CMAKE_BUILD_TYPE}}")
    endif(DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION_${CMAKE_BUILD_TYPE})

    foreach(lang C CXX CUDA)
        if (DEFINED CMAKE_${lang}_COMPILER_LAUNCHER)
            list(APPEND LIBRETT_CMAKE_ARGS
                    "-DCMAKE_${lang}_COMPILER_LAUNCHER=${CMAKE_${lang}_COMPILER_LAUNCHER}")
        endif()
    endforeach()

    if (BUILD_SHARED_LIBS)
        set(LIBRETT_DEFAULT_LIBRARY_SUFFIX ${CMAKE_SHARED_LIBRARY_SUFFIX})
    else(BUILD_SHARED_LIBS)
        set(LIBRETT_DEFAULT_LIBRARY_SUFFIX ${CMAKE_STATIC_LIBRARY_SUFFIX})
    endif(BUILD_SHARED_LIBS)

    # N.B. Ninja needs spelling out the byproducts of custom targets, see https://cmake.org/cmake/help/v3.3/policy/CMP0058.html
    set(LIBRETT_BUILD_BYPRODUCTS "${EXTERNAL_BUILD_DIR}/src/librett${LIBRETT_DEFAULT_LIBRARY_SUFFIX}")
    message(STATUS "custom target librett is expected to build these byproducts: ${LIBRETT_BUILD_BYPRODUCTS}")

    ExternalProject_Add(librett
            PREFIX ${FETCHCONTENT_BASE_DIR}
            STAMP_DIR ${FETCHCONTENT_BASE_DIR}/librett-ep-artifacts
            TMP_DIR ${FETCHCONTENT_BASE_DIR}/librett-ep-artifacts  # needed in case CMAKE_INSTALL_PREFIX is not writable
            #--Download step--------------
            DOWNLOAD_DIR ${EXTERNAL_SOURCE_DIR}
            GIT_REPOSITORY ${LIBRETT_URL}
            GIT_TAG ${LIBRETT_TAG}
            #--Configure step-------------
            SOURCE_DIR ${EXTERNAL_SOURCE_DIR}
            LIST_SEPARATOR ::
            UPDATE_DISCONNECTED 1
            CMAKE_ARGS
            ${LIBRETT_CMAKE_ARGS}
            	${EXTERNAL_SOURCE_DIR}
            #--Build step-----------------
            BINARY_DIR ${EXTERNAL_BUILD_DIR}
            BUILD_COMMAND ${CMAKE_COMMAND} --build . --target librett -v
            BUILD_BYPRODUCTS ${LIBRETT_BUILD_BYPRODUCTS}
            #--Install step---------------
            INSTALL_COMMAND ${CMAKE_COMMAND} -E echo "LibreTT will be installed during TiledArray's installation."
            #--Custom targets-------------
            STEP_TARGETS build
            )

    # TiledArray_LIBRETT target depends on existence of this directory to be usable from the build tree at configure time
    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${EXTERNAL_SOURCE_DIR}/src")

    # do install of LibreTT as part of building TiledArray's install target
    install(CODE
            "execute_process(
               COMMAND \"${CMAKE_COMMAND}\" \"--build\" \".\" \"--target\" \"install\"
               WORKING_DIRECTORY \"${EXTERNAL_BUILD_DIR}\"
               RESULT_VARIABLE error_code)
               if(error_code)
                 message(FATAL_ERROR \"Failed to install LibreTT\")
               endif()
            ")

    # Add LibreTT dependency to External
    add_dependencies(External-tiledarray librett)

    set(_LIBRETT_INSTALL_DIR ${EXTERNAL_INSTALL_DIR})

endif(_LIBRETT_INSTALL_DIR)

add_library(TiledArray_LIBRETT INTERFACE)

set_target_properties(TiledArray_LIBRETT
        PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES
        "$<BUILD_INTERFACE:${EXTERNAL_SOURCE_DIR}/src>;$<INSTALL_INTERFACE:${_LIBRETT_INSTALL_DIR}/include>"
        INTERFACE_LINK_LIBRARIES
        "$<BUILD_INTERFACE:${LIBRETT_BUILD_BYPRODUCTS}>;$<INSTALL_INTERFACE:${_LIBRETT_INSTALL_DIR}/lib/librett.${LIBRETT_DEFAULT_LIBRARY_SUFFIX}>"
        INTERFACE_LINK_OPTIONS
        "LINKER:-rpath,${EXTERNAL_BUILD_DIR}/src"
        )
if (TA_CUDA)
    set_target_properties(TiledArray_LIBRETT
            PROPERTIES
            INTERFACE_COMPILE_DEFINITIONS
            "LIBRETT_USES_CUDA=1"
    )
endif()
if (TA_HIP)
    set_target_properties(TiledArray_LIBRETT
            PROPERTIES
            INTERFACE_COMPILE_DEFINITIONS
            "LIBRETT_USES_HIP=1"
    )
endif()

install(TARGETS TiledArray_LIBRETT EXPORT tiledarray COMPONENT tiledarray)


#TODO test LibreTT
