# Try find_package
if (NOT TARGET slate)
    find_package(slate QUIET CONFIG)
    if (TARGET slate)
        message(STATUS "Found SLATE CONFIG at ${slate_CONFIG}")
    endif (TARGET slate)
endif (NOT TARGET slate)

# If not found, build via FetchContent
if (NOT TARGET slate)

    # Make sure BLAS++/LAPACK++ are already in place
    # (will typically be loaded from BTAS)
    include(${vg_cmake_kit_SOURCE_DIR}/modules/FindOrFetchLinalgPP.cmake)
    
    if (NOT TILEDARRAY_HAS_CUDA)
        set(gpu_backend none CACHE STRING "Device Backend for ICL Linalg++/SLATE")
    endif (NOT TILEDARRAY_HAS_CUDA)

    include(FetchContent)
    FetchContent_Declare(
        slate
        GIT_REPOSITORY  https://github.com/icl-utk-edu/slate.git
        GIT_TAG         ${TA_TRACKED_SLATE_TAG}
    )
    FetchContent_MakeAvailable(slate)

endif (NOT TARGET slate)

if (NOT TARGET slate)
    message( FATAL_ERROR "FindOrFetchSLATE could not make slate target available")
endif (NOT TARGET slate)
