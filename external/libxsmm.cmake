##
## Fetch + build libxsmm from source, and expose it as the TiledArray_LIBXSMM
## INTERFACE target. Enabled by -DTA_LIBXSMM=ON.
##
## libxsmm provides a JIT small-GEMM fast path for the strided tensor-of-tensors
## micro-GEMMs (ce+e, ce+ce, scale). Unlike most TA deps, libxsmm's canonical
## build is a GNU Makefile (not CMake), so this uses ExternalProject_Add with a
## custom `make ... install` build command rather than CMAKE_ARGS.
##
## No system install is assumed: if libxsmm is not found via LIBXSMM_INSTALL_DIR
## (an optional hint), it is cloned and built from source under
## ${FETCHCONTENT_BASE_DIR}. There is intentionally NO TA_LIBXSMM_ROOT knob.
##

# Optional: reuse a pre-built libxsmm ONLY if the user explicitly points at one
# via -DLIBXSMM_INSTALL_DIR=... We deliberately do NOT search default system
# paths (no NO_DEFAULT_PATH omission): a stray /usr/local install must never be
# picked up silently. Absent an explicit hint, libxsmm is always fetched+built.
# clear any stale value cached by a prior configure (e.g. before this guard)
unset(_LIBXSMM_INSTALL_DIR CACHE)
set(_LIBXSMM_PREBUILT _LIBXSMM_PREBUILT-NOTFOUND)
if (DEFINED LIBXSMM_INSTALL_DIR)
  find_path(_LIBXSMM_PREBUILT NAMES include/libxsmm.h lib/libxsmm.a
            HINTS ${LIBXSMM_INSTALL_DIR} NO_DEFAULT_PATH)
endif ()

if (_LIBXSMM_PREBUILT)

    set(_LIBXSMM_INSTALL_DIR ${_LIBXSMM_PREBUILT})
    message(STATUS "libxsmm found at ${_LIBXSMM_INSTALL_DIR}")

elseif (TA_EXPERT)

    message("** libxsmm was not found")
    message(STATUS "** Downloading and building libxsmm is explicitly disabled in EXPERT mode")
    message(FATAL_ERROR "** Either provide a pre-built libxsmm via -DLIBXSMM_INSTALL_DIR=... or disable -DTA_LIBXSMM=OFF")

else ()

    include(ExternalProject)

    # libxsmm is a C library; make sure CMAKE_C_COMPILER is configured
    enable_language(C)

    set(EXTERNAL_SOURCE_DIR  ${FETCHCONTENT_BASE_DIR}/libxsmm-src)
    set(_LIBXSMM_INSTALL_DIR ${FETCHCONTENT_BASE_DIR}/libxsmm-install)

    if (NOT LIBXSMM_URL)
        set(LIBXSMM_URL https://github.com/libxsmm/libxsmm.git)
    endif (NOT LIBXSMM_URL)
    if (NOT LIBXSMM_TAG)
        set(LIBXSMM_TAG ${TA_TRACKED_LIBXSMM_TAG})
    endif (NOT LIBXSMM_TAG)

    message("** Will clone libxsmm from ${LIBXSMM_URL}")

    # Compiler for libxsmm's sub-make. libxsmm builds with -target
    # <arch>-apple-macos, which makes a bare CommandLineTools clang stop
    # auto-injecting the macOS SDK sysroot, so it cannot find system headers
    # (pthread.h) or libSystem at link time (and CMAKE_OSX_SYSROOT is often
    # empty). On Apple, build libxsmm with the /usr/bin/{cc,c++} xcrun shims,
    # which always resolve the active SDK for both compile and link; the
    # resulting libxsmm.a is C-ABI-compatible with the rest of TiledArray.
    # Elsewhere, honor the project's configured compilers.
    if (APPLE)
        set(_libxsmm_cc /usr/bin/cc)
        set(_libxsmm_cxx /usr/bin/c++)
    else ()
        set(_libxsmm_cc ${CMAKE_C_COMPILER})
        set(_libxsmm_cxx ${CMAKE_CXX_COMPILER})
    endif ()

    # parallelism for the libxsmm sub-make (overridable; default mirrors the
    # project's typical build budget rather than hardcoding into the command)
    set(LIBXSMM_BUILD_NJOBS 6 CACHE STRING "Parallel jobs for the libxsmm sub-make")

    # libxsmm Make knobs:
    #   STATIC=1   build libxsmm.a (we link the archive into tiledarray)
    #   FORTRAN=0  skip the Fortran interface (no gfortran needed)
    #   BLAS=0     do not wrap an external BLAS (we only use the JIT SMM path,
    #              and TA already links its own BLAS); avoids a second BLAS dep
    #   PREFIX=... install headers+lib into our private prefix
    set(LIBXSMM_BUILD_BYPRODUCTS "${_LIBXSMM_INSTALL_DIR}/lib/libxsmm.a")
    message(STATUS "custom target libxsmm is expected to build these byproducts: ${LIBXSMM_BUILD_BYPRODUCTS}")

    ExternalProject_Add(libxsmm
            PREFIX ${FETCHCONTENT_BASE_DIR}
            STAMP_DIR ${FETCHCONTENT_BASE_DIR}/libxsmm-ep-artifacts
            TMP_DIR ${FETCHCONTENT_BASE_DIR}/libxsmm-ep-artifacts  # in case CMAKE_INSTALL_PREFIX is not writable
            #--Download step--------------
            DOWNLOAD_DIR ${EXTERNAL_SOURCE_DIR}
            GIT_REPOSITORY ${LIBXSMM_URL}
            GIT_TAG ${LIBXSMM_TAG}
            #--Configure step------------- (none: libxsmm uses a plain Makefile)
            SOURCE_DIR ${EXTERNAL_SOURCE_DIR}
            UPDATE_DISCONNECTED 1
            BUILD_IN_SOURCE 1
            CONFIGURE_COMMAND ""
            #--Build step----------------- build + install in one make invocation
            BUILD_COMMAND make -j${LIBXSMM_BUILD_NJOBS} STATIC=1 FORTRAN=0 BLAS=0
                          CC=${_libxsmm_cc} CXX=${_libxsmm_cxx} AR=${CMAKE_AR}
                          PREFIX=${_LIBXSMM_INSTALL_DIR} install
            BUILD_BYPRODUCTS ${LIBXSMM_BUILD_BYPRODUCTS}
            #--Install step--------------- (done by BUILD_COMMAND above)
            INSTALL_COMMAND ${CMAKE_COMMAND} -E echo "libxsmm installed to ${_LIBXSMM_INSTALL_DIR}"
            #--Custom targets-------------
            STEP_TARGETS build
            )

    # the include dir must exist at configure time so the INTERFACE target's
    # BUILD_INTERFACE include path validates (it is populated at build time)
    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${_LIBXSMM_INSTALL_DIR}/include")

    # build libxsmm before any TiledArray translation unit links
    add_dependencies(External-tiledarray libxsmm)

endif (_LIBXSMM_INSTALL_DIR)

# Fold libxsmm's static lib + headers into TiledArray's OWN install prefix, so
# the exported TiledArray config is self-contained and does not reference TA's
# build tree (which a downstream find_package(TiledArray) consumer like MPQC
# would otherwise link against -- and which breaks once the build tree is
# wiped/relocated). Done for both the fetched and the prebuilt cases so the TA
# install is identical either way.
install(FILES "${_LIBXSMM_INSTALL_DIR}/lib/libxsmm.a"
        DESTINATION "${TILEDARRAY_INSTALL_LIBDIR}" COMPONENT tiledarray)
install(DIRECTORY "${_LIBXSMM_INSTALL_DIR}/include/"
        DESTINATION "${TILEDARRAY_INSTALL_INCLUDEDIR}" COMPONENT tiledarray)

# Synthetic target carrying the include dir, the static archive, and the gating
# define. PUBLIC propagation (via _TILEDARRAY_DEPENDENCIES) makes the libxsmm.a
# link requirement reach consumers (libtiledarray is a static archive, so its
# undefined libxsmm symbols are resolved at the consumer's final link), plus
# TILEDARRAY_HAS_LIBXSMM + the include path. The link/include paths are split
# into BUILD_INTERFACE (TA's build tree) and INSTALL_INTERFACE (the installed
# copy above), so the exported config never references the build tree. The
# include INSTALL_INTERFACE is relative (CMake prepends the import prefix); the
# link library INSTALL_INTERFACE must be an absolute path to the installed
# archive -- CMake does NOT prepend the import prefix to INTERFACE_LINK_LIBRARIES
# entries, so a relative path there would be resolved against the consumer's cwd
# and fail to link. Absolute CMAKE_INSTALL_PREFIX is leak-free (the install tree
# is the stable final location, unlike the build tree).
add_library(TiledArray_LIBXSMM INTERFACE)
set_target_properties(TiledArray_LIBXSMM
        PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES
        "$<BUILD_INTERFACE:${_LIBXSMM_INSTALL_DIR}/include>;$<INSTALL_INTERFACE:${TILEDARRAY_INSTALL_INCLUDEDIR}>"
        INTERFACE_LINK_LIBRARIES
        "$<BUILD_INTERFACE:${_LIBXSMM_INSTALL_DIR}/lib/libxsmm.a>;$<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/${TILEDARRAY_INSTALL_LIBDIR}/libxsmm.a>;${CMAKE_DL_LIBS}"
        INTERFACE_COMPILE_DEFINITIONS
        "TILEDARRAY_HAS_LIBXSMM"
        )

install(TARGETS TiledArray_LIBXSMM EXPORT tiledarray COMPONENT tiledarray)
