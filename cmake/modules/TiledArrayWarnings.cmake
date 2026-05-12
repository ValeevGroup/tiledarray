include_guard(GLOBAL)

# TiledArray internal warning-policy target.
#
# Owns an INTERFACE library `tiledarray_internal_warnings` that carries
# the warning flags applied to TiledArray's own translation units. The
# target is linked PRIVATE-ly to the `tiledarray` library and to
# in-tree executables (via add_ta_executable). PRIVATE scope is
# load-bearing: it keeps these flags out of INTERFACE_COMPILE_OPTIONS
# on the installed/exported `tiledarray` target, so downstream
# consumers (MPQC, ...) do not inherit -Werror through
# find_package(tiledarray).
#
# The target is also NOT installed/exported, which keeps it out of the
# package.

add_library(tiledarray_internal_warnings INTERFACE)

if (TA_WERROR)
  if (CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang|IntelLLVM)$")
    target_compile_options(tiledarray_internal_warnings INTERFACE
      $<$<OR:$<COMPILE_LANGUAGE:CXX>,$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:HIP>>:-Werror>
      # NVCC: forward -Werror to the host compiler, not to nvcc itself
      # (nvcc's own -Werror is a different switch with a different surface).
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>)
    # gcc's interprocedural-after-inlining warnings have a long history of
    # false positives — particularly across template-heavy inlining and
    # idiomatic throw-on-assert patterns — that are repeatedly traded
    # across releases (gcc-12/13/14/15 all have outstanding upstream
    # bugzilla PRs in this family). Demote the noisiest of them to plain
    # warnings on gcc so they still surface in build logs but do not
    # gate CI. Clang does not exhibit these and stays under -Werror.
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      set(_ta_gcc_ipa_warnings
          -Wno-error=nonnull
          -Wno-error=stringop-overflow
          -Wno-error=stringop-overread
          -Wno-error=array-bounds
          -Wno-error=dangling-pointer
          -Wno-error=use-after-free
          -Wno-error=restrict
          -Wno-error=maybe-uninitialized)
      foreach(_flag IN LISTS _ta_gcc_ipa_warnings)
        target_compile_options(tiledarray_internal_warnings INTERFACE
          $<$<OR:$<COMPILE_LANGUAGE:CXX>,$<COMPILE_LANGUAGE:C>>:${_flag}>
          $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${_flag}>)
      endforeach()
      unset(_ta_gcc_ipa_warnings)
    endif()
  else()
    message(WARNING "TA_WERROR=ON but compiler '${CMAKE_CXX_COMPILER_ID}' is not in the supported set; ignoring.")
  endif()
endif()
