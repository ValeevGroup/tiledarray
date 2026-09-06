include(CheckCXXSourceCompiles)
include(CMakePushCheckState)

macro (detect_MADNESS_configuration)

  cmake_push_check_state()
  set(CMAKE_REQUIRED_QUIET)
  # only extract include dirs, don't use MADworld target directly since it may have not been built yet
  # unfortunately this is not easy to check since the target is defined but not ready
  get_property(_MADNESS_INCLUDE_DIRS TARGET MADworld PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
  # Every entry here is a generator expression (see add_mad_library), but they
  # are not all equal:
  #
  #   $<BUILD_INTERFACE:...> / $<INSTALL_INTERFACE:...>
  #       reference no target, evaluate fine inside a try_compile project, and
  #       carry the directory holding madness/config.h -- the probe below needs
  #       these, so they must be kept.
  #
  #   $<TARGET_PROPERTY:MADmisc,INTERFACE_INCLUDE_DIRECTORIES>
  #       names a MADNESS target. CHECK_CXX_SOURCE_COMPILES runs try_compile(),
  #       which generates a standalone project where no MADNESS target exists,
  #       so this is a hard error there ("Target MADmisc not found") rather than
  #       an include path.
  #
  # MADworld only grew a $<TARGET_PROPERTY:...> entry once it took a dependency
  # on another MAD* library target (MADmisc, in MADNESS 48dc6494b); before that
  # its interface was $<BUILD_INTERFACE>/$<INSTALL_INTERFACE> only, which is why
  # this never bit earlier.
  #
  # So filter on target-referencing genexes specifically, not on "$<". This also
  # supersedes an earlier special case for El -- its entry is a
  # $<TARGET_PROPERTY:El,...> -- and is safer than the substring match on "El"
  # it replaces, which would have dropped a real path containing those letters.
  set(MADNESS_INTERNAL_INCLUDE_DIRS )
  foreach(_inc ${_MADNESS_INCLUDE_DIRS})
    if (NOT (_inc MATCHES "\\$<TARGET_"))
      list(APPEND MADNESS_INTERNAL_INCLUDE_DIRS "${_inc}")
    endif()
  endforeach()
  set(MADNESS_INTERNAL_INCLUDE_DIRS "${MADNESS_INTERNAL_INCLUDE_DIRS}"
          CACHE STRING "Sanitized list of MADNESS include directories usable in build tree")

  list(APPEND CMAKE_REQUIRED_INCLUDES ${MADNESS_INTERNAL_INCLUDE_DIRS})
  if (NOT DEFINED MADNESS_HAS_TBB)
    CHECK_CXX_SOURCE_COMPILES(
        "
    #include <madness/config.h>
    #ifndef HAVE_INTEL_TBB
    # error \"MADNESS does not have TBB\"
    #endif
    int main(int argc, char** argv) {
      return 0;
    }
    "  MADNESS_HAS_TBB)
  endif()

  if (MADNESS_HAS_TBB)
    unset(MADNESS_HAS_TBB)
    set(MADNESS_HAS_TBB ON CACHE BOOL "MADNESS detected usable Intel TBB" FORCE)
  endif()

  unset(CMAKE_REQUIRED_QUIET)
  cmake_pop_check_state()

endmacro (detect_MADNESS_configuration)
