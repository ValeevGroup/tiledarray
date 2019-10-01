include(CheckCXXSourceCompiles)
include(CMakePushCheckState)

macro (detect_MADNESS_configuration)

  cmake_push_check_state()
  set(CMAKE_REQUIRED_QUIET)
  # only extract include dirs, don't use MADworld since it may have not been built yet
  get_property(MADNESS_INCLUDE_DIRS TARGET MADworld PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
  list(APPEND CMAKE_REQUIRED_INCLUDES ${MADNESS_INCLUDE_DIRS})

  CHECK_CXX_SOURCE_COMPILES(
          "
  #include <madness/config.h>
  #ifndef HAVE_INTEL_MKL
  # error \"MADNESS does not have MKL\"
  #endif
  int main(int argc, char** argv) {
    return 0;
  }
  "  MADNESS_HAS_MKL)

  if (MADNESS_HAS_MKL)
    unset(MADNESS_HAS_MKL)
    set(MADNESS_HAS_MKL ON CACHE BOOL "MADNESS detected usable Intel MKL")
  endif()

  unset(CMAKE_REQUIRED_QUIET)
  cmake_pop_check_state()

endmacro (detect_MADNESS_configuration)
