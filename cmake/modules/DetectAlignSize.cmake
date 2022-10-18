# see https://stackoverflow.com/a/69952705 and https://gitlab.kitware.com/cmake/cmake/-/blob/master/Modules/CMakeDetermineCompilerABI.cmake

set(BIN "${CMAKE_PLATFORM_INFO_DIR}/cmake/modules/DetectAlignSize.bin")
try_compile(DETECT_ALIGN_SIZE_COMPILED
      ${CMAKE_BINARY_DIR}
      SOURCES ${PROJECT_SOURCE_DIR}/cmake/modules/DetectAlignSize.cpp
      CMAKE_FLAGS ${CMAKE_CXX_FLAGS}
      COPY_FILE "${BIN}"
      COPY_FILE_ERROR copy_error
      OUTPUT_VARIABLE OUTPUT
      )
if (DETECT_ALIGN_SIZE_COMPILED AND NOT copy_error)
  file(STRINGS "${BIN}" data REGEX "INFO:align_size\\[[^]]*\\]")
  if (data MATCHES "INFO:align_size\\[0*([^]]*)\\]")
     set(TA_ALIGN_SIZE_DETECTED "${CMAKE_MATCH_1}" CACHE INTERNAL "")
  endif()
endif()
