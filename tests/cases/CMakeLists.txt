# hec_* + 4d_e per-cell case binaries (arena vs heap).

set(_cases
  case_hec_h
  case_hec_e
  case_hec_ec
  case_hec_scale
  case_4d_e
)

foreach(_case ${_cases})
  add_ta_executable(${_case} "${_case}.cpp" "tiledarray")
  target_include_directories(${_case} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  add_dependencies(test-cases-tiledarray ${_case})
endforeach()
