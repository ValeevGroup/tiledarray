macro(add_ta_executable _name _source_files _libs)

  add_executable(${_name} EXCLUDE_FROM_ALL "${_source_files}")
  target_link_libraries(${_name} PRIVATE "${_libs}")

endmacro()
