macro(append_flags _flags _append_flag)

  string(REGEX REPLACE "^[ ]+(.*)$" "\\1" _temp_flags "${_append_flag}")
  string(REGEX REPLACE "^(.*)[ ]+$" "\\1" _temp_flags "${_temp_flags}")
  
  set(${_flags} "${${_flags}} ${_temp_flags}")

  string(REGEX REPLACE "^[ ]+(.*)$" "\\1" ${_flags} "${${_flags}}")
  string(REGEX REPLACE "^(.*)[ ]+$" "\\1" ${_flags} "${${_flags}}")

endmacro()