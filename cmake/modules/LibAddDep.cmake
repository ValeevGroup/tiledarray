# Add library dependencies to a list if they are missing


macro(lib_add_dep _lib_list _lib _deps)
  
  list(FIND ${_lib_list} ${_lib} _lib_find)
  if(NOT _lib_find EQUAL -1)
    foreach(_dep ${_deps})
      list(FIND ${_lib_list} ${_dep} _dep_find)
      if(_dep_find EQUAL -1)
        list(APPEND ${_lib_list} ${_dep})
      endif()
    endforeach()
  endif()

endmacro()