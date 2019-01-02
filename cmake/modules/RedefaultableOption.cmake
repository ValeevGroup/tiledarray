# if local variable is defined, use its value as the default, otherwise use _default
# this is consistent with cmake 3.13 and later (see policy CMP0077)
macro(redefaultable_option _name _descr _default)

  if (${_name})
    set(${_name}_DEFAULT ${${_name}})
  else()
    set(${_name}_DEFAULT ${${_default}})
  endif()
  option(${_name} ${_descr} ${${_name}_DEFAULT})

endmacro()