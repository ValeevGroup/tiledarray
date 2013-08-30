#
# converts a list of libraries (second argument, don't forget to enclose the list in quotes)
# into a list of command-line parameters to the compiler/linker. The main purpose is to
# handle Apple's frameworks.
#

macro(convert_libs_to_compargs _args _libs )
  # transform library list into compiler args
  set(_framework_paths)
  set(_framework_names)
  set(_lib_paths)
  set(_lib_names)
  foreach (_lib ${_libs})
    # Extract the components of the full library path
    get_filename_component(_path ${_lib} PATH)
    get_filename_component(_ext ${_lib} EXT)
    get_filename_component(_libname ${_lib} NAME_WE)

    if(APPLE AND "${_ext}" STREQUAL ".framework")
      list(APPEND _framework_paths ${_path})
      list(APPEND _framework_names ${_libname})
    else()
      set(_name ${_libname})
      foreach(_prefix ${CMAKE_FIND_LIBRARY_PREFIXES})
        if(_libname MATCHES "^${_prefix}")
          string(REGEX REPLACE "^${_prefix}(.*)$" "\\1" _name ${_libname})
          break()
        endif()
      endforeach()

      list(APPEND _lib_paths ${_path})
      list(APPEND _lib_names ${_name})
    endif()
  endforeach()

  if(APPLE)
    # Add Framework paths to _args
    list(LENGTH _framework_paths _num)
    if(_num GREATER 1)
      list(REMOVE_DUPLICATES _framework_paths)
    endif()
    foreach(_framework_path ${_framework_paths})
      set(${_args} "${${_args}} -F${_framework_path}")
    endforeach()

    # Add Frameworks to _args
    list(LENGTH _framework_names _num)
    if(_num GREATER 1)
      list(REMOVE_DUPLICATES _framework_names)
    endif()
    foreach(_framework_name ${_framework_names})
      set(${_args} "${${_args}} -framework ${_framework_name}")
    endforeach()
  endif()

  # Add library paths to _args
  list(LENGTH _lib_paths _num)
  if(_num GREATER 1)
    list(REMOVE_DUPLICATES _lib_paths)
  endif()
  foreach(_lib_path ${_lib_paths})
    set(${_args} "${${_args}} -L${_lib_path}")
  endforeach()

  # Add libraries to _args
  foreach(_lib_name ${_lib_names})
    set(${_args} "${${_args}} -l${_lib_name}")
  endforeach()
endmacro()
