#
# Converts a list of libraries (second argument, don't forget to enclose the 
# list in quotes) into a list of command-line parameters to the compiler/linker.
#

macro(convert_libs_to_compargs _args _libs )
  # transform library list into compiler args
  foreach (_lib ${_libs})
    get_filename_component(_ext ${_lib} EXT)
    get_filename_component(_libname ${_lib} NAME_WE)
    
    if(APPLE AND "${_ext}" STREQUAL ".framework")

      # Handle Apple Frameworks
      get_filename_component(_path ${_lib} PATH)
      if(${_path} STREQUAL "/System/Library/Frameworks")
        set(MAD_LIBS "${${_args}} -F${_path} -framework ${_libname}")
      else()
        set(MAD_LIBS "${${_args}} -framework ${_libname}")
      endif()

    else()
      
      # Handle the general case
      set(MAD_LIBS "${${_args}} ${_lib}")
    endif()

  endforeach()
endmacro()
