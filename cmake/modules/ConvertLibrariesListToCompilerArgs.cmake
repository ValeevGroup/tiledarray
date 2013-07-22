#
# converts a list of libraries (second argument, don't forget to enclose the list in quotes)
# into a list of command-line parameters to the compiler/linker. The main purpose is to
# handle Apple's frameworks.
#

macro(convert_libs_to_compargs _args _libs )
 # transform library list into compiler args
 foreach (_lib ${_libs})
  set(_libpath _libpath-NOTFOUND) # DON'T REMOVE THIS LINE
  find_library(_libpath ${_lib})
  #message(STATUS "_lib = ${_lib}, _libpath = ${_libpath}")
  if (_libpath)
    set(_library "${_libpath}")
  else()
    set(_library "${_lib}")
  endif()
  #message(STATUS "_library = ${_library}")
  
  # Apple framework libs
  if (APPLE)
   get_filename_component(_ext ${_library} EXT)
   if ("${_ext}" STREQUAL ".framework")
    get_filename_component(_name ${_library} NAME_WE)
    get_filename_component(_path ${_library} PATH)
    #message(STATUS "${_library} = ${_name} ${_path} ${_ext}")
    set(_library "-F ${_path} -framework ${_name}")
   endif()
   #message(STATUS "${_lib} => ${_library}")
   set(${_args} "${${_args}} ${_library}")
  endif() # APPLE only
  
 endforeach()
endmacro()
