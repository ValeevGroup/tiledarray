#
# converts a list of include paths (second argument, don't forget to enclose the
# list in quotes) into a list of command-line parameters to the compiler/.
#

macro(convert_incs_to_compargs _args _inc_paths )
  # transform library list into compiler args

  # Add include paths to _args
  foreach(_inc_path ${_inc_paths})
    set(${_args} "${${_args}} -I${_inc_path}")
  endforeach()
endmacro()
