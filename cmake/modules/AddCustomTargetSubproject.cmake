# Copyright 2020 Eduard F Valeyev
# Distributed under the OSI-approved BSD 3-Clause License.
# See https://opensource.org/licenses/BSD-3-Clause for details.

# copy of https://github.com/BTAS/BTAS/blob/master/cmake/modules/AddCustomTargetSubproject.cmake
#
# add_custom_target_subproject(proj X ...) defines custom target X-proj and
# - if target X already exists, makes it depend on X-proj
# - else creates target X depending on X-proj
#
# use case: if custom target names (e.g. "check", "doc", etc.) clash
#           with other project's target when used as a subproject
#
# example: add_custom_target_subproject(myproject check USES_TERMINAL COMMAND ${CMAKE_CTEST_COMMAND} -V)
#

macro(add_custom_target_subproject _subproj _name)

  set(extra_args "${ARGN}")
  add_custom_target(${_name}-${_subproj} ${extra_args})

  # does the newly-created target get compiled by default?
  list(FIND extra_args "ALL" extra_args_has_all)
  if (NOT (extra_args_has_all EQUAL -1))
    set (target_built_by_default ON)
  endif()

  if (TARGET ${_name})
    # is existing target ${_name} also compiled by default?
    # warn if not, but this project's target is since that
    # may indicate inconsistent creation of generic targets
    get_target_property(supertarget_not_built_by_default ${_name} EXCLUDE_FROM_ALL)
    if (target_built_by_default AND supertarget_not_built_by_default)
      message(WARNING "Created target ${_name}-${_subproj} is built by default but \"super\"-target ${_name} is not; perhaps it should be?")
    endif()
    add_dependencies(${_name} ${_name}-${_subproj})
  else (TARGET ${_name})
    # use ALL if given
    if (target_built_by_default)
      add_custom_target(${_name} ALL DEPENDS ${_name}-${_subproj})
    else (target_built_by_default)
      add_custom_target(${_name} DEPENDS ${_name}-${_subproj})
    endif(target_built_by_default)
  endif (TARGET ${_name})

endmacro()
