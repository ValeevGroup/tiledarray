# Determines how the unit tests must use Boost.Test.
#
# ta_boost_test_is_shared(<target> <result-var>)
#
# Sets <result-var> to TRUE if linking <target> (Boost::unit_test_framework)
# links a shared Boost.Test library, FALSE otherwise (a static library, or a
# target that only carries usage requirements).
#
# The answer decides which variant of the framework the tests include (see
# unit_test_config.h): with a shared library they must use it
# (BOOST_TEST_DYN_LINK); otherwise they include the header-only variant.
# Including the header-only variant while linking the shared library defines
# the framework's global objects twice, in the executable and in the library,
# and they are then destroyed twice at exit.
function(ta_boost_test_is_shared _target _result)
  set(${_result} FALSE PARENT_SCOPE)
  if (NOT TARGET ${_target})
    return()
  endif()

  # look through an alias
  get_target_property(_aliased ${_target} ALIASED_TARGET)
  if (_aliased)
    set(_target ${_aliased})
  endif()

  get_target_property(_type ${_target} TYPE)
  if (_type STREQUAL "SHARED_LIBRARY")
    set(${_result} TRUE PARENT_SCOPE)
  elseif (_type STREQUAL "UNKNOWN_LIBRARY")
    # an imported library of unstated kind: go by the file it names
    set(_props IMPORTED_LOCATION)
    get_target_property(_configs ${_target} IMPORTED_CONFIGURATIONS)
    if (_configs)
      foreach (_config IN LISTS _configs)
        list(APPEND _props IMPORTED_LOCATION_${_config})
      endforeach()
    endif()
    foreach (_prop IN LISTS _props)
      get_target_property(_location ${_target} ${_prop})
      if (_location MATCHES "\\.(so|dylib)(\\.[0-9.]+)?$")
        set(${_result} TRUE PARENT_SCOPE)
      endif()
    endforeach()
  elseif (_type STREQUAL "INTERFACE_LIBRARY")
    # a target that only forwards to others: shared if any of them is
    get_target_property(_deps ${_target} INTERFACE_LINK_LIBRARIES)
    if (_deps)
      foreach (_dep IN LISTS _deps)
        if (TARGET ${_dep})
          ta_boost_test_is_shared(${_dep} _dep_is_shared)
          if (_dep_is_shared)
            set(${_result} TRUE PARENT_SCOPE)
          endif()
        endif()
      endforeach()
    endif()
  endif()
endfunction()
