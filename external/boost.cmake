# -*- mode: cmake -*-

# update the Boost version that we can tolerate
if (NOT DEFINED Boost_OLDEST_BOOST_VERSION)
    set(Boost_OLDEST_BOOST_VERSION ${TA_OLDEST_BOOST_VERSION})
else()
    if (${Boost_OLDEST_BOOST_VERSION} VERSION_LESS ${TA_OLDEST_BOOST_VERSION})
        if (DEFINED CACHE{Boost_OLDEST_BOOST_VERSION})
            set(Boost_OLDEST_BOOST_VERSION "${TA_OLDEST_BOOST_VERSION}" CACHE STRING "Oldest Boost version to use" FORCE)
        else()
            set(Boost_OLDEST_BOOST_VERSION ${TA_OLDEST_BOOST_VERSION})
        endif()
    endif()
endif()

# Boost can be discovered by every (sub)package but only the top package can *build* it ...
# in either case must declare the components used by TA
set(required_components
        headers
        algorithm
        container
        iterator
        random
        tuple
)
if (BUILD_TESTING)
    list(APPEND required_components
            test
    )
endif()
if (DEFINED Boost_REQUIRED_COMPONENTS)
    list(APPEND Boost_REQUIRED_COMPONENTS
            ${required_components})
    list(REMOVE_DUPLICATES Boost_REQUIRED_COMPONENTS)
else()
    set(Boost_REQUIRED_COMPONENTS "${required_components}" CACHE STRING "Components of Boost to discovered or built")
endif()
set(optional_components
        serialization     # BTAS
)
if (DEFINED Boost_OPTIONAL_COMPONENTS)
    list(APPEND Boost_OPTIONAL_COMPONENTS
            ${optional_components}
    )
    list(REMOVE_DUPLICATES Boost_OPTIONAL_COMPONENTS)
else()
    set(Boost_OPTIONAL_COMPONENTS "${optional_components}" CACHE STRING "Optional components of Boost to discovered or built")
endif()

if (NOT DEFINED Boost_FETCH_IF_MISSING)
    set(Boost_FETCH_IF_MISSING 1)
endif()

include(${vg_cmake_kit_SOURCE_DIR}/modules/FindOrFetchBoost.cmake)
