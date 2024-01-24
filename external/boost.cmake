# Boost can be discovered by every (sub)package but only the top package can *build* it ...
# in either case must declare the components used by BTAS
set(required_components
        headers           # TA, BTAS
        algorithm         # TA
        container         # TA, BTAS
        iterator          # TA, BTAS
        random            # TA, BTAS
        tuple             # TA
)
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
