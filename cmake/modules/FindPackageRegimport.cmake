# find_package and register for re-import and superproject
macro(find_package_regimport _package)
  find_package(${_package} ${ARGN})
  file(APPEND "${PROJECT_BINARY_DIR}/cmake/modules/ReimportTargets.cmake" "find_package(${_package} \"${ARGN}\")\n")
endmacro()

macro(init_package_regimport)
  file(WRITE "${PROJECT_BINARY_DIR}/cmake/modules/ReimportTargets.cmake" "# load this in superproject of TiledArray to re-import the targets imported during its build\n")
endmacro()
