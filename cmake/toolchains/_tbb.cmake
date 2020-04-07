############## Only usable on UNIX platforms
if (NOT UNIX)
  message(FATAL_ERROR "intel-tbb.cmake is only usable on UNIX platforms")
endif(NOT UNIX)

############## Top tools dir
if (DEFINED ENV{INTEL_DIR})
  set(_intel_dir $ENV{INTEL_DIR})
else(DEFINED ENV{INTEL_DIR})
  set(_intel_dir /opt/intel)
endif()
set(INTEL_DIR "${_intel_dir}" CACHE PATH "Intel tools root directory")

############## TBB
if (DEFINED ENV{TBBROOT})
  set(_tbb_root_dir "$ENV{TBBROOT}")
else(DEFINED ENV{TBBROOT})
  set(_tbb_root_dir "${INTEL_DIR}/tbb")
endif(DEFINED ENV{TBBROOT})
set(TBB_ROOT_DIR "${_tbb_root_dir}" CACHE PATH "Intel TBB root directory")
