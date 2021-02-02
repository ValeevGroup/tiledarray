set(
  keys

  OS_NAME
  OS_RELEASE
  OS_VERSION
  OS_PLATFORM

  
  PROCESSOR_DESCRIPTION
  NUMBER_OF_PHYSICAL_CORES

  TOTAL_VIRTUAL_MEMORY
  AVAILABLE_VIRTUAL_MEMORY
  TOTAL_PHYSICAL_MEMORY
  AVAILABLE_PHYSICAL_MEMORY
  )

foreach (key ${keys})
  cmake_host_system_information(RESULT result QUERY ${key})
  message(STATUS "${key}: ${result}")
endforeach()
