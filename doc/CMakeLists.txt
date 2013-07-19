#-- Add an Option to toggle the generation of the API documentation

find_package(Doxygen)
if(DOXYGEN_FOUND)
  configure_file(Doxyfile.in Doxyfile @ONLY IMMEDIATE)
  add_custom_target (doc
    COMMAND ${DOXYGEN_EXECUTABLE} ${PROJECT_BINARY_DIR}/doc/Doxyfile
    SOURCES ${PROJECT_BINARY_DIR}/doc/Doxyfile)
endif()
