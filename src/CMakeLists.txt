# library headers
file(GLOB_RECURSE headers RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} TiledArray/*.h)
foreach (_file ${headers})
  get_filename_component(_path ${_file} PATH)
  # message(STATUS "gonna install ${_file} in include/${_path}" )
  install(FILES ${_file} DESTINATION include/${_path})
endforeach()
install(FILES tiled_array.h DESTINATION include)