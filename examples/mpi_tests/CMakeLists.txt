# Create the mpi_bandwidth and mpi_latency executable

# Add the mpi_bandwidth executable
add_executable(mpi_bandwidth EXCLUDE_FROM_ALL mpi_bandwidth.c)
set_target_properties(mpi_bandwidth PROPERTIES
    COMPILE_FLAGS "${CMAKE_CPP_FLAGS} ${MPI_C_COMPILE_FLAGS}"
    LINK_FLAGS "${MPI_C_LINK_FLAGS}")
target_link_libraries(mpi_bandwidth ${MPI_C_LIBRARIES})
add_dependencies(example mpi_bandwidth)

# Add the mpi_latency executable
add_executable(mpi_latency EXCLUDE_FROM_ALL mpi_latency.c)
set_target_properties(mpi_latency PROPERTIES
    COMPILE_FLAGS "${CMAKE_CPP_FLAGS} ${MPI_C_COMPILE_FLAGS}"
    LINK_FLAGS "${MPI_C_LINK_FLAGS}")
target_link_libraries(mpi_latency ${MPI_C_LIBRARIES})
add_dependencies(example mpi_latency)
