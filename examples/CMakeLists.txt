# Create example programs target so all examples programs
add_custom_target(example)

# Add Subdirectories
add_subdirectory (cc)
add_subdirectory (dgemm)
add_subdirectory (mpi_tests)
add_subdirectory (pmap_test)
add_subdirectory (vector_tests)
