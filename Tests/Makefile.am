# TiledArray/Makefile

AM_CPPFLAGS = -I$(top_srcdir)/src/ $(I_MADNESS_DIR) $(I_BOOST_DIR) $(I_EIGEN_DIR) $(I_BLAS_DIR)
AM_CXXFLAGS = $(STDCPP0X)

check_PROGRAMS = ta-test
ta_test_SOURCES = TiledArrayTest.cpp permutation.cpp algorithm.cpp \
  coordinate_system.cpp coordinates.cpp range.cpp tiled_range1.cpp \
  array_dim.cpp dense_array.cpp key.cpp distributed_array.cpp \
  tiled_range.cpp shape.cpp variable_list.cpp tile.cpp \
  tile_slice.cpp packed_tile.cpp array_ref.cpp annotation.cpp annotated_tile.cpp \
  tile_math.cpp array.cpp annotated_array.cpp array_math.cpp \
  array_slice.cpp array_fixtures.h iteration_test.h madness_fixture.h
ta_test_LDFLAGS = $(L_MADNESS_DIR) $(L_BOOST_DIR) $(L_BLAS_DIR)
ta_test_LDADD = $(LIBMADWORLD) $(LIBBLAS)
TESTS = $(check_PROGRAMS)

check_parallel: $(TESTS)
	$(MPIEXEC) -n 4 ./$(TESTS)