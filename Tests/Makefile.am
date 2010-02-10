# TiledArray/Makefile

AM_CPPFLAGS = -I$(top_srcdir)/src/

check_PROGRAMS = ta-test
ta_test_SOURCES = TiledArrayTest.cpp \
                  permutation.cpp \
                  algorithm.cpp \
                  coordinate_system.cpp \
                  coordinates.cpp \
                  range.cpp \
                  tiled_range1.cpp \
                  array_dim.cpp \
                  dense_array.cpp \
                  key.cpp \
                  distributed_array.cpp \
                  tiled_range.cpp \
                  variable_list.cpp \
                  tile.cpp \
                  tile_slice.cpp \
                  packed_tile.cpp \
                  array_ref.cpp \
                  transform_iterator.cpp \
                  annotation.cpp \
                  annotated_tile.cpp \
                  tile_math.cpp \
                  array.cpp \
                  annotated_array.cpp \
                  array_math.cpp \
                  array_slice.cpp \
                  array_fixtures.h \
                  iteration_test.h \
                  global_fixture.h \
                  unit_test_config.h
TESTS = $(check_PROGRAMS)

check_parallel: $(TESTS)
	$(MPIEXEC) -n 4 ./ta-test