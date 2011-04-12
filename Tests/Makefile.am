# TiledArray/Makefile

AM_CPPFLAGS = -I$(top_srcdir)/src/

check_PROGRAMS = ta-test
ta_test_SOURCES = TiledArrayTest.cpp \
                  permutation.cpp \
                  coordinates.cpp \
                  key.cpp \
                  coordinate_system.cpp \
                  algorithm.cpp \
                  range_iterator.cpp \
                  range.cpp \
                  tiled_range1.cpp \
                  tiled_range.cpp \
                  tile.cpp \
                  bitset.cpp \
                  versioned_pmap.cpp \
                  dense_shape.cpp \
                  sparse_shape.cpp \
                  pred_shape.cpp \
                  shape.cpp \
                  variable_list.cpp \
                  annotated_array.cpp \
                  indexed_iterator.cpp \
                  array.cpp \
                  math.cpp \
                  variable_list_math.cpp \
                  range_math.cpp \
                  tiled_range_math.cpp \
                  tile_math.cpp \
                  shape_math.cpp \
                  array_math.cpp \
                  expressions.cpp \
                  array_fixtures.h \
                  range_fixture.h \
                  shape_fixtures.h \
                  iteration_test.h \
                  global_fixture.h \
                  unit_test_config.h
TESTS = $(check_PROGRAMS)

check_parallel: $(TESTS)
	$(MPIEXEC) -n 4 ./ta-test