# TiledArray/Makefile

AM_CPPFLAGS = -I$(top_srcdir)/src/

check_PROGRAMS = ta-test
ta_test_SOURCES = TiledArrayTest.cpp \
                  permutation.cpp \
                  coordinates.cpp \
                  coordinate_system.cpp \
                  algorithm.cpp \
                  range.cpp \
                  tiled_range1.cpp \
                  tiled_range.cpp \
                  tile.cpp \
                  variable_list.cpp \
                  array_math.cpp \
                  annotated_array.cpp \
                  variable_list_math.cpp \
                  range_math.cpp \
                  tile_math.cpp \
                  shape_math.cpp \
                  key.cpp \
                  array.cpp \
                  array_math.cpp \
                  expressions.cpp \
                  dense_shape.cpp \
                  sparse_shape.cpp \
                  pred_shape.cpp \
                  shape.cpp \
                  array_fixtures.h \
                  range_fixture.h \
                  shape_fixtures.h \
                  iteration_test.h \
                  global_fixture.h \
                  unit_test_config.h
TESTS = $(check_PROGRAMS)

check_parallel: $(TESTS)
	$(MPIEXEC) -n 4 ./ta-test