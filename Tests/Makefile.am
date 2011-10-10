# TiledArray/Makefile

AM_CPPFLAGS = -I$(top_srcdir)/src/

check_PROGRAMS = ta-test
ta_test_SOURCES = TiledArrayTest.cpp \
                  permutation.cpp \
                  coordinates.cpp \
                  coordinate_system.cpp \
                  range_iterator.cpp \
                  range.cpp \
                  tiled_range1.cpp \
                  tiled_range.cpp \
                  transform_iterator.cpp \
                  dense_storage.cpp \
                  distributed_storage.cpp \
                  tensor.cpp \
                  unary_tensor.cpp \
                  binary_tensor.cpp \
                  permute_tensor.cpp \
                  contraction_tensor.cpp \
                  bitset.cpp \
                  versioned_pmap.cpp \
                  variable_list.cpp \
                  annotated_array.cpp \
                  indexed_iterator.cpp \
                  array.cpp \
                  reduce_task.cpp \
                  array_math.cpp \
                  expressions.cpp \
                  array_fixture.h \
                  range_fixture.h \
                  shape_fixtures.h \
                  iteration_test.h \
                  global_fixture.h \
                  unit_test_config.h
TESTS = $(check_PROGRAMS)

check_parallel: $(TESTS)
	$(MPIEXEC) -n 4 ./ta-test