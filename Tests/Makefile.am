# TiledArray/Makefile

AM_CPPFLAGS = -I$(top_srcdir)/src/

check_PROGRAMS = ta-test
ta_test_SOURCES = TiledArrayTest.cpp \
                  permutation.cpp \
                  coordinates.cpp \
                  range_iterator.cpp \
                  range.cpp \
                  transform_iterator.cpp \
                  dense_storage.cpp \
                  tensor.cpp \
                  unary_tensor.cpp \
                  binary_tensor.cpp \
                  permute_tensor.cpp \
                  contraction_tensor.cpp \
                  bitset.cpp \
                  tiled_range1.cpp \
                  tiled_range.cpp \
                  indexed_iterator.cpp \
                  distributed_storage.cpp \
                  array.cpp \
                  variable_list.cpp \
                  annotated_array.cpp \
                  unary_tiled_tensor.cpp \
                  binary_tiled_tensor.cpp \
                  reduce_task.cpp \
                  contraction_tiled_tensor.cpp \
                  permute_tiled_tensor.cpp \
                  expressions.cpp \
                  array_fixture.h \
                  range_fixture.h \
                  iteration_test.h \
                  tensor_fixture.h \
                  global_fixture.h \
                  unit_test_config.h
TESTS = $(check_PROGRAMS)

check_parallel: $(TESTS)
	$(MPIEXEC) -n 4 ./ta-test