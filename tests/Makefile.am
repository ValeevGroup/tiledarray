# tests/Makefile

AM_CPPFLAGS = -I$(top_srcdir)/src/

check_PROGRAMS = ta_test
ta_test_SOURCES = ta_test.cpp \
                  permutation.cpp \
                  coordinates.cpp \
                  range_iterator.cpp \
                  range.cpp \
                  transform_iterator.cpp \
                  dense_storage.cpp \
                  tensor.cpp \
                  bitset.cpp \
                  tiled_range1.cpp \
                  tiled_range.cpp \
                  indexed_iterator.cpp \
                  blocked_pmap.cpp \
                  cyclic_pmap.cpp \
                  hash_pmap.cpp \
                  distributed_storage.cpp \
                  tensor_impl.cpp \
                  array.cpp \
                  variable_list.cpp \
                  tensor_expression_impl.cpp \
                  annotated_tensor.cpp \
                  unary_tensor.cpp \
                  binary_tensor.cpp \
                  reduce_task.cpp \
                  contraction_tensor_impl.cpp \
                  vspgemm.cpp \
                  summa.cpp \
                  contraction_tensor.cpp \
                  array_fixture.h \
                  range_fixture.h \
                  iteration_test.h \
                  tensor_fixture.h \
                  global_fixture.h \
                  unit_test_config.h
TESTS = $(check_PROGRAMS)

check_parallel: $(TESTS)
	$(MPIEXEC) -n 4 ./ta_test
	
all-local: ta_test