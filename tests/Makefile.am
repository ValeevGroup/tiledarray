#
#  This file is a part of TiledArray.
#  Copyright (C) 2013  Virginia Tech
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

# tests/Makefile

include $(top_srcdir)/MakeGlobal.am

AM_CPPFLAGS += $(MADNESS_CPPFLAGS) -DTILEDARRAY_NO_USER_ERROR_MESSAGES=1
LDADD = $(MADNESS_LIB)

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
                  shape.cpp \
                  tiled_range1.cpp \
                  tiled_range.cpp \
                  indexed_iterator.cpp \
                  blocked_pmap.cpp \
                  cyclic_pmap.cpp \
                  hash_pmap.cpp \
                  distributed_storage.cpp \
                  tensor_impl.cpp \
                  array.cpp \
                  eigen.cpp \
                  variable_list.cpp \
                  tile_op_permute.cpp \
                  tile_op_add.cpp \
                  tile_op_scal_add.cpp \
                  tile_op_subt.cpp \
                  tile_op_scal_subt.cpp \
                  tile_op_mult.cpp \
                  tile_op_scal_mult.cpp \
                  tile_op_neg.cpp \
                  tile_op_scal_neg.cpp \
                  tensor_expression_impl.cpp \
                  annotated_tensor.cpp \
                  unary_tensor.cpp \
                  binary_tensor.cpp \
                  reduce_task.cpp \
                  contraction_tensor_impl.cpp \
                  vspgemm.cpp \
                  summa.cpp \
                  contraction_tensor.cpp \
                  expressions.cpp \
                  array_fixture.h \
                  range_fixture.h \
                  iteration_test.h \
                  global_fixture.h \
                  unit_test_config.h

TESTS = $(check_PROGRAMS)

check_parallel: $(TESTS)
	$(MPIEXEC) -n 4 ./ta_test
	
all-local: ta_test