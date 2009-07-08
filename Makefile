include Make.path

VPATH = src:Tests
CXX = $(MPICXX)
INCDIR = -I$(MADNESSDIR)/include -I$(BOOSTDIR) -I./src -I./Tests
LIBDIR = 
LIBS = -lMADworld -lcblas -lblas
CXXFLAGS = -g -Wall -fmessage-length=0 $(INCDIR) -DTA_EXCEPTION_ERROR

OBJS = Tests/permutationtest.o Tests/coordinatestest.o Tests/rangetest.o \
  Tests/tiledrange1test.o Tests/arraystoragetest.o Tests/tiledrangetest.o Tests/shapetest.o \
  Tests/tiletest.o Tests/arraytest.o TiledArrayTest.o

TARGET =	TiledArrayTest

$(TARGET):	$(OBJS)
	$(CXX) -v -o $(TARGET) $(OBJS) $(LIBDIR) $(LIBS) $(DEBUGLEVEL)
	./TiledArrayTest

permutationtest.o: src/permutation.h src/coordinates.h
coordinatestest.o: src/coordinates.h src/coordinate_system.h src/permutation.h
rangetest.o: src/range.h src/permutation.h src/coordinates.h src/iterator.h src/error.h
tiledrange1test.o: src/tiled_range1.h src/range.h src/coordinates.h src/error.h
arraystoragetest.o: src/range.h src/array_storage.h src/error.h
tiledrangetest.o: src/tiled_range.h src/tiled_range1.h src/array_storage.h
shapetest.o: src/shape.h src/predicate.h src/tiled_range.h src/tiled_range1.h
arraytest.o: src/array.h src/array_storage.h
tiletest.o: src/tile.h src/array_storage.h

src/array_storage.h:
src/array.h:
src/range.h:
src/coordinate_system.h:
src/iterator.h:
src/madness_runtime.h:
src/permutation.h:
src/predicate.h:
src/tiled_range.h:
src/tiled_range1.h:
src/shape.h:
src/tile.h:

all:	$(TARGET)

check:
	./TiledArrayTest --log_level=test_suite

check_permutation:
	./TiledArrayTest --log_level=test_suite --run_test=permutation_suite

check_array_coordinate:
	./TiledArrayTest --log_level=test_suite --run_test=array_coordinate_suite

check_block:
	./TiledArrayTest --log_level=test_suite --run_test=block_suite

check_range1:
	./TiledArrayTest --log_level=test_suite --run_test=range1_suite

.PHONY: clean
clean:
	rm -f $(OBJS) $(TARGET)
