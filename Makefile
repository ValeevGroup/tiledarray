include Make.path

CXX = $(MPICXX)
INCDIR = -I$(MADNESSDIR)/include -I$(BOOSTDIR) -I./src -I./Tests
LIBS = -lMADworld
CXXFLAGS = -g -Wall -fmessage-length=0 $(INCDIR)

OBJS = src/range1.o TiledArrayTest.o Tests/coordinatestest.o Tests/permutationtest.o \
  Tests/range1test.o Tests/rangetest.o Tests/shapetest.o Tests/tiletest.o \
  Tests/arraytest.o

TARGET =	TiledArrayTest

$(TARGET):	$(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LIBDIR) $(LIBS) $(DEBUGLEVEL)

src/range1.o: src/range1.h
Tests/coordinatestest.o: Tests/coordinatestest.h src/coordinates.h
Tests/permutationtest.o: Tests/permutationtest.h src/permutation.h src/coordinates.h
Tests/range1test.o: Tests/range1test.h src/range1.h
Tests/rangetest.o: Tests/rangetest.h src/range.h src/range1.h
Tests/shapetest.o: Tests/shapetest.h src/shape.h src/predicate.h src/range.h src/range1.h
Tests/arraytest.o: Tests/arraytest.h src/array.h
Tests/tiletest.o: Tests/tiletest.h src/tile.h

src/madness_runtime.h:
src/coordinate_system.h:
src/permutation.h: 
src/predicate.h: src/permutation.h 
src/iterator.h: src/coordinates.h
src/coordinates.h: src/coordinate_system.h
src/block.h: src/iterator.h
src/array_storage.h: src/coordinate_system.h src/block.h src/madness_runtime.h
src/range1.h:
src/range.h: src/range1.h src/array_storage.h src/coordinates.h
src/shape.h: src/iterator.h
src/tile.h: src/array_storage.h
src/array.h: src/tile.h src/range.h src/shape.h src/iterator.h src/permutation.h
src/local_array.h: src/array.h
src/replcated_array.h: src/array.h
src/distributed_array.h: src/array.h

all:	$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
