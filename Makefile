include Make.path

VPATH = src:Tests
CXX = $(MPICXX)
INCDIR = -I$(MADNESSDIR)/include -I$(BOOSTDIR) -I./src -I./Tests -DTA_EXCEPTION_ERROR
LIBS = -lMADworld
CXXFLAGS = -g -Wall -fmessage-length=0 $(INCDIR)

OBJS = Tests/permutationtest.o Tests/coordinatestest.o Tests/blocktest.o \
  Tests/range1test.o Tests/rangetest.o Tests/shapetest.o Tests/tiletest.o \
  Tests/arraytest.o TiledArrayTest.o

TARGET =	TiledArrayTest

$(TARGET):	$(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LIBDIR) $(LIBS) $(DEBUGLEVEL)
	./TiledArrayTest

permutationtest.o: permutation.h coordinates.h
coordinatestest.o: coordinates.h coordinate_system.h permutation.h
blocktest.o: block.h permutation.h coordinates.h iterator.h
range1test.o: range1.h block.h coordinates.h
rangetest.o: range.h range1.h array_storage.h
shapetest.o: shape.h predicate.h range.h range1.h
arraytest.o: array.h array_storage.h
tiletest.o: tile.h array_storage.h

array_storage.h:
array.h:
block.h:
coordinate_system.h:
iterator.h:
madness_runtime.h:
permutation.h:
predicate.h:
range.h:
range1.h:
shape.h:
tile.h:

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

clean:
	rm -f $(OBJS) $(TARGET)
