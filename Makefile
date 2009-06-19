include Make.path

VPATH = src:Tests
CXX = $(MPICXX)
INCDIR = -I$(MADNESSDIR)/include -I$(BOOSTDIR) -I./src -I./Tests
LIBS = -lMADworld
CXXFLAGS = -g -Wall -fmessage-length=0 $(INCDIR)

OBJS = Tests/permutationtest.o Tests/coordinatestest.o Tests/blocktest.o \
  Tests/range1test.o Tests/rangetest.o Tests/shapetest.o Tests/tiletest.o \
  Tests/arraytest.o TiledArrayTest.o

TARGET =	TiledArrayTest

$(TARGET):	$(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LIBDIR) $(LIBS) $(DEBUGLEVEL)

coordinatestest.o: coordinates.h coordinate_system.h
permutationtest.o: permutationtest.h permutation.h coordinates.h
range1test.o: range1.h
rangetest.o: range.h range1.h array_storage.h
shapetest.o: shape.h predicate.h range.h range1.h
arraytest.o: array.h array_storage.h
tiletest.o: tile.h array_storage.h

all:	$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
