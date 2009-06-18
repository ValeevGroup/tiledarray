include Make.path

VPATH = src:Tests
CXX = $(MPICXX)
INCDIR = -I$(MADNESSDIR)/include -I$(BOOSTDIR) -I./src -I./Tests
LIBS = -lMADworld
CXXFLAGS = -g -Wall -fmessage-length=0 $(INCDIR)

OBJS = src/range1.o TiledArrayTest.o Tests/coordinatestest.o Tests/permutationtest.o \
  Tests/range1test.o Tests/rangetest.o Tests/shapetest.o Tests/tiletest.o \
  Tests/arraytest.o Tests/blocktest.o

TARGET =	TiledArrayTest

$(TARGET):	$(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LIBDIR) $(LIBS) $(DEBUGLEVEL)

range1.o: range1.h
coordinatestest.o: coordinates.h coordinate_system.h
permutationtest.o: permutationtest.h permutation.h coordinates.h
range1test.o: range1test.h range1.h
rangetest.o: rangetest.h range.h range1.h array_storage.h
shapetest.o: shapetest.h shape.h predicate.h range.h range1.h
arraytest.o: arraytest.h array.h array_storage.h
tiletest.o: tiletest.h tile.h array_storage.h

all:	$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
