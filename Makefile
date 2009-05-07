include Make.path

CXX = $(MPICXX)
CXXFLAGS = -g -Wall -fmessage-length=0 -I$(BOOSTDIR) -I./src -I./Tests

OBJS = src/tilemap.o src/env.o TiledArrayTest.o Tests/coordinatestest.o Tests/range1test.o  \
  Tests/rangetest.o Tests/shapetest.o Tests/tiletest.o Tests/arraytest.o

INCDIR = -I$(BOOSTDIR) -I./src -I./Tests
LIBS = -lMADworld

TARGET =	TiledArrayTest

$(TARGET):	$(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LIBDIR) $(LIBS) $(INCDIR) $(DEBUGLEVEL)

all:	$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
