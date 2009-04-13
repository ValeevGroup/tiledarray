#MADNESSROOT = $(HOME)/Development/MADNESS/install
MADNESSDIR = $(HOME)/Development/workspace/MADNESS-install
BOOSTDIR = $(HOME)/Development/boost/trunk

CXX = g++
CXXFLAGS = -g -Wall -fmessage-length=0 -I$(MADNESSDIR)/include -I$(BOOSTDIR) -I./src -I./Tests -DTA_DLEVEL=3 -DTA_WLEVEL=3

OBJS = src/tilemap.o src/env.o TiledArrayTest.o Tests/coordinatestest.o Tests/range1test.o  \
  Tests/rangetest.o Tests/shapetest.o Tests/arraytest.o

LIBDIR = -L$(MADNESSDIR)/lib
LIBS = -lMADworld

TARGET =	TiledArrayTest

$(TARGET):	$(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LIBDIR) $(LIBS) $(INCDIR) $(DEBUGLEVEL)

all:	$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
