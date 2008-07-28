CXX = /usr/local/bin/g++
CXXFLAGS = -g -Wall -fmessage-length=0 -I./src -I./Tests -DTA_DLEVEL=3 -DTA_WLEVEL=3

OBJS =		./Tests/tupletest.o ./Tests/triplettest.o ./Tests/shapetest.o TiledArrayTest.o

LIBS =

TARGET =	TiledArrayTest

$(TARGET):	$(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LIBDIR) $(LIBS) $(INCDIR) $(DEBUGLEVEL)

all:	$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
