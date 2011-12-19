CPPFLAGS = -I/Users/justus/Developer/src/tiledarray/src -I/Users/justus/Developer/install/madness_apple_clang/include
CXXFLAGS = -g -Wall -fmessage-length=0

OBJS =		CCD.o

LDFLAGS = -L/Users/justus/Developer/install/madness_apple_clang/lib
LIBS = -lMADworld

TARGET =	CCD

$(TARGET):	$(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LDFLAGS) $(LIBS)

all:	$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
