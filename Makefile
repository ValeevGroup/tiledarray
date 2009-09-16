include Make.path

VPATH = src:Tests
INCDIR = -I$(MADNESSDIR)/include -I$(BOOSTDIR) -I./src -I./Tests -I$(BLASINCLUDEDIR) -I$(EIGENDIR)
LIBDIR = -L$(MADNESSDIR)/lib
LIBS = -lMADworld -lboost_unit_test_framework
ifeq ($(CXX),icpc)
	WARNING = -wd981 -wd383 -wd1419 -wd444
endif
CXXFLAGS = -g -Wall -fmessage-length=0 $(INCDIR) -DTA_EXCEPTION_ERROR -std=gnu++0x $(WARNING)
CXXSUF = cpp
OBJSUF = o
CXXDEPEND = $(CXX)
CXXDEPENDSUF = none
CXXDEPENDFLAGS = -M
CXX = $(MPICXX)
TESTSRC = TiledArrayTest.cpp permutationtest.cpp coordinatestest.cpp rangetest.cpp \
	tiledrange1test.cpp arraystoragetest.cpp tiledrangetest.cpp shapetest.cpp \
	variablelisttest.cpp tiletest.cpp tileslicetest.cpp packedtiletest.cpp \
	annotatedtiletest.cpp tilemathtest.cpp arraytest.cpp
OBJS = $(TESTSRC:%.cpp=%.$(OBJSUF))

TARGET =	TiledArrayTest

$(TARGET):	$(OBJS)
	$(MPICXX) -o $(TARGET) $(OBJS) $(LIBDIR) $(LIBS) $(DEBUGLEVEL)
#	./TiledArrayTest

all:	$(TARGET)

check:
	./TiledArrayTest --log_level=test_suite

.PHONY: clean dclean
clean:
	-rm -f $(OBJS) $(TARGET)

dclean:
	-rm -f *.d

ifneq ($(CXXDEPENDSUF),none)
%.d: %.$(CXXSUF)
	$(CXXDEPEND) $(CXXDEPENDFLAGS) -c $(CPPFLAGS) $(CXXFLAGS) $< > /dev/null
	sed 's/^$*.o/$*.$(OBJSUF) $*.d/g' < $(*F).$(CXXDEPENDSUF) > $(@F)
	/bin/rm -f $(*F).$(CXXDEPENDSUF)
else
%.d: %.$(CXXSUF)
	$(CXXDEPEND) $(CXXDEPENDFLAGS) -c $(CPPFLAGS) $(CXXFLAGS) $< | sed 's/^$*.o/$*.$(OBJSUF) $*.d/g' > $(@F)
endif

ifneq ($(DODEPEND),no)
include $(OBJS:%.o=%.d)
endif
	