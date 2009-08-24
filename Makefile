include Make.path

VPATH = src:Tests
CXX = $(MPICXX)
INCDIR = -I$(MADNESSDIR)/include -I$(BOOSTDIR) -I./src -I./Tests -I$(BLASINCLUDEDIR) -I$(EIGENDIR)
LIBDIR = -L$(MADNESSDIR)/lib
LIBS = -lMADworld -lcblas -lblas
CXXFLAGS = -g -Wall -fmessage-length=0 $(INCDIR) -DTA_EXCEPTION_ERROR -std=gnu++0x
CXXSUF = cpp
OBJSUF = o
CXXDEPEND = $(CXX)
CXXDEPENDSUF = none
CXXDEPENDFLAGS = -M

TESTSRC = permutationtest.cpp coordinatestest.cpp rangetest.cpp \
 	tiledrange1test.cpp arraystoragetest.cpp tiledrangetest.cpp shapetest.cpp \
	variablelisttest.cpp tiletest.cpp tileslicetest.cpp packedtiletest.cpp \
	annotatedtiletest.cpp tilemathtest.cpp arraytest.cpp TiledArrayTest.cpp
OBJS = $(TESTSRC:%.cpp=%.$(OBJSUF))

TARGET =	TiledArrayTest

$(TARGET):	$(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LIBDIR) $(LIBS) $(DEBUGLEVEL)
#	./TiledArrayTest

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
	