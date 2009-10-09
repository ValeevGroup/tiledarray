include Make.path

VPATH = src:Tests
INCDIR = -I$(MADNESSDIR)/include -I$(BOOSTDIR)/include -I./src -I./Tests -I$(BLASINCLUDEDIR) -I$(EIGENDIR)
LIBDIR = -L$(MADNESSDIR)/lib -L$(BOOSTDIR)/lib
#LIBS = -lMADworld
#LIBS = -lMADworld
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
TESTSRC = TiledArrayTest.cpp permutationtest.cpp algorithmtest.cpp coordinatestest.cpp \
	rangetest.cpp tiledrange1test.cpp arraystoragetest.cpp tiledrangetest.cpp \
	shapetest.cpp variablelisttest.cpp tiletest.cpp tileslicetest.cpp packedtiletest.cpp \
	annotatedtiletest.cpp tilemathtest.cpp arraytest.cpp 
OBJS = $(TESTSRC:%.cpp=%.$(OBJSUF))

TARGET =	TiledArrayTest

$(TARGET):	$(OBJS)
	$(MPICXX) -o $(TARGET) $(OBJS) $(LIBDIR) $(LIBS) $(DEBUGLEVEL)
#	./TiledArrayTest

all:	$(TARGET)

check: check_serial check_parallel

check_serial: $(TARGET)
	./TiledArrayTest --run_test="permutation_suite,algorithm_suite,array_coordinate_suite,range_suite,range1_suite,array_dim_suite,dense_storage_suite,tiled_range_suite,shape_suite,variable_list_suite,tile_suite,tile_slice_suite,packed_tile_suite,annotated_tile_suite,tile_math_suite"
	
check_parallel: $(TARGET)
	./TiledArrayTest --run_test="distributed_storage_suite"
	mpiexec -n 4 ./TiledArrayTest --run_test="distributed_storage_suite"

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
	