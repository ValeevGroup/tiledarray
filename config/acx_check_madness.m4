AC_DEFUN([ACX_CHECK_MADNESS], [
  AC_CHECK_HEADER([world/world.h], [], [
    AC_MSG_ERROR([Unable to find the required M-A-D-N-E-S-S header file.])
  ])
  
  ACX_CXX_CHECK_LIB([MADworld], [madness::initialize(int argc = 0, char** argv = 0)],
    [LIBS="$LIBS -lMADworld"], [AC_MSG_ERROR([The required library MADworld could not be found.])])
])