AC_DEFUN([ACX_PROG_MPICXX], [
  AC_ARG_VAR([MPICXX], [MPI C++ compiler])
  if test "x$MPICXX" = x; then
    AC_CHECK_PROGS([MPICXX], [mpicxx mpic++ mpiCC mpCC hcp mpxlC mpxlC_r cmpic++], [$CXX])
  fi
  AC_SUBST(MPICXX)
  acx_prog_mpicxx_CXX="$CXX"
  CXX="$MPICXX"
])
