# ACX_ENABLE_CXXFLAGS_ARG( enable-feature , flag , default arg , default=yes|no )
AC_DEFUN([ACX_ENABLE_CXXFLAGS_ARG], [
  AC_ARG_ENABLE([$1],
    [AC_HELP_STRING([--enable-[$1]@<:@=yes|no|ARG@:>@],
      [Enable $1 compiler flag Ex: $2=$3 @<:@default=$4@:>@.]) ],
    [
      case $enableval in
        yes)
          CXXFLAGS="$CXXFLAGS $2=$3"
        ;;
        no)
        ;;
        *)
          CXXFLAGS="$CXXFLAGS $2=$enableval"
        ;;
      esac
    ], [
      if test "$4" = yes; then
        CXXFLAGS="$CXXFLAGS $2=$3"
      fi 
    ]
  )
])